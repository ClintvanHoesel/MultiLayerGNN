#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace
{

    constexpr int kThreads = 256;

    __device__ __forceinline__ float minimum_image(float delta, float box)
    {
        return delta - roundf(delta / box) * box;
    }

    __device__ __forceinline__ float pbc_distance2(
        const float3 xi,
        const float3 xj,
        const float box0,
        const float box1,
        const float box2)
    {
        const float dx = minimum_image(xi.x - xj.x, box0);
        const float dy = minimum_image(xi.y - xj.y, box1);
        const float dz = minimum_image(xi.z - xj.z, box2);
        return dx * dx + dy * dy + dz * dz;
    }

    // One thread owns one source atom. Cooperative tiling lets a block reuse each
    // target position across all of its source atoms rather than loading it once per
    // source/target pair from global memory.
    __global__ void count_edges_kernel(
        const float3 *__restrict__ pos,
        int64_t N,
        const float *__restrict__ box,
        float r2,
        bool loop,
        int64_t *__restrict__ counts)
    {
        extern __shared__ float3 target_tile[];

        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const bool active = i < N;
        const float3 xi = active ? pos[i] : make_float3(0.0f, 0.0f, 0.0f);
        const float box0 = box[0];
        const float box1 = box[1];
        const float box2 = box[2];
        int64_t count = 0;

        for (int64_t tile_begin = 0; tile_begin < N; tile_begin += blockDim.x)
        {
            const int64_t j = tile_begin + threadIdx.x;
            if (j < N)
            {
                target_tile[threadIdx.x] = pos[j];
            }
            __syncthreads();

            const int64_t remaining = N - tile_begin;
            const int tile_size = static_cast<int>(
                remaining < blockDim.x ? remaining : blockDim.x);
            if (active)
            {
                for (int local_j = 0; local_j < tile_size; ++local_j)
                {
                    const int64_t global_j = tile_begin + local_j;
                    if ((loop || i != global_j) &&
                        pbc_distance2(xi, target_tile[local_j], box0, box1, box2) <= r2)
                    {
                        ++count;
                    }
                }
            }
            __syncthreads();
        }

        if (active)
        {
            counts[i] = count;
        }
    }

    // StoreDistance=true is used when capped-neighbor selection needs the squared
    // distances. The unrestricted path writes edge_index directly and therefore
    // avoids all scratch allocations and the selection kernel.
    template <bool StoreDistance>
    __global__ void collect_edges_kernel(
        const float3 *__restrict__ pos,
        int64_t N,
        const float *__restrict__ box,
        float r2,
        bool loop,
        const int64_t *__restrict__ starts,
        float *__restrict__ out_dist,
        int64_t *__restrict__ out_rows,
        int64_t *__restrict__ out_cols)
    {
        extern __shared__ float3 target_tile[];

        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const bool active = i < N;
        const float3 xi = active ? pos[i] : make_float3(0.0f, 0.0f, 0.0f);
        const float box0 = box[0];
        const float box1 = box[1];
        const float box2 = box[2];
        int64_t write_index = active ? starts[i] : 0;

        for (int64_t tile_begin = 0; tile_begin < N; tile_begin += blockDim.x)
        {
            const int64_t j = tile_begin + threadIdx.x;
            if (j < N)
            {
                target_tile[threadIdx.x] = pos[j];
            }
            __syncthreads();

            const int64_t remaining = N - tile_begin;
            const int tile_size = static_cast<int>(
                remaining < blockDim.x ? remaining : blockDim.x);
            if (active)
            {
                for (int local_j = 0; local_j < tile_size; ++local_j)
                {
                    const int64_t global_j = tile_begin + local_j;
                    const float dist2 = pbc_distance2(
                        xi, target_tile[local_j], box0, box1, box2);
                    if ((loop || i != global_j) && dist2 <= r2)
                    {
                        if constexpr (StoreDistance)
                        {
                            out_dist[write_index] = dist2;
                        }
                        else
                        {
                            out_rows[write_index] = i;
                        }
                        out_cols[write_index++] = global_j;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Selects the closest max_k neighbors for each source atom. This runs only on
    // the capped path; the common unlimited path bypasses it entirely.
    __global__ void select_k_edges_kernel(
        int64_t N,
        int64_t max_k,
        const int64_t *__restrict__ full_starts,
        const int64_t *__restrict__ capped_starts,
        const float *__restrict__ in_dist,
        const int64_t *__restrict__ in_col,
        unsigned char *__restrict__ chosen,
        int64_t *__restrict__ out_rows,
        int64_t *__restrict__ out_cols)
    {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= N)
        {
            return;
        }

        const int64_t begin = full_starts[i];
        const int64_t count = full_starts[i + 1] - begin;
        const int64_t k = max_k < count ? max_k : count;
        int64_t out_pos = capped_starts[i];

        for (int64_t m = 0; m < count; ++m)
        {
            chosen[begin + m] = 0;
        }

        for (int64_t selected = 0; selected < k; ++selected)
        {
            float best_distance = CUDART_INF_F;
            int64_t best_offset = -1;
            for (int64_t m = 0; m < count; ++m)
            {
                if (!chosen[begin + m] && in_dist[begin + m] < best_distance)
                {
                    best_distance = in_dist[begin + m];
                    best_offset = m;
                }
            }
            chosen[begin + best_offset] = 1;
            out_rows[out_pos] = i;
            out_cols[out_pos++] = in_col[begin + best_offset];
        }
    }

} // namespace

torch::Tensor radius_graph_pbc_cuda(
    torch::Tensor pos,
    double r,
    torch::Tensor lattice,
    bool loop,
    int64_t max_num_neighbors)
{
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(lattice.is_cuda(), "lattice must be a CUDA tensor");
    TORCH_CHECK(pos.scalar_type() == torch::kFloat32, "pos must have dtype float32");
    TORCH_CHECK(lattice.scalar_type() == torch::kFloat32, "lattice must have dtype float32");
    TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");
    TORCH_CHECK(lattice.is_contiguous(), "lattice must be contiguous");
    TORCH_CHECK(pos.device() == lattice.device(), "pos and lattice must share a CUDA device");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos must have shape (N, 3)");
    TORCH_CHECK(lattice.dim() == 1 && lattice.size(0) == 3, "lattice must have shape (3,)");

    const int64_t N = pos.size(0);
    const auto index_options = torch::dtype(torch::kInt64).device(pos.device());
    if (N == 0)
    {
        return torch::empty({2, 0}, index_options);
    }

    const float r2 = static_cast<float>(r * r);
    const int blocks = static_cast<int>((N + kThreads - 1) / kThreads);
    const size_t shared_bytes = kThreads * sizeof(float3);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto pos_ptr = reinterpret_cast<const float3 *>(pos.data_ptr<float>());
    const auto box_ptr = lattice.data_ptr<float>();

    auto counts = torch::empty({N}, index_options);
    count_edges_kernel<<<blocks, kThreads, shared_bytes, stream.stream()>>>(
        pos_ptr, N, box_ptr, r2, loop, counts.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto full_cum = torch::cumsum(counts, 0);
    auto full_starts = torch::cat({torch::zeros({1}, index_options), full_cum});

    // With no cap, counts already provide the exact output layout. Write the
    // final edge index directly instead of materialising distances, columns and
    // a per-edge selection mask.
    if (max_num_neighbors <= 0)
    {
        const int64_t total_full = full_cum[N - 1].item<int64_t>();
        auto edge_index = torch::empty({2, total_full}, index_options);
        if (total_full > 0)
        {
            collect_edges_kernel<false><<<blocks, kThreads, shared_bytes, stream.stream()>>>(
                pos_ptr,
                N,
                box_ptr,
                r2,
                loop,
                full_starts.data_ptr<int64_t>(),
                nullptr,
                edge_index[0].data_ptr<int64_t>(),
                edge_index[1].data_ptr<int64_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return edge_index;
    }

    auto capped_counts = torch::clamp_max(counts, max_num_neighbors);
    auto capped_cum = torch::cumsum(capped_counts, 0);
    auto capped_starts = torch::cat(
        {torch::zeros({1}, index_options), capped_cum.narrow(0, 0, N - 1)});
    // Both output sizes are needed on the host. Transfer them together so the
    // capped path has a single host/device synchronization point.
    auto totals_cpu = torch::stack({full_cum[N - 1], capped_cum[N - 1]}).cpu();
    const int64_t total_full = totals_cpu[0].item<int64_t>();
    const int64_t total_out = totals_cpu[1].item<int64_t>();
    auto edge_index = torch::empty({2, total_out}, index_options);

    if (total_full == 0)
    {
        return edge_index;
    }

    auto scratch_dist = torch::empty({total_full}, pos.options());
    auto scratch_col = torch::empty({total_full}, index_options);
    auto chosen = torch::empty(
        {total_full}, torch::dtype(torch::kUInt8).device(pos.device()));
    collect_edges_kernel<true><<<blocks, kThreads, shared_bytes, stream.stream()>>>(
        pos_ptr,
        N,
        box_ptr,
        r2,
        loop,
        full_starts.data_ptr<int64_t>(),
        scratch_dist.data_ptr<float>(),
        nullptr,
        scratch_col.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    select_k_edges_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
        N,
        max_num_neighbors,
        full_starts.data_ptr<int64_t>(),
        capped_starts.data_ptr<int64_t>(),
        scratch_dist.data_ptr<float>(),
        scratch_col.data_ptr<int64_t>(),
        chosen.data_ptr<unsigned char>(),
        edge_index[0].data_ptr<int64_t>(),
        edge_index[1].data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return edge_index;
}
