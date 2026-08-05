#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

static inline __device__ float minimum_image(float dx, float box)
{
    return dx - roundf(dx / box) * box;
}

__global__ void count_edges_kernel(
    const float *__restrict__ pos,
    int64_t N,
    const float *__restrict__ box,
    float r2,
    bool loop,
    int64_t *__restrict__ counts)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    const float xi0 = pos[3 * i + 0];
    const float xi1 = pos[3 * i + 1];
    const float xi2 = pos[3 * i + 2];
    int64_t count = 0;

    for (int64_t j = 0; j < N; ++j)
    {
        if (i == j && !loop)
        {
            continue;
        }

        const float xj0 = pos[3 * j + 0];
        const float xj1 = pos[3 * j + 1];
        const float xj2 = pos[3 * j + 2];

        const float dx = minimum_image(xi0 - xj0, box[0]);
        const float dy = minimum_image(xi1 - xj1, box[1]);
        const float dz = minimum_image(xi2 - xj2, box[2]);
        const float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= r2)
        {
            count += 1;
        }
    }

    counts[i] = count;
}

// Writes every valid (distance, neighbor) pair for each node into a flat
// scratch buffer laid out according to the (full) prefix-sum `starts`.
__global__ void collect_edges_kernel(
    const float *__restrict__ pos,
    int64_t N,
    const float *__restrict__ box,
    float r2,
    bool loop,
    const int64_t *__restrict__ starts,
    float *__restrict__ out_dist,
    int64_t *__restrict__ out_col)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    const float xi0 = pos[3 * i + 0];
    const float xi1 = pos[3 * i + 1];
    const float xi2 = pos[3 * i + 2];
    int64_t write_index = starts[i];

    for (int64_t j = 0; j < N; ++j)
    {
        if (i == j && !loop)
        {
            continue;
        }

        const float xj0 = pos[3 * j + 0];
        const float xj1 = pos[3 * j + 1];
        const float xj2 = pos[3 * j + 2];

        const float dx = minimum_image(xi0 - xj0, box[0]);
        const float dy = minimum_image(xi1 - xj1, box[1]);
        const float dz = minimum_image(xi2 - xj2, box[2]);
        const float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= r2)
        {
            out_dist[write_index] = dist2;
            out_col[write_index] = j;
            write_index += 1;
        }
    }
}

// Selects the closest max_k neighbors (by distance) for each node from the
// collected scratch buffer and writes the final edge_index. If max_k <= 0,
// every collected neighbor is kept. Ties are broken by scan order.
__global__ void select_k_edges_kernel(
    int64_t N,
    int64_t max_k,
    const int64_t *__restrict__ full_starts,   // size N + 1
    const int64_t *__restrict__ capped_starts, // size N
    const float *__restrict__ in_dist,
    const int64_t *__restrict__ in_col,
    unsigned char *__restrict__ chosen, // size total_full
    int64_t *__restrict__ out_rows,
    int64_t *__restrict__ out_cols)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
    {
        return;
    }

    const int64_t begin = full_starts[i];
    const int64_t cnt = full_starts[i + 1] - begin;
    const int64_t k = (max_k > 0 && max_k < cnt) ? max_k : cnt;

    for (int64_t m = 0; m < cnt; ++m)
    {
        chosen[begin + m] = 0;
    }

    int64_t out_pos = capped_starts[i];
    for (int64_t s = 0; s < k; ++s)
    {
        float best = INFINITY;
        int64_t best_m = -1;
        for (int64_t m = 0; m < cnt; ++m)
        {
            if (!chosen[begin + m] && in_dist[begin + m] < best)
            {
                best = in_dist[begin + m];
                best_m = m;
            }
        }
        // best_m is always valid because cnt >= k >= s + 1.
        chosen[begin + best_m] = 1;
        out_rows[out_pos + s] = i;
        out_cols[out_pos + s] = in_col[begin + best_m];
    }
}

torch::Tensor radius_graph_pbc_cuda(
    torch::Tensor pos,
    double r,
    torch::Tensor lattice,
    bool loop,
    int64_t max_num_neighbors)
{
    TORCH_CHECK(pos.is_cuda(), "pos must be a CUDA tensor");
    TORCH_CHECK(lattice.is_cuda(), "lattice must be a CUDA tensor");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos must have shape (N, 3)");
    TORCH_CHECK(lattice.dim() == 1 && lattice.size(0) == 3, "lattice must have shape (3,)");

    int64_t N = pos.size(0);
    float r2 = static_cast<float>(r * r);

    auto counts = torch::empty({N}, torch::dtype(torch::kInt64).device(pos.device()));
    const int threads = 128;
    const int blocks = static_cast<int>((N + threads - 1) / threads);

    count_edges_kernel<<<blocks, threads>>>(
        pos.data_ptr<float>(),
        N,
        lattice.data_ptr<float>(),
        r2,
        loop,
        counts.data_ptr<int64_t>());
    cudaDeviceSynchronize();

    auto counts_cpu = counts.to(torch::kCPU);
    auto counts_data = counts_cpu.data_ptr<int64_t>();

    // Full prefix sums (all collected neighbors) and capped prefix sums
    // (the closest max_num_neighbors, or all of them when max_num_neighbors <= 0).
    auto full_starts_cpu = torch::empty({N + 1}, torch::dtype(torch::kInt64));
    auto full_starts_data = full_starts_cpu.data_ptr<int64_t>();
    auto capped_starts_cpu = torch::empty({N}, torch::dtype(torch::kInt64));
    auto capped_starts_data = capped_starts_cpu.data_ptr<int64_t>();

    int64_t total_full = 0;
    int64_t total_out = 0;
    full_starts_data[0] = 0;
    for (int64_t i = 0; i < N; ++i)
    {
        total_full += counts_data[i];
        full_starts_data[i + 1] = total_full;

        int64_t k = (max_num_neighbors > 0 && max_num_neighbors < counts_data[i])
                        ? max_num_neighbors
                        : counts_data[i];
        capped_starts_data[i] = total_out;
        total_out += k;
    }

    auto full_starts = full_starts_cpu.to(pos.device());
    auto capped_starts = capped_starts_cpu.to(pos.device());
    auto edge_index = torch::empty({2, total_out}, torch::dtype(torch::kInt64).device(pos.device()));

    if (total_full > 0)
    {
        auto scratch_dist = torch::empty({total_full}, torch::dtype(torch::kFloat32).device(pos.device()));
        auto scratch_col = torch::empty({total_full}, torch::dtype(torch::kInt64).device(pos.device()));
        auto chosen = torch::empty({total_full}, torch::dtype(torch::kUInt8).device(pos.device()));

        collect_edges_kernel<<<blocks, threads>>>(
            pos.data_ptr<float>(),
            N,
            lattice.data_ptr<float>(),
            r2,
            loop,
            full_starts.data_ptr<int64_t>(),
            scratch_dist.data_ptr<float>(),
            scratch_col.data_ptr<int64_t>());
        cudaDeviceSynchronize();

        select_k_edges_kernel<<<blocks, threads>>>(
            N,
            max_num_neighbors,
            full_starts.data_ptr<int64_t>(),
            capped_starts.data_ptr<int64_t>(),
            scratch_dist.data_ptr<float>(),
            scratch_col.data_ptr<int64_t>(),
            chosen.data_ptr<unsigned char>(),
            edge_index[0].data_ptr<int64_t>(),
            edge_index[1].data_ptr<int64_t>());
        cudaDeviceSynchronize();
    }

    return edge_index;
}
