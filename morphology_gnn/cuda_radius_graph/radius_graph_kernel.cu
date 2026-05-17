#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

static inline __device__ float minimum_image(float dx, float box) {
    return dx - roundf(dx / box) * box;
}

__global__ void count_edges_kernel(
    const float* __restrict__ pos,
    int64_t N,
    const float* __restrict__ box,
    float r2,
    int64_t* __restrict__ counts
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const float xi0 = pos[3 * i + 0];
    const float xi1 = pos[3 * i + 1];
    const float xi2 = pos[3 * i + 2];
    int64_t count = 0;

    for (int64_t j = 0; j < N; ++j) {
        if (i == j) {
            continue;
        }

        const float xj0 = pos[3 * j + 0];
        const float xj1 = pos[3 * j + 1];
        const float xj2 = pos[3 * j + 2];

        const float dx = minimum_image(xi0 - xj0, box[0]);
        const float dy = minimum_image(xi1 - xj1, box[1]);
        const float dz = minimum_image(xi2 - xj2, box[2]);
        const float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= r2) {
            count += 1;
        }
    }

    counts[i] = count;
}

__global__ void fill_edges_kernel(
    const float* __restrict__ pos,
    int64_t N,
    const float* __restrict__ box,
    float r2,
    const int64_t* __restrict__ starts,
    int64_t* __restrict__ out_rows,
    int64_t* __restrict__ out_cols
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const float xi0 = pos[3 * i + 0];
    const float xi1 = pos[3 * i + 1];
    const float xi2 = pos[3 * i + 2];
    int64_t write_index = starts[i];

    for (int64_t j = 0; j < N; ++j) {
        if (i == j) {
            continue;
        }

        const float xj0 = pos[3 * j + 0];
        const float xj1 = pos[3 * j + 1];
        const float xj2 = pos[3 * j + 2];

        const float dx = minimum_image(xi0 - xj0, box[0]);
        const float dy = minimum_image(xi1 - xj1, box[1]);
        const float dz = minimum_image(xi2 - xj2, box[2]);
        const float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= r2) {
            out_rows[write_index] = i;
            out_cols[write_index] = j;
            write_index += 1;
        }
    }
}

torch::Tensor radius_graph_pbc_cuda(
    torch::Tensor pos,
    double r,
    torch::Tensor lattice,
    bool loop,
    int64_t max_num_neighbors
) {
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
        counts.data_ptr<int64_t>()
    );
    cudaDeviceSynchronize();

    auto counts_cpu = counts.to(torch::kCPU);
    auto counts_data = counts_cpu.data_ptr<int64_t>();

    int64_t total_edges = 0;
    for (int64_t i = 0; i < N; ++i) {
        total_edges += counts_data[i];
    }

    auto starts_cpu = torch::empty({N}, torch::dtype(torch::kInt64));
    auto starts_data = starts_cpu.data_ptr<int64_t>();
    int64_t current = 0;
    for (int64_t i = 0; i < N; ++i) {
        starts_data[i] = current;
        current += counts_data[i];
    }

    auto starts = starts_cpu.to(pos.device());
    auto edge_index = torch::empty({2, total_edges}, torch::dtype(torch::kInt64).device(pos.device()));

    if (total_edges > 0) {
        fill_edges_kernel<<<blocks, threads>>>(
            pos.data_ptr<float>(),
            N,
            lattice.data_ptr<float>(),
            r2,
            starts.data_ptr<int64_t>(),
            edge_index[0].data_ptr<int64_t>(),
            edge_index[1].data_ptr<int64_t>()
        );
        cudaDeviceSynchronize();
    }

    return edge_index;
}
