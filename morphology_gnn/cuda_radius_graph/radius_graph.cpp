#include <torch/extension.h>

torch::Tensor radius_graph_pbc_cuda(torch::Tensor pos, double r, torch::Tensor lattice, bool loop, int64_t max_num_neighbors);

torch::Tensor radius_graph_pbc_cuda_wrapper(
    torch::Tensor pos,
    double r,
    torch::Tensor lattice,
    bool loop,
    int64_t max_num_neighbors
) {
    if (!pos.is_cuda()) {
        pos = pos.to(torch::kCUDA);
    }
    if (!lattice.is_cuda()) {
        lattice = lattice.to(torch::kCUDA);
    }

    pos = pos.contiguous();
    lattice = lattice.contiguous();

    auto edge_index = radius_graph_pbc_cuda(pos, r, lattice, loop, max_num_neighbors);
    return edge_index.cpu();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "radius_graph_pbc_cuda",
        &radius_graph_pbc_cuda_wrapper,
        "Build a periodic radius graph using CUDA"
    );
}
