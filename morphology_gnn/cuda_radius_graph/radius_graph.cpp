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
    // Keep the edge_index on the GPU (no device->host copy): the hot diffusion
    // path rebuilds this graph every training step, so a D2H+H2D round-trip per
    // call is pure overhead. Callers that need a CPU tensor move it themselves
    // (e.g. morphology_gnn.radius_graph.try_cuda_radius_graph_pbc -> .to(pos.device)).
    return edge_index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "radius_graph_pbc_cuda",
        &radius_graph_pbc_cuda_wrapper,
        "Build a periodic radius graph using CUDA"
    );
}
