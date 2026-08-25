#include <torch/extension.h>

torch::Tensor radius_graph_pbc_cuda(torch::Tensor pos, double r, torch::Tensor lattice, bool loop, int64_t max_num_neighbors);

torch::Tensor radius_graph_pbc_cuda_wrapper(
    torch::Tensor pos,
    double r,
    torch::Tensor lattice,
    bool loop,
    int64_t max_num_neighbors
) {
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3, "pos must have shape (N, 3)");
    TORCH_CHECK(
        lattice.dim() == 1 && lattice.size(0) == 3,
        "lattice must have shape (3,)");

    if (!pos.is_cuda()) {
        pos = pos.to(torch::kCUDA);
    }
    if (!lattice.is_cuda() || lattice.device() != pos.device()) {
        lattice = lattice.to(pos.device());
    }

    // The kernels operate on packed float3 positions. The Python entry point
    // already normalises to float32, but keeping this conversion here makes the
    // bound extension safe and avoids a second copy when inputs are ready.
    if (pos.scalar_type() != torch::kFloat32) {
        pos = pos.to(torch::kFloat32);
    }
    if (lattice.scalar_type() != torch::kFloat32) {
        lattice = lattice.to(torch::kFloat32);
    }
    if (!pos.is_contiguous()) {
        pos = pos.contiguous();
    }
    if (!lattice.is_contiguous()) {
        lattice = lattice.contiguous();
    }

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
