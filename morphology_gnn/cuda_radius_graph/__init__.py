from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.cpp_extension import load

try:
    from . import _cuda_radius_graph as _extension

    _cuda_available = True
except ImportError:
    _extension = None
    _cuda_available = False

_this_dir = Path(__file__).resolve().parent
_build_dir = _this_dir / "_build"
_build_error = None


def _load_extension() -> None:
    global _extension, _build_error, _cuda_available
    if _extension is not None or _build_error is not None:
        return

    try:
        _extension = load(
            name="morphology_gnn_cuda_radius_graph",
            sources=[
                str(_this_dir / "radius_graph.cpp"),
                str(_this_dir / "radius_graph_kernel.cu"),
            ],
            build_directory=str(_build_dir),
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
        _cuda_available = True
    except Exception as exc:
        _build_error = exc
        _cuda_available = False


_load_extension()


def radius_graph_pbc(
    pos: torch.Tensor,
    r: float,
    lattice: torch.Tensor,
    loop: bool = False,
    max_num_neighbors: int | None = None,
) -> torch.Tensor:
    if not _cuda_available:
        raise RuntimeError(
            "CUDA extension for morphology_gnn is not available. "
            f"Original error: {_build_error}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos, dtype=torch.float32)
    if not isinstance(lattice, torch.Tensor):
        lattice = torch.as_tensor(lattice, dtype=torch.float32)

    if lattice.ndim == 2:
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3,) or (3, 3)")
        lattice_diag = torch.diagonal(lattice)
        if not torch.allclose(lattice, torch.diag(lattice_diag)):
            raise ValueError("CUDA PBC extension only supports orthorhombic cells.")
        lattice = lattice_diag

    lattice = lattice.contiguous().to(torch.float32)
    pos = pos.contiguous().to(torch.float32)

    max_neighbors = -1 if max_num_neighbors is None else int(max_num_neighbors)
    return _extension.radius_graph_pbc_cuda(
        pos, float(r), lattice, bool(loop), max_neighbors
    )
