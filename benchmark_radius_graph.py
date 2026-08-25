"""Verify CUDA == Python radius-graph equivalence and benchmark both paths.

The CUDA extension (morphology_gnn.cuda_radius_graph.radius_graph_pbc) is
compared against the pure-PyTorch reference (morphology_gnn.radius_graph.
radius_graph_pbc). Edge column order is not a contract, so graphs are compared
as unordered edge sets.
"""

from __future__ import annotations

import statistics
import time

import torch

from morphology_gnn.cuda_radius_graph import (
    _cuda_available,
    radius_graph_pbc as cuda_radius_graph_pbc,
)
from morphology_gnn.radius_graph import radius_graph_pbc

torch.manual_seed(0)


def undirected_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Represent an edge list as a set of sorted (src, dst) tuples."""
    edges = edge_index.cpu().t().tolist()
    return {tuple(sorted((a, b))) for a, b in edges}


def gen_case(n: int, box: float, r: float) -> torch.Tensor:
    return (torch.rand(n, 3) * box).to(torch.float32)


def verify() -> bool:
    """Return True if all CUDA/Python results agree."""
    print("=" * 70)
    print("EQUIVALENCE CHECK (CUDA vs Python, unordered edge sets)")
    print("=" * 70)
    ok = True

    cases = [
        # (N, box, r, loop, max_num_neighbors)
        (50, 12.0, 3.0, False, None),
        (50, 12.0, 3.0, True, None),
        (137, 18.0, 4.0, False, 8),
        (137, 18.0, 4.0, True, 8),
        (500, 25.0, 5.0, False, None),
        (500, 25.0, 5.0, True, 12),
        (2000, 40.0, 6.0, False, None),
        (2000, 40.0, 6.0, True, 16),
    ]
    for n, box, r, loop, mnn in cases:
        pos = gen_case(n, box, r).cuda()
        box_t = torch.tensor([box] * 3, dtype=torch.float32).cuda()
        py = radius_graph_pbc(pos, r=r, lattice=box_t, loop=loop, max_num_neighbors=mnn)
        cu = cuda_radius_graph_pbc(
            pos, r=r, lattice=box_t, loop=loop, max_num_neighbors=mnn
        ).cpu()
        match = undirected_set(py) == undirected_set(cu)
        ok &= match
        print(
            f"  N={n:5d} box={box:4.1f} r={r:.1f} loop={int(loop)} "
            f"mnn={str(mnn):>4}  edges(py)={py.shape[1]:6d} "
            f"edges(cu)={cu.shape[1]:6d}  {'OK' if match else 'MISMATCH'}"
        )
        if not match:
            only_py = undirected_set(py) - undirected_set(cu)
            only_cu = undirected_set(cu) - undirected_set(py)
            print(f"    only in python: {list(only_py)[:5]}")
            print(f"    only in cuda:   {list(only_cu)[:5]}")

    # Edge-case: empty graph
    pos = torch.empty((0, 3), dtype=torch.float32).cuda()
    box_t = torch.tensor([10.0, 10.0, 10.0]).cuda()
    py = radius_graph_pbc(pos, r=3.0, lattice=box_t, loop=False)
    cu = cuda_radius_graph_pbc(pos, r=3.0, lattice=box_t, loop=False).cpu()
    match = undirected_set(py) == undirected_set(cu)
    ok &= match
    print(
        f"  N=0 (empty)                                                      "
        f"{'OK' if match else 'MISMATCH'}"
    )
    return ok


def timed(fn, *args, cuda: bool = False, repeats: int = 30, **kwargs) -> float:
    """Median wall time in microseconds."""
    # warmup
    for _ in range(5):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


def benchmark() -> None:
    print()
    print("=" * 70)
    print("BENCHMARK (median wall time over 30 runs, warmup 5)")
    print("  python: pure-PyTorch on GPU | cuda: CUDA kernel on GPU")
    print("=" * 70)
    print(
        f"  {'N':>6} {'r':>4} {'mnn':>4} | {'python (us)':>12} "
        f"{'cuda (us)':>12} {'speedup':>8}"
    )

    for n, r, mnn in [
        (50, 3.0, None),
        (137, 4.0, 8),
        (500, 5.0, None),
        (1000, 5.0, None),
        (2000, 6.0, None),
        (4000, 6.0, None),
        (8000, 6.0, None),
    ]:
        box = max(r * 2.5, 10.0)
        pos_c = gen_case(n, box, r).cuda()
        box_c = torch.tensor([box] * 3, dtype=torch.float32).cuda()

        t_py = timed(
            radius_graph_pbc,
            pos_c,
            r=r,
            lattice=box_c,
            loop=False,
            max_num_neighbors=mnn,
            cuda=True,
        )
        t_cu = timed(
            cuda_radius_graph_pbc,
            pos_c,
            r=r,
            lattice=box_c,
            loop=False,
            max_num_neighbors=mnn,
            cuda=True,
        )
        speedup = t_py / t_cu if t_cu > 0 else float("inf")
        print(
            f"  {n:>6} {r:>4.1f} {str(mnn):>4} | {t_py:12.1f} "
            f"{t_cu:12.1f} {speedup:7.2f}x"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is NOT available; nothing to verify/benchmark.")
        return
    if not _cuda_available:
        print("CUDA extension is NOT loaded; cannot compare.")
        return
    ok = verify()
    benchmark()
    print()
    print("RESULT:", "ALL EQUIVALENT ✅" if ok else "MISMATCHES FOUND ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
