"""Plot a generated cubic-lattice diagnostic against an ideal 10³ reference.

Example::

    python examples/plot_cubic_lattice_diagnostic.py \
        /tmp/simple-cubic-ve-final/simple_cubic_lattice.pt \
        --output examples/artifacts/cubic_lattice_diagnostic.png

The generated panel always shows the exact positions saved by the diffusion
example.  The 10 x 10 x 10 panel is an *ideal reference*, not an extrapolated
or tiled model sample; keeping that distinction explicit prevents a small,
failed sample from being mistaken for a large generated lattice.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from morphology_gnn.model.diffusion_trainer import pair_correlation


def infer_spacing(truth: torch.Tensor) -> float:
    """Infer the nearest-neighbour spacing from an ordered cubic reference."""
    values = torch.unique(truth.flatten()).sort().values
    steps = values[1:] - values[:-1]
    steps = steps[steps > 1e-6]
    if not len(steps):
        raise ValueError("could not infer a lattice spacing from the reference")
    return float(steps.min())


def simple_cubic(side: int, spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a periodic simple cubic lattice and its matching cell."""
    points = torch.tensor(
        [
            [spacing * x, spacing * y, spacing * z]
            for z in range(side)
            for y in range(side)
            for x in range(side)
        ],
        dtype=torch.float32,
    )
    return points, torch.full((3,), side * spacing)


def draw_points(ax, points: torch.Tensor, title: str, size: float) -> None:
    """Draw points colored by z on an equal-scale 3-D axis."""
    xyz = points.cpu().numpy()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], s=size, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    centre = 0.5 * (lo + hi)
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(value - span / 2, value + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help=".pt artifact from the lattice example")
    parser.add_argument("--reference-side", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/artifacts/cubic_lattice_diagnostic.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_side < 2:
        raise ValueError("--reference-side must be at least 2")
    result = torch.load(args.sample, map_location="cpu", weights_only=True)
    generated = result["generated"].float()
    truth = result["truth"].float()
    cell = result["cell"].float()
    rms = float(result["rms"])
    spacing = infer_spacing(truth)
    reference, reference_cell = simple_cubic(args.reference_side, spacing)
    rmax = min(float(cell.min()) / 2, 3.0 * spacing)
    g_generated, edges = pair_correlation(generated, cell, dr=0.05, rmax=rmax)
    g_reference, _ = pair_correlation(reference, reference_cell, dr=0.05, rmax=rmax)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    generated_ax = fig.add_subplot(1, 3, 1, projection="3d")
    reference_ax = fig.add_subplot(1, 3, 2, projection="3d")
    rdf_ax = fig.add_subplot(1, 3, 3)
    draw_points(
        generated_ax,
        generated,
        f"Actual diffusion output ({len(generated)} points)\nRMS = {rms:.2f} Å",
        size=48,
    )
    draw_points(
        reference_ax,
        reference,
        f"Ideal simple cubic reference ({args.reference_side}³ = {len(reference)} points)",
        size=8,
    )
    rdf_ax.plot(centres, g_generated, color="tab:orange", lw=1.8, label="diffusion output")
    rdf_ax.plot(centres, g_reference, color="black", lw=1.2, label="ideal 10³ cubic")
    for n in range(1, 10):
        distance = spacing * math.sqrt(n)
        if distance <= rmax:
            rdf_ax.axvline(distance, color="black", alpha=0.12, lw=0.8)
    rdf_ax.set_title("Periodic radial distribution function")
    rdf_ax.set_xlabel("distance (Å)")
    rdf_ax.set_ylabel("g(r)")
    rdf_ax.set_ylim(bottom=0)
    rdf_ax.legend(frameon=False)
    rdf_ax.text(
        0.02,
        0.98,
        "Vertical lines: ideal cubic shell distances",
        transform=rdf_ax.transAxes,
        va="top",
        fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
