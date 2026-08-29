"""Train and check sequential diffusion generation on a simple cubic lattice.

This is a small, synthetic smoke test for the sequential diffusion runner.  A
3 x 3 x 3 simple-cubic lattice is converted into one training graph per
*next-point* prediction: all earlier lattice sites are clean context and the
last site is the only noisy target.  At sampling time a complete lower layer
and the first site of the next layer fix the lattice orientation, then
``sample_sequential_z`` generates each remaining site while keeping every site
already generated fixed as context.

The test deliberately orders sites by ``z, y, x``.  This agrees with the
sampler's +z frontier: points in the same layer may share z, while later layers
cannot be generated below earlier ones.

Run from the repository root::

    python examples/diffusion_simple_cubic_lattice.py

The command writes ``simple_cubic_lattice.pt`` and ``simple_cubic_lattice.xyz``
to ``--outdir``.  It exits non-zero when the generated lattice's optimal
site-matching RMS error exceeds ``--max-rms``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel
from morphology_gnn.model.diffusion_trainer import DiffusionMoleculeModule


def simple_cubic_lattice(side: int, spacing: float, margin: float) -> torch.Tensor:
    """Return sites in the sequential sampler's ``z, y, x`` generation order."""
    return torch.tensor(
        [
            [margin + spacing * x, margin + spacing * y, margin + spacing * z]
            for z in range(side)
            for y in range(side)
            for x in range(side)
        ],
        dtype=torch.float32,
    )


def next_point_dataset(
    lattice: torch.Tensor,
    cell: torch.Tensor,
    repeats: int,
    seed_points: int,
) -> list[Data]:
    """Make z-ordered graphs with a clean prefix and one noisy next-point target."""
    atom_type = torch.tensor([[6]], dtype=torch.long)  # one pseudo-species (carbon)
    examples: list[Data] = []
    # A complete lower layer and one point above it establish all three cubic
    # axes.  An SE(3)-equivariant directional noise head cannot infer a normal
    # direction from a coplanar context, so asking it to begin a new layer
    # without that anchor is ill-posed. Repetition provides stochastic draws.
    for _ in range(repeats):
        for target_index in range(seed_points, len(lattice)):
            count = target_index + 1
            examples.append(
                Data(
                    x=atom_type.expand(count, -1).clone(),
                    pos=lattice[:count].clone(),
                    box=cell.reshape(1, 3).clone(),
                    target_mask=torch.tensor(
                        [False] * target_index + [True], dtype=torch.bool
                    ),
                )
            )
    return examples


def optimal_site_rms(generated: torch.Tensor, truth: torch.Tensor) -> float:
    """RMS site error after one-to-one nearest-site assignment."""
    # scipy is already a project dependency and avoids treating an arbitrary
    # permutation of otherwise correct lattice points as a failure.
    from scipy.optimize import linear_sum_assignment

    distances = torch.cdist(generated.cpu(), truth.cpu()).numpy()
    rows, cols = linear_sum_assignment(distances)
    return float((distances[rows, cols] ** 2).mean() ** 0.5)


def write_xyz(path: Path, points: torch.Tensor) -> None:
    """Write generated pseudo-atoms for a quick external visualization."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(points)}\nSimple cubic diffusion sample\n")
        for x, y, z in points.cpu().tolist():
            handle.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=3, help="sites per lattice axis")
    parser.add_argument("--spacing", type=float, default=2.0, help="lattice spacing")
    parser.add_argument("--margin", type=float, default=2.0, help="cell-edge margin")
    parser.add_argument(
        "--seed-points",
        type=int,
        default=None,
        help="fixed context sites (default: first layer plus one site above it)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="forward noise amplitude (Angstrom) at sigma=1; default: 1.5 * spacing",
    )
    parser.add_argument("--steps", type=int, default=64, help="reverse diffusion steps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rms", type=float, default=0.75)
    parser.add_argument("--outdir", type=Path, default=Path("examples/artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.side < 2:
        raise ValueError("--side must be at least 2 so there is a next point to generate")
    if min(args.spacing, args.margin, args.lr) <= 0:
        raise ValueError("--spacing, --margin, and --lr must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lattice = simple_cubic_lattice(args.side, args.spacing, args.margin)
    seed_points = args.seed_points if args.seed_points is not None else args.side**2 + 1
    if not 4 <= seed_points < len(lattice):
        raise ValueError("--seed-points must be at least 4 and fewer than all lattice sites")
    # Keep the lattice well inside an orthorhombic cell: this example evaluates
    # next-point conditioning, not periodic-image ambiguity at the boundary.
    cell = torch.full((3,), 2 * args.margin + args.spacing * args.side)
    dataset = next_point_dataset(
        lattice, cell, args.repeats, seed_points
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # The target is noised at a data-scale amplitude (--noise-scale, default
    # 1.5 * spacing), NOT the cell width: cell-scaled noise would make the
    # denoising target irreducibly ambiguous at every useful noise level.  An
    # all-to-all PBC radius keeps the next point connected to its clean prefix
    # even at the noisiest reverse step.
    radius = float(torch.linalg.vector_norm(cell))
    noise_scale = args.noise_scale if args.noise_scale is not None else 1.5 * args.spacing
    model = DiffusionMoleculeModel(
        hidden_dim=64,
        num_layers=3,
        num_rbf=24,
        cutoff_upper=radius,
        cell_embed_dim=16,
        dropout=0.0,
        noise_schedule="linear",
    )
    runner = DiffusionMoleculeModule(
        model=model,
        radius=radius,
        lr=args.lr,
        sample_steps=args.steps,
        sample_ddim=True,
        sample_eta=0.0,
        noise_scale=noise_scale,
        z_ordered=True,
    ).to(device)
    optimizer = torch.optim.Adam(runner.parameters(), lr=args.lr)

    runner.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = runner.training_step(batch, epoch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch + 1:4d}/{args.epochs}: loss={sum(losses) / len(losses):.5f}")

    # The fixed lower layer and first upper-layer site establish an oriented
    # cubic frame. The test generates the rest one at a time from this prefix.
    seed_pos = lattice[:seed_points].to(device)
    atom_types = torch.full(
        (len(lattice) - seed_points,), 6, dtype=torch.long, device=device
    )
    generated_tail = runner.sample_sequential_z(
        atom_types,
        cell.to(device),
        initial_pos=seed_pos,
        initial_types=torch.full((seed_points,), 6, dtype=torch.long, device=device),
        num_points=len(atom_types),
        steps=args.steps,
        ddim=True,
        eta=0.0,
        seed=args.seed + 1,
    )
    generated = torch.cat([seed_pos, generated_tail]).cpu()
    rms = optimal_site_rms(generated, lattice)
    z_is_ordered = bool(torch.all(generated[1:, 2] >= generated[:-1, 2] - 1e-5))

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"truth": lattice, "generated": generated, "cell": cell, "rms": rms},
        args.outdir / "simple_cubic_lattice.pt",
    )
    write_xyz(args.outdir / "simple_cubic_lattice.xyz", generated)
    print(f"simple-cubic next-point RMS: {rms:.4f} (threshold {args.max_rms:.4f})")
    print(f"generated in non-decreasing z order: {z_is_ordered}")
    print(f"artifacts: {args.outdir}")
    if not z_is_ordered or rms > args.max_rms:
        raise SystemExit("simple-cubic sequential generation check failed")


if __name__ == "__main__":
    main()
