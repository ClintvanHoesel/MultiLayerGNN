"""Post-training evaluation: sequential (+z) generation + pair-correlation check.

Loads the best trained ``DiffusionMoleculeModule`` checkpoint, generates a
z-ordered stack of ``--num-points`` molecules with the sequential sampler, and
compares its pair-correlation function ``g(r)`` to the ground-truth film
(both the full box and the same-number-of-molecules lowest-z slice), reporting
the RDF mean-absolute-difference and the first-shell peak position, plus an
overlay plot.

Usage:
    python runs/check_rdf.py --ckpt runs/artifacts_diffusion/<run>/checkpoints/<file>.ckpt \
        [--num-points 150 --steps 50 --bottom-z 5.0 --seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import run_diffusion  # noqa: E402
from morphology_gnn.data import ZOrderedBoxMolecularDataset  # noqa: E402
from morphology_gnn.model.diffusion_model import DiffusionMoleculeModel  # noqa: E402
from morphology_gnn.model.diffusion_trainer import (  # noqa: E402
    DiffusionMoleculeModule,
    pair_correlation,
    rdf_mad,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Best checkpoint .ckpt file")
    ap.add_argument("--num-points", type=int, default=150)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bottom-z", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="PNG output path (default: next to ckpt)")
    args = ap.parse_args()

    # Rebuild the dataset so we get the reference box (truth COMs + cell).
    run_argv = ["run_diffusion.py", "--dataset", "box_zordered"]
    rargs, dotted = run_diffusion.parse_cli(run_argv)
    config = run_diffusion.resolve_config(rargs, dotted)
    ds = run_diffusion.build_dataset(config)

    ref = ds.box_reference(0)          # per-molecule reference metadata
    box_ref_sample = ds.box_sample(0)  # full box sample (species, cell, truth pos)
    species = box_ref_sample.x.squeeze(-1)
    cell = box_ref_sample.box.squeeze(0)
    truth = box_ref_sample.pos          # (N, 3) truth COMs
    box = ref["box"].squeeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = run_diffusion.build_diffusion_model(
        config["model"], radius=config["radius"], noise_schedule=config["diffusion"]["schedule"]
    )
    module = DiffusionMoleculeModule.load_from_checkpoint(
        args.ckpt, model=model, map_location=device
    ).to(device)
    module.eval()
    print(f"Loaded checkpoint {os.path.basename(args.ckpt)} (z_ordered={module.z_ordered})")

    K = min(args.num_points, int(species.shape[0]))
    gen = module.sample_sequential_z(
        species[:K].to(device),
        cell.to(device),
        num_points=K,
        bottom_z_exclusion=args.bottom_z,
        z_step=0.0,
        steps=args.steps,
        seed=args.seed,
        device=device,
    ).cpu()
    print(f"Sequential generation: {gen.shape[0]} molecules, "
          f"z range {gen[:, 2].min():.2f}..{gen[:, 2].max():.2f}, "
          f"monotone non-decreasing={bool((gen[1:, 2] >= gen[:-1, 2] - 1e-6).all())}, "
          f"z >= bottom={bool((gen[:, 2] >= args.bottom_z - 1e-6).all())}")

    # Ground-truth references: full film and the same-count lowest-z slice.
    order = torch.argsort(truth[:, 2])
    truth_slice = truth[order[:K]]  # the bottom K molecules of the film

    dr = 0.3
    g_gen, e_gen = pair_correlation(gen, box, dr=dr)
    g_truth_full, _ = pair_correlation(truth, box, dr=dr)
    g_truth_slice, _ = pair_correlation(truth_slice, box, dr=dr)
    centers = 0.5 * (e_gen[:-1] + e_gen[1:])

    def first_peak(g, centers, lo=1.0, hi=12.0):
        m = (centers >= lo) & (centers <= hi)
        if not m.any():
            return float("nan")
        return float(centers[m][torch.argmax(g[m])])

    print("\nPair-correlation (g(r)) comparison:")
    print(f"  generated ({K})      : first-peak r = {first_peak(g_gen, centers):.2f} A")
    print(f"  truth lowest-{K}      : first-peak r = {first_peak(g_truth_slice, centers):.2f} A")
    print(f"  truth full ({truth.shape[0]}): first-peak r = {first_peak(g_truth_full, centers):.2f} A")
    print(f"  rdf_mad (gen vs truth slice) = {rdf_mad(g_gen, g_truth_slice).item():.4f}")
    print(f"  rdf_mad (gen vs truth full)  = {rdf_mad(g_gen, g_truth_full).item():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centers, g_gen, "C0-", lw=1.8, label=f"generated (sequential, {K})")
    ax.plot(centers, g_truth_slice, "C1--", lw=1.5, label=f"truth lowest-{K}")
    ax.plot(centers, g_truth_full, "k-", lw=1.2, alpha=0.6, label=f"truth full ({truth.shape[0]})")
    ax.set_xlabel("r ($\\AA$)"); ax.set_ylabel(r"$g(r)$")
    ax.set_title("Pair correlation: generated (sequential +z) vs truth")
    ax.legend(); ax.grid(alpha=0.3)
    out = args.out or (os.path.splitext(args.ckpt)[0] + "_rdf.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved RDF plot to {out}")


if __name__ == "__main__":
    main()
