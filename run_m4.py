"""
M4: we want to see if the distance to outcome a linear probe reads off the hidden
state (M3) generalize outside the DTZ range it was trained on, or if it's
memorizing the specific values seen during probe training.

Trains a DistanceProbe only on positions with
|dtz| <= --train_max_dtz ("near" the outcome).
Test it on positions with
|dtz| >= --test_min_dtz ("far" from the outcome, magnitude range never
seen in training). A gap is left between the two ranges on purpose (default
train<=5, test>=10) so there's no near-boundary overlap in the test.

Reports per layer: test_mae (the number that matters), baseline_mae (the
floor a probe with no generalizable notion of distance can't beat), and indist_mae (a
held-out slice of the near range, for context on interpolation vs
extrapolation difficulty).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, random_split

from dataset import KPKDataset, collate_fn
from model import MoveGPT
from probe import (
    DistanceProbe, DistanceProbeMLP, ProbeTrainConfig,
    train_distance_probe_ood, collect_all_layers,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--out", type=str, default="./m4_results.json")
    ap.add_argument("--probe_data_frac", type=float, default=0.05,
                     help="Fraction of the dataset to extract for M4. Higher than "
                          "run_probes.py's default since the far range (|dtz|>=test_min_dtz) "
                          "is a minority of positions and needs enough raw examples on its "
                          "own, not just enough total examples.")
    ap.add_argument("--train_max_dtz", type=int, default=5,
                     help="'Near' range for probe training: |dtz| <= this value.")
    ap.add_argument("--test_min_dtz", type=int, default=10,
                     help="'Far' range for probe testing: |dtz| >= this value. Left with a "
                          "gap above --train_max_dtz on purpose (no overlap).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=8)
    args = ap.parse_args()

    if args.test_min_dtz <= args.train_max_dtz:
        raise ValueError(
            f"--test_min_dtz ({args.test_min_dtz}) must be > --train_max_dtz "
            f"({args.train_max_dtz}) -- M4 tests generalization to a DTZ range strictly "
            f"beyond what the probe trained on."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = MoveGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1}, "
          f"best_val_loss={ckpt['best_val_loss']:.4f}")

    print("Loading dataset...")
    t0 = time.time()
    full_ds = KPKDataset(args.data_dir, domain=args.domain, max_len=cfg.max_len)
    print(f"  {len(full_ds)} games loaded in {time.time() - t0:.1f}s")

    n_probe = max(1, int(len(full_ds) * args.probe_data_frac))
    probe_ds, _rest = random_split(
        full_ds, [n_probe, len(full_ds) - n_probe],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  using {n_probe} games for M4 ({args.probe_data_frac:.0%} of dataset)")

    loader = DataLoader(
        probe_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
    )

    print("Extracting all-layer hidden states in a single pass...")
    t0 = time.time()
    layer_features, labels = collect_all_layers(model, loader, device=device)
    n_tokens = layer_features[0].shape[0]
    print(f"  {n_tokens} token positions x {cfg.n_layer} layers extracted in {time.time() - t0:.1f}s")

    dtz_raw = labels["dtz_raw"]
    valid = ~torch.isnan(dtz_raw)
    abs_dtz = dtz_raw.abs()
    train_mask = (abs_dtz <= args.train_max_dtz) & valid
    test_mask = (abs_dtz >= args.test_min_dtz) & valid
    print(f"  near range (|dtz|<={args.train_max_dtz}): {train_mask.sum().item()} positions")
    print(f"  far range  (|dtz|>={args.test_min_dtz}): {test_mask.sum().item()} positions")
    if test_mask.sum().item() < 200:
        print("  WARNING: very few far-range positions -- test_mae below will be noisy. "
              "Consider raising --probe_data_frac.")

    results = {}
    probe_cfg = ProbeTrainConfig(epochs=args.probe_epochs)

    for layer_idx in range(cfg.n_layer):
        feats = layer_features[layer_idx].to(device)
        dtz_dev = dtz_raw.to(device)
        train_mask_dev = train_mask.to(device)
        test_mask_dev = test_mask.to(device)

        results[layer_idx] = {}
        for probe_name, probe_cls in (("linear", DistanceProbe), ("mlp", DistanceProbeMLP)):
            print(f"  layer {layer_idx}  {probe_name:6s}  training...", end="", flush=True)
            t_probe = time.time()

            # same seed before building+training each probe: identical
            # weight init distribution and identical minibatch order as
            # the other probe would see at this layer.
            torch.manual_seed(args.seed + layer_idx)
            probe = probe_cls(cfg.d_model).to(device)
            _, metrics = train_distance_probe_ood(
                probe, feats, dtz_dev, train_mask_dev, test_mask_dev, cfg=probe_cfg
            )
            results[layer_idx][probe_name] = metrics
            gap = metrics["baseline_mae"] - metrics["test_mae"]
            print(f"\r  layer {layer_idx}  {probe_name:6s}  test_mae={metrics['test_mae']:.2f}  "
                  f"baseline_mae={metrics['baseline_mae']:.2f}  "
                  f"indist_mae={metrics['indist_mae']:.2f}  "
                  f"(beats baseline by {gap:+.2f})  ({time.time() - t_probe:.1f}s)")

    print("\n=== Summary: M4 generalization, linear vs MLP control "
          "(near |dtz|<=%d -> far |dtz|>=%d) ===" % (args.train_max_dtz, args.test_min_dtz))
    header = (f"{'layer':>5} | {'lin test_mae':>12} | {'lin beats_bl':>12} | "
              f"{'mlp test_mae':>12} | {'mlp beats_bl':>12}")
    print(header)
    print("-" * len(header))
    for layer_idx in range(cfg.n_layer):
        lin, mlp = results[layer_idx]["linear"], results[layer_idx]["mlp"]
        lin_gap = lin["baseline_mae"] - lin["test_mae"]
        mlp_gap = mlp["baseline_mae"] - mlp["test_mae"]
        print(f"{layer_idx:>5} | {lin['test_mae']:>12.2f} | {lin_gap:>+12.2f} | "
              f"{mlp['test_mae']:>12.2f} | {mlp_gap:>+12.2f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
