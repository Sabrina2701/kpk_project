"""
Runs both M2 (static probes: own_king, other_king, special_sq; 64-class
square classification tests whether the model reconstructs board state
from the move-history it's fed) and M3 (outcome probes: wdl, dtz_bucket: tests whether
 something requiring look-ahead is also linearly present) at
every layer, using a single forward pass over the probing data.

Uses the same held-out split M1 was validated on (same --val_frac/--seed
default as train_m1.py), so the probes are trained/evaluated on games M1
never trained on.
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
    SquareProbe, OutcomeProbe, ProbeTrainConfig,
    train_probe, collect_all_layers,
)

# task -> (label key in collect_all_layers's labels dict, n_classes, class_weighted)
TASKS = {
    "own_king": ("own_king", 64, False),
    "other_king": ("other_king", 64, False),
    "special_sq": ("special_sq", 64, False),
    "wdl": ("wdl", 5, True),          # skewed ~66/24/10 (draw/win/loss)
    "dtz_bucket": ("dtz_bucket", 6, True),  # edges=(0,2,5,10,20) -> 6 buckets
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--out", type=str, default="./probe_results.json")
    ap.add_argument("--probe_data_frac", type=float, default=0.02,
                     help="Fraction of the full dataset to run probing on. Probes are "
                          "simple linear layers: ~4000 games (2%% of 200k) is already "
                          "~200k-plus token positions, far more than a linear classifier "
                          "over a 256-dim input needs. Raise this only if a probe's "
                          "accuracy looks noisy/unstable across reruns.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2,
                     help="DataLoader worker processes. KPKDataset.__getitem__ replays each "
                          "game with python-chess (CPU-bound), so >0 workers overlaps that "
                          "with GPU compute instead of blocking on it. Colab free tier "
                          "typically has 2 CPU cores.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=10)
    args = ap.parse_args()

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
    print(f"  model: {cfg}")

    print("Loading dataset...")
    t0 = time.time()
    full_ds = KPKDataset(args.data_dir, domain=args.domain, max_len=cfg.max_len)
    print(f"  {len(full_ds)} games loaded in {time.time() - t0:.1f}s")

    n_probe = max(1, int(len(full_ds) * args.probe_data_frac))
    probe_ds, _rest = random_split(
        full_ds, [n_probe, len(full_ds) - n_probe],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  using {n_probe} games for probing ({args.probe_data_frac:.0%} of dataset)")

    loader = DataLoader(
        probe_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
    )

    print("Extracting all-layer hidden states in a single pass...")
    t0 = time.time()
    layer_features, labels = collect_all_layers(model, loader, device=device)
    n_tokens = layer_features[0].shape[0]
    print(f"  {n_tokens} token positions x {cfg.n_layer} layers extracted in {time.time() - t0:.1f}s")

    results = {}  # results[layer][task] = metrics dict
    probe_cfg = ProbeTrainConfig(epochs=args.probe_epochs)

    for layer_idx in range(cfg.n_layer):
        results[layer_idx] = {}
        feats = layer_features[layer_idx]
        for task_name, (label_key, n_classes, class_weighted) in TASKS.items():
            print(f"  layer {layer_idx}  {task_name:12s}  training...", end="", flush=True)
            t_probe = time.time()

            probe_cls = SquareProbe if n_classes == 64 else (lambda d: OutcomeProbe(d, n_classes))
            probe = probe_cls(cfg.d_model).to(device)
            feats_dev = feats.to(device)
            task_labels = labels[label_key].to(device)

            _, metrics = train_probe(
                probe, feats_dev, task_labels, cfg=probe_cfg, class_weighted=class_weighted
            )
            results[layer_idx][task_name] = metrics
            print(f"\r  layer {layer_idx}  {task_name:12s}  "
                  f"acc={metrics['acc']:.3f}  macro_acc={metrics['macro_acc']:.3f}  "
                  f"({time.time() - t_probe:.1f}s)")

    print("\n=== Summary: accuracy by layer x task ===")
    header = f"{'layer':>5} | " + " | ".join(f"{t:>12}" for t in TASKS)
    print(header)
    print("-" * len(header))
    for layer_idx in range(cfg.n_layer):
        row = f"{layer_idx:>5} | " + " | ".join(
            f"{results[layer_idx][t]['acc']:>12.3f}" for t in TASKS
        )
        print(row)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results (incl. macro_acc, n_train/n_val) saved to {args.out}")


if __name__ == "__main__":
    main()
