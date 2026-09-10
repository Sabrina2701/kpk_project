"""
A script to see if merging a KPK model with a KRK specialist produce a model
that's still good at both or if one domain's knowledge overwrite the
other.

Important precondition: --finetuned_checkpoint must be a model trained with
train_m1.py --init_from pointing at --base_checkpoint, like in our case where the KRK
specialist was fine-tuned starting from the KPK weights, not trained from
scratch. This is what makes the merging well-defined: base and finetuned live in the same region of weight space,
so a simple per-parameter interpolation is meaningful. 

For each alpha in --alphas, builds merged = base + alpha * (finetuned - base)
and evaluates its next-move top-1 accuracy on held-out KPK and KRK
validation data.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from torch.utils.data import DataLoader, random_split

from dataset import KPKDataset, collate_fn
from model import MoveGPT
from train_m1 import evaluate


def load_val_loader(data_dir, domain, max_len, val_frac, seed, batch_size):
    full_ds = KPKDataset(data_dir, domain=domain, max_len=max_len)
    n_val = max(1, int(len(full_ds) * val_frac))
    n_train = len(full_ds) - n_val
    _train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", type=str, required=True,
                     help="The KPK vanilla checkpoint that was the STARTING POINT for "
                          "fine-tuning (i.e. what --finetuned_checkpoint used as --init_from).")
    ap.add_argument("--finetuned_checkpoint", type=str, required=True,
                     help="The KRK specialist, fine-tuned FROM --base_checkpoint.")
    ap.add_argument("--data_dir_kpk", type=str, required=True)
    ap.add_argument("--data_dir_krk", type=str, required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                     default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", type=str, default="./merge_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    ft_ckpt = torch.load(args.finetuned_checkpoint, map_location=device, weights_only=False)
    cfg = base_ckpt["cfg"]

    if ft_ckpt["cfg"].vocab_size != cfg.vocab_size or ft_ckpt["cfg"].d_model != cfg.d_model \
            or ft_ckpt["cfg"].n_layer != cfg.n_layer:
        raise ValueError(
            "base and finetuned checkpoints have different architectures -- merging requires "
            "identical shapes. Did you fine-tune KRK with --init_from the base checkpoint?"
        )

    base_state = base_ckpt["model"]
    ft_state = ft_ckpt["model"]
    delta = {k: (ft_state[k] - base_state[k]) for k in base_state}

    print("Loading held-out validation sets (same 5%/seed=0 split as training)...")
    kpk_loader = load_val_loader(args.data_dir_kpk, "kpk", cfg.max_len, args.val_frac,
                                  args.seed, args.batch_size)
    krk_loader = load_val_loader(args.data_dir_krk, "krk", cfg.max_len, args.val_frac,
                                  args.seed, args.batch_size)

    model = MoveGPT(cfg).to(device)
    results = []

    print(f"\n{'alpha':>6} | {'KPK top1_acc':>12} | {'KRK top1_acc':>12}")
    print("-" * 36)
    for alpha in args.alphas:
        merged_state = {k: base_state[k] + alpha * delta[k] for k in base_state}
        model.load_state_dict(merged_state)
        model.eval()

        kpk_metrics = evaluate(model, kpk_loader, device)
        krk_metrics = evaluate(model, krk_loader, device)
        results.append({
            "alpha": alpha,
            "kpk_top1_acc": kpk_metrics["top1_acc"],
            "krk_top1_acc": krk_metrics["top1_acc"],
        })
        print(f"{alpha:>6.2f} | {kpk_metrics['top1_acc']:>12.3f} | {krk_metrics['top1_acc']:>12.3f}")

    print("\nalpha=0.0 is the pure base (KPK) model; alpha=1.0 is the pure fine-tuned "
          "(KRK specialist) model -- both are useful reference points, not just endpoints. "
          "A merge is 'successful' if some intermediate alpha keeps BOTH accuracies "
          "reasonably close to their respective alpha=0/alpha=1 ceiling, rather than one "
          "collapsing as soon as alpha moves away from its own endpoint.")

    with open(args.out, "w") as f:
        json.dump({"results": results, "alphas": args.alphas}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
