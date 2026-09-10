"""
Every epoch saves model + optimizer + scheduler state + epoch number to <out_dir>/last.pt (always
overwritten) and to <out_dir>/best.pt (only when validation loss improves).
If training stops, rerun the same command with --resume and it picks up from
last.pt instead of starting over.

"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import KPKDataset, collate_fn
from model import MoveGPT, GPTConfig
from probe import dtz_to_bucket
from tokenizer import VOCAB, PAD


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    aux_correct = {"wdl": 0, "dtz": 0}
    aux_count = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["next_tokens"].to(device)
        attn = batch["attn_mask"].to(device)

        if model.cfg.aux_heads:
            logits, hidden = model(tokens, attn_mask=attn, return_hidden=True)
            wdl_logits, dtz_logits = model.aux_logits(hidden[-1])
            valid = attn.reshape(-1)
            wdl_target = (batch["wdl"] + 2).to(device).reshape(-1)
            dtz_target = dtz_to_bucket(batch["dtz"]).to(device).reshape(-1)
            aux_correct["wdl"] += (wdl_logits.reshape(-1, wdl_logits.size(-1)).argmax(-1)[valid]
                                    == wdl_target[valid]).sum().item()
            aux_correct["dtz"] += (dtz_logits.reshape(-1, dtz_logits.size(-1)).argmax(-1)[valid]
                                    == dtz_target[valid]).sum().item()
            aux_count += valid.sum().item()
        else:
            logits = model(tokens, attn_mask=attn)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )

        mask = targets != -100
        preds = logits.argmax(-1)
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_count += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    out = {
        "loss": total_loss / max(total_count, 1),
        "ppl": math.exp(min(total_loss / max(total_count, 1), 20)),  # cap to avoid overflow
        "top1_acc": total_correct / max(total_count, 1),
    }
    if model.cfg.aux_heads:
        out["wdl_acc"] = aux_correct["wdl"] / max(aux_count, 1)
        out["dtz_acc"] = aux_correct["dtz"] / max(aux_count, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./checkpoints")
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--multitask", action="store_true",
                     help="Add auxiliary WDL/DTZ-bucket prediction heads on the last layer's "
                          "hidden state, trained jointly with the main next-move objective. "
                          "Tests whether the weak/distributed representation found in M3-M5 "
                          "improves when the model is explicitly incentivized to represent "
                          "it, instead of picking it up only as a next-move side effect. "
                          "Changes the model's architecture (see GPTConfig.aux_heads) -- for "
                          "a clean comparison, train this from scratch rather than "
                          "--init_from a vanilla checkpoint.")
    ap.add_argument("--aux_weight_wdl", type=float, default=1.0)
    ap.add_argument("--aux_weight_dtz", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                     help="Continue training in THIS --out_dir: reloads model+optimizer+"
                          "scheduler+epoch count from <out_dir>/last.pt. Use after a Colab "
                          "disconnect, same run.")
    ap.add_argument("--init_from", type=str, default=None,
                     help="Start a NEW run (fresh optimizer/scheduler/epoch count, new "
                          "--out_dir) but seed the model weights from this checkpoint path "
                          "(e.g. an earlier run's best.pt). Mutually exclusive with --resume.")
    args = ap.parse_args()
    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive: --resume continues "
                          "the same run's optimizer/scheduler state, --init_from starts a new "
                          "run seeded only with another run's weights.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    print("Loading dataset (replays every game once to build M2/M3 labels -- "
          "can take a bit on a large dataset, this only happens once per run)...")
    t0 = time.time()
    full_ds = KPKDataset(args.data_dir, domain=args.domain, max_len=args.max_len)
    print(f"  {len(full_ds)} games loaded in {time.time() - t0:.1f}s")

    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  train={n_train}  val={n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    cfg = GPTConfig(
        vocab_size=VOCAB.size, n_layer=args.n_layer, n_head=args.n_head,
        d_model=args.d_model, max_len=args.max_len, aux_heads=args.multitask,
    )
    model = MoveGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg}  ({n_params / 1e6:.2f}M params)")

    if args.init_from:
        seed_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        seed_cfg = seed_ckpt.get("cfg")
        if seed_cfg is not None and seed_cfg.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"--init_from checkpoint was trained with vocab_size={seed_cfg.vocab_size}, "
                f"but the current tokenizer has vocab_size={cfg.vocab_size} -- the embedding/"
                f"head tables are a different shape and weights cannot be loaded as-is. "
                f"(This can happen if tokenizer.py changed between runs.)"
            )
        model.load_state_dict(seed_ckpt["model"])
        print(f"  seeded model weights from {args.init_from} "
              f"(fresh optimizer/scheduler/epoch count)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps)
    )

    start_epoch = 0
    best_val_loss = float("inf")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    best_ckpt = os.path.join(args.out_dir, "best.pt")

    if args.resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from {last_ckpt}: starting at epoch {start_epoch}, "
              f"best_val_loss so far = {best_val_loss:.4f}")
    elif args.resume:
        print(f"--resume passed but no checkpoint found at {last_ckpt}; starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss, running_count = 0.0, 0

        for i, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device)
            targets = batch["next_tokens"].to(device)
            attn = batch["attn_mask"].to(device)

            if args.multitask:
                logits, hidden = model(tokens, attn_mask=attn, return_hidden=True)
                main_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )

                wdl_logits, dtz_logits = model.aux_logits(hidden[-1])
                valid = attn.reshape(-1)  # padding positions have no meaningful wdl/dtz target
                wdl_target = (batch["wdl"] + 2).to(device).reshape(-1)
                dtz_target = dtz_to_bucket(batch["dtz"]).to(device).reshape(-1)
                wdl_loss = F.cross_entropy(
                    wdl_logits.reshape(-1, wdl_logits.size(-1))[valid], wdl_target[valid]
                )
                dtz_loss = F.cross_entropy(
                    dtz_logits.reshape(-1, dtz_logits.size(-1))[valid], dtz_target[valid]
                )
                loss = main_loss + args.aux_weight_wdl * wdl_loss + args.aux_weight_dtz * dtz_loss
            else:
                logits = model(tokens, attn_mask=attn)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            running_loss += loss.item()
            running_count += 1

            if i == 5: 
                per_batch = (time.time() - t0) / 6
                eta_epoch_min = per_batch * len(train_loader) / 60
                print(f"  [throughput check] ~{per_batch:.2f}s/batch -> "
                      f"~{eta_epoch_min:.1f} min for this epoch's training pass")

        train_loss = running_loss / max(running_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        aux_str = ""
        if args.multitask:
            aux_str = f" | val_wdl_acc={val_metrics['wdl_acc']:.3f} val_dtz_acc={val_metrics['dtz_acc']:.3f}"
        print(
            f"epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['ppl']:.2f} "
            f"val_top1_acc={val_metrics['top1_acc']:.3f}{aux_str} | {elapsed:.1f}s"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            "cfg": cfg,
        }
        torch.save(ckpt, last_ckpt)  

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(ckpt, best_ckpt)
            print(f"  new best (val_loss={best_val_loss:.4f}) -> saved {best_ckpt}")

    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoints in {args.out_dir}/")


if __name__ == "__main__":
    main()
