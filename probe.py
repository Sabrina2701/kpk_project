"""
Linear probes over frozen MoveGPT hidden states, a nonlinear probe would blur the M2 vs M3 distinction, since
a powerful enough nonlinear probe can sometimes recover information that
required real computation to extract, defeating the point of M3.

Three probe families:

  SquareProbe   : d_model -> 64 classes (which square a piece is on). Used
                  for M2 (own_king, other_king, special_sq) at a given ply.

  OutcomeProbe  : d_model -> {DTZ-bucket classes} or {WDL classes}. Used
                  for M3 (in-distribution probing: random train/val split
                  over the whole DTZ range).

  DistanceProbe : d_model -> 1 (regresses raw |DTZ|, not bucketed). Used
                  for M4.
                 
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquareProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 64)

    def forward(self, h):
        return self.fc(h)


class DistanceProbe(nn.Module):
    
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)


class DistanceProbeMLP(nn.Module):
    
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class OutcomeProbe(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, h):
        return self.fc(h)


def dtz_to_bucket(dtz: torch.Tensor, edges=(0, 2, 5, 10, 20)) -> torch.Tensor:
    
    ad = dtz.abs()
    bucket = torch.zeros_like(ad)
    for e in edges:
        bucket += (ad > e).long()
    return bucket





@dataclass
class ProbeTrainConfig:
    epochs: int = 5
    lr: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 4096  


def train_probe(probe: nn.Module, features: torch.Tensor, labels: torch.Tensor,
                 cfg: ProbeTrainConfig = ProbeTrainConfig(), val_frac: float = 0.15,
                 class_weighted: bool = False):
    """features: [N, d_model] (already detached/frozen), labels: [N] long.
    Ignores label == -1 (padding).

    class_weighted=True applies inverse-frequency weights (computed from the
    train split only, to avoid leaking val-set class balance into training)
    to the cross-entropy loss.
"""
    mask = labels != -1
    features, labels = features[mask], labels[mask]
    n = features.shape[0]
    n_val = int(n * val_frac)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    weight = None
    if class_weighted:
        n_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels[train_idx], minlength=n_classes).float().clamp(min=1)
        weight = (1.0 / counts)
        weight = weight / weight.mean()  # normalize so the loss scale doesn't drift

    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _epoch in range(cfg.epochs):
        probe.train()
        for i in range(0, len(train_idx), cfg.batch_size):
            idx = train_idx[i:i + cfg.batch_size]
            logits = probe(features[idx])
            loss = F.cross_entropy(logits, labels[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(features[val_idx])
        val_preds = val_logits.argmax(-1)
        val_labels = labels[val_idx]
        acc = (val_preds == val_labels).float().mean().item()

        per_class_recall = []
        for c in val_labels.unique():
            class_mask = val_labels == c
            per_class_recall.append((val_preds[class_mask] == c).float().mean().item())
        macro_acc = sum(per_class_recall) / len(per_class_recall)

    return probe, {"acc": acc, "macro_acc": macro_acc, "n_train": len(train_idx), "n_val": n_val}


def train_distance_probe_ood(probe: nn.Module, features: torch.Tensor, dtz_raw: torch.Tensor,
                              train_mask: torch.Tensor, test_mask: torch.Tensor,
                              cfg: ProbeTrainConfig = ProbeTrainConfig(), val_frac: float = 0.15):
    
    valid = ~torch.isnan(dtz_raw)
    target = dtz_raw.abs()

    train_pool = (train_mask & valid).nonzero(as_tuple=True)[0]
    test_idx = (test_mask & valid).nonzero(as_tuple=True)[0]

    perm = train_pool[torch.randperm(len(train_pool))]
    n_val = int(len(perm) * val_frac)
    indist_val_idx, train_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _epoch in range(cfg.epochs):
        probe.train()
        ep_perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(ep_perm), cfg.batch_size):
            idx = ep_perm[i:i + cfg.batch_size]
            pred = probe(features[idx])
            loss = F.smooth_l1_loss(pred, target[idx])  # Huber: robust to the long DTZ tail
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        train_mean = target[train_idx].mean().item()

        indist_pred = probe(features[indist_val_idx])
        indist_mae = (indist_pred - target[indist_val_idx]).abs().mean().item()

        test_pred = probe(features[test_idx])
        test_mae = (test_pred - target[test_idx]).abs().mean().item()
        baseline_mae = (torch.full_like(target[test_idx], train_mean) - target[test_idx]).abs().mean().item()

    return probe, {
        "test_mae": test_mae,
        "baseline_mae": baseline_mae,
        "indist_mae": indist_mae,
        "n_train": len(train_idx), "n_indist_val": len(indist_val_idx), "n_test": len(test_idx),
        "train_mean_dtz": train_mean,
    }


@torch.no_grad()
def collect_hidden_states(model, loader, layer_idx: int, device: str = "cpu"):
    """Runs the model over a DataLoader built from dataset.py's
    collate_fn and returns flattened (hidden, own_king, other_king,
    special_sq, dtz_bucket, wdl) tensors ready for train_probe(). Padding
    positions are marked with label -1 and filtered inside train_probe().

    """
    model.eval().to(device)
    all_h, all_ok, all_otk, all_sp, all_dtz, all_wdl = [], [], [], [], [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        attn = batch["attn_mask"].to(device)
        h = model.hidden_at(tokens, layer_idx, attn_mask=attn)  # [B,T,d]

        pad = ~batch["attn_mask"]
        ok = batch["own_king"].clone(); ok[pad] = -1
        otk = batch["other_king"].clone(); otk[pad] = -1
        sp = batch["special_sq"].clone(); sp[pad] = -1
        wdl = (batch["wdl"] + 2).clone(); wdl[pad] = -1  # shift {-2..2} -> {0..4}
        dtz_b = dtz_to_bucket(batch["dtz"]); dtz_b[pad] = -1

        B, T, D = h.shape
        all_h.append(h.reshape(B * T, D).cpu())
        all_ok.append(ok.reshape(B * T).cpu())
        all_otk.append(otk.reshape(B * T).cpu())
        all_sp.append(sp.reshape(B * T).cpu())
        all_dtz.append(dtz_b.reshape(B * T).cpu())
        all_wdl.append(wdl.reshape(B * T).cpu())

    return (
        torch.cat(all_h), torch.cat(all_ok), torch.cat(all_otk),
        torch.cat(all_sp), torch.cat(all_dtz), torch.cat(all_wdl),
    )


@torch.no_grad()
def collect_all_layers(model, loader, device: str = "cpu", progress_every: int = 10):
    """Like collect_hidden_states, but extracts every layer's residual
    stream in a single pass over the loader (one forward call per batch,
    return_hidden=True, then slice per layer) instead of one full dataset
    pass per layer. 

    """
    import time

    model.eval().to(device)
    n_layer = model.cfg.n_layer
    per_layer_h = [[] for _ in range(n_layer)]
    all_ok, all_otk, all_sp, all_dtz, all_wdl, all_dtz_raw = [], [], [], [], [], []

    n_batches = len(loader)
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        tokens = batch["tokens"].to(device)
        attn = batch["attn_mask"].to(device)
        _, hidden_states = model(tokens, attn_mask=attn, return_hidden=True)  # list[n_layer] of [B,T,D]

        pad = ~batch["attn_mask"]
        ok = batch["own_king"].clone(); ok[pad] = -1
        otk = batch["other_king"].clone(); otk[pad] = -1
        sp = batch["special_sq"].clone(); sp[pad] = -1
        wdl = (batch["wdl"] + 2).clone(); wdl[pad] = -1
        dtz_b = dtz_to_bucket(batch["dtz"]); dtz_b[pad] = -1
        dtz_raw = batch["dtz"].float().clone(); dtz_raw[pad] = float("nan")

        B, T = tokens.shape
        all_ok.append(ok.reshape(B * T).cpu())
        all_otk.append(otk.reshape(B * T).cpu())
        all_sp.append(sp.reshape(B * T).cpu())
        all_dtz.append(dtz_b.reshape(B * T).cpu())
        all_wdl.append(wdl.reshape(B * T).cpu())
        all_dtz_raw.append(dtz_raw.reshape(B * T).cpu())
        for l in range(n_layer):
            per_layer_h[l].append(hidden_states[l].reshape(B * T, -1).cpu())

        if progress_every and (batch_idx + 1) % progress_every == 0:
            elapsed = time.time() - t0
            per_batch = elapsed / (batch_idx + 1)
            eta = per_batch * (n_batches - batch_idx - 1)
            print(f"    batch {batch_idx + 1}/{n_batches}  "
                  f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

    labels = {
        "own_king": torch.cat(all_ok),
        "other_king": torch.cat(all_otk),
        "special_sq": torch.cat(all_sp),
        "dtz_bucket": torch.cat(all_dtz),
        "wdl": torch.cat(all_wdl),
        "dtz_raw": torch.cat(all_dtz_raw),
    }
    layer_features = [torch.cat(h) for h in per_layer_h]
    return layer_features, labels
