"""
M5: activation patching.

The core experimental design (standard causal-tracing recipe, as used
for Othello-GPT's causal interventions and ROME-style factual tracing):

--> Build a minimal pair of positions: `clean` and `corrupt`, identical
     move-prefix length, differing in exactly one tempo-relevant detail so
     that the tablebase DTZ/outcome differs meaningfully between them.

--> Cache the clean run's hidden states at every layer.

--> Run the model on `corrupt`, but at a chosen layer isolate and inject
     only that layer's marginal contribution (delta) from the clean run,
     keeping everything else corrupt

--> Measure how far this patched run's output moves toward the clean
     run's output.

A causal effect concentrated at a specific layer (rather than spread
evenly or absent) is the evidence needed to say the model performs
look-ahead rather than pattern-matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Run:
    tokens: torch.Tensor        # [1, T]
    attn_mask: torch.Tensor     # [1, T]
    logits: torch.Tensor        # [1, T, vocab]
    hidden: list                # n_layer tensors, each [1, T, d_model]


@torch.no_grad()
def run_with_cache(model, tokens: torch.Tensor, attn_mask: torch.Tensor) -> Run:
    logits, hidden = model(tokens, attn_mask=attn_mask, return_hidden=True)
    return Run(tokens=tokens, attn_mask=attn_mask, logits=logits, hidden=hidden)


@torch.no_grad()
def patched_run(model, corrupt: Run, clean: Run, layer_idx: int, pos_idx: int):
    """
    vector = corrupt.hidden[layer_idx] + (clean_delta - corrupt_delta)
    where delta = hidden[layer_idx] - hidden[layer_idx-1]. 
    """
    device = clean.hidden[layer_idx].device
    batch_idx = torch.zeros(1, dtype=torch.long, device=device)
    pos = torch.tensor([pos_idx], dtype=torch.long, device=device)

    if layer_idx == 0:
        clean_delta = clean.hidden[0][0, pos_idx, :]
        corrupt_delta = corrupt.hidden[0][0, pos_idx, :]
    else:
        clean_delta = clean.hidden[layer_idx][0, pos_idx, :] - clean.hidden[layer_idx - 1][0, pos_idx, :]
        corrupt_delta = corrupt.hidden[layer_idx][0, pos_idx, :] - corrupt.hidden[layer_idx - 1][0, pos_idx, :]

    delta_diff = clean_delta - corrupt_delta
    vector = (corrupt.hidden[layer_idx][0, pos_idx, :] + delta_diff).unsqueeze(0)  # [1, d_model]

    patch = {layer_idx: (batch_idx, pos, vector)}
    logits, hidden = model(
        corrupt.tokens, attn_mask=corrupt.attn_mask, return_hidden=True, patch=patch
    )
    return logits, hidden


def probe_readout(probe, hidden: torch.Tensor, pos_idx: int) -> torch.Tensor:
    
    with torch.no_grad():
        return F.softmax(probe(hidden[0, pos_idx, :].unsqueeze(0)), dim=-1).squeeze(0)


def patching_effect(model, probe, clean: Run, corrupt: Run, layer_idx: int, pos_idx: int,
                     target_class: int, readout_layer: int = -1) -> dict:
 
    n_layer = len(clean.hidden)
    resolved_readout = readout_layer if readout_layer >= 0 else n_layer + readout_layer
    if resolved_readout < layer_idx:
        raise ValueError(
            f"readout_layer ({readout_layer}) must be >= layer_idx ({layer_idx})."
        )

    clean_conf = probe_readout(probe, clean.hidden[readout_layer], pos_idx)[target_class].item()
    corrupt_conf = probe_readout(probe, corrupt.hidden[readout_layer], pos_idx)[target_class].item()

    patched_logits, patched_hidden = patched_run(model, corrupt, clean, layer_idx, pos_idx)
    patched_conf = probe_readout(probe, patched_hidden[readout_layer], pos_idx)[target_class].item()

    denom = (clean_conf - corrupt_conf)
    recovered = (patched_conf - corrupt_conf) / denom if abs(denom) > 1e-4 else float("nan")
    return {
        "clean_conf": clean_conf,
        "corrupt_conf": corrupt_conf,
        "patched_conf": patched_conf,
        "denom": denom,
        "frac_recovered": recovered,
    }


def sweep(model, probe, clean: Run, corrupt: Run, target_class: int, readout_layer: int = -1):
    """Layer x position grid of `frac_recovered` evaluated at readout_layer."""
    n_layer = len(clean.hidden)
    T = clean.tokens.shape[1]
    grid = torch.full((n_layer, T), float("nan"))
    for layer_idx in range(n_layer):
        for pos_idx in range(T):
            eff = patching_effect(model, probe, clean, corrupt, layer_idx, pos_idx,
                                   target_class, readout_layer=readout_layer)
            grid[layer_idx, pos_idx] = eff["frac_recovered"]
    return grid
