"""
For each of --n_pairs minimal clean/corrupt pairs (see m5_pairs.py) and
each layer in --layers, isolate that layer's marginal contribution (see
patch.py's patched_run) between the clean and corrupt runs, inject it and
measure how much of a fixed downstream readout is recovered.

Reports mean, median and quartiles of frac_recovered per layer: mean and
median can diverge substantially when a few pairs have an unstable (near-
zero) denominator that blows up the ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader, random_split

from dataset import KPKDataset, collate_fn
from model import MoveGPT
from probe import OutcomeProbe, ProbeTrainConfig, train_probe, collect_all_layers
from tablebase import Tablebase
from m5_pairs import build_minimal_pairs, pair_to_tokens
from patch import run_with_cache, patching_effect


def train_wdl_probes(model, cfg, data_dir, domain, layers, probe_data_frac, batch_size,
                      num_workers, seed, probe_epochs, device):
    """Trains one WDL OutcomeProbe per requested layer, reusing the same
    extraction pipeline as run_probes.py (M3)."""
    print("Training WDL probes at target layers (reusing M3's pipeline)...")
    full_ds = KPKDataset(data_dir, domain=domain, max_len=cfg.max_len)
    n_probe = max(1, int(len(full_ds) * probe_data_frac))
    probe_ds, _rest = random_split(
        full_ds, [n_probe, len(full_ds) - n_probe],
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        probe_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    layer_features, labels = collect_all_layers(model, loader, device=device)
    wdl_labels = labels["wdl"].to(device)

    probe_cfg = ProbeTrainConfig(epochs=probe_epochs)
    probes = {}
    for layer_idx in layers:
        feats = layer_features[layer_idx].to(device)
        probe = OutcomeProbe(cfg.d_model, 5).to(device)
        _, metrics = train_probe(probe, feats, wdl_labels, cfg=probe_cfg, class_weighted=True)
        probes[layer_idx] = probe
        print(f"  layer {layer_idx}  wdl probe  acc={metrics['acc']:.3f}  "
              f"macro_acc={metrics['macro_acc']:.3f}")
    return probes


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _agg(records, min_denom):
    """records: list of patching_effect() dicts for one layer, across all
    pairs. Drops pairs whose |clean_conf - corrupt_conf| < min_denom
    before aggregating. Those ratios are dividing by near-zero and can
    swing wildly without reflecting a real effect."""
    stable = [r for r in records if abs(r["denom"]) >= min_denom and not math.isnan(r["frac_recovered"])]
    dropped = len(records) - len(stable)
    vals = sorted(r["frac_recovered"] for r in stable)
    if not vals:
        return {"n_valid": 0, "n_dropped_unstable": dropped}
    n = len(vals)
    mean = sum(vals) / n
    median = _quantile(vals, 0.5)
    return {
        "n_valid": n,
        "n_dropped_unstable": dropped,
        "mean": mean,
        "median": median,
        "q25": _quantile(vals, 0.25),
        "q75": _quantile(vals, 0.75),
        "frac_above_0.5": sum(v > 0.5 for v in vals) / n,
        "mean_clean_conf": sum(r["clean_conf"] for r in stable) / n,
        "mean_corrupt_conf": sum(r["corrupt_conf"] for r in stable) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tablebase_dir", type=str, required=True)
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--out", type=str, default="./m5_results.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 3, 4, 5],
                     help="Layers to test. Layer 0's isolated-delta patch degenerates to a "
                          "full overwrite (no previous layer to net out) and is expected to "
                          "trivially recover ~100%% for this pair design regardless of what "
                          "layer 0 represents -- see patch.py. Don't read it as a control.")
    ap.add_argument("--n_pairs", type=int, default=250,
                     help="Raised from the non-stratified default of 150: pairs are now also "
                          "split into zugzwang vs. generic branching positions, so each "
                          "stratum needs enough of its own sample.")
    ap.add_argument("--near_dtz_max", type=int, default=5)
    ap.add_argument("--max_games_scanned", type=int, default=20000)
    ap.add_argument("--probe_data_frac", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--probe_epochs", type=int, default=5)
    ap.add_argument("--min_denom", type=float, default=0.05,
                     help="Drop pairs whose |clean_conf - corrupt_conf| falls below this from "
                          "the aggregate stats -- their frac_recovered ratio divides by a "
                          "near-zero number and can swing wildly without reflecting a real "
                          "effect. Raw per-pair values (incl. dropped ones) are still saved "
                          "to --out.")
    ap.add_argument("--seed", type=int, default=0)
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

    probes = train_wdl_probes(
        model, cfg, args.data_dir, args.domain, args.layers, args.probe_data_frac,
        args.batch_size, args.num_workers, args.seed, args.probe_epochs, device,
    )

    print(f"\nBuilding {args.n_pairs} minimal pairs (|dtz| <= {args.near_dtz_max})...")
    tb = Tablebase(args.tablebase_dir)
    pairs = build_minimal_pairs(
        args.data_dir, tb, args.n_pairs, near_dtz_max=args.near_dtz_max,
        seed=args.seed, max_games_scanned=args.max_games_scanned,
    )
    if not pairs:
        raise RuntimeError("No minimal pairs found -- relax --near_dtz_max or raise "
                            "--max_games_scanned.")

    print(f"\nRunning patching over {len(pairs)} pairs x {len(args.layers)} layers...")
    t0 = time.time()
    per_layer_results = {layer_idx: [] for layer_idx in args.layers}
    per_layer_by_group = {layer_idx: {"zugzwang": [], "generic": []} for layer_idx in args.layers}
    final_layer = args.layers[-1]
    target_probe = probes[final_layer]

    for pair_idx, pair in enumerate(pairs):
        clean_tokens, corrupt_tokens, attn, pos_idx = pair_to_tokens(pair)
        clean_tokens, corrupt_tokens, attn = (
            clean_tokens.to(device), corrupt_tokens.to(device), attn.to(device)
        )
        clean_run = run_with_cache(model, clean_tokens, attn)
        corrupt_run = run_with_cache(model, corrupt_tokens, attn)
    
        target_class = -pair.wdl_a + 2
        group = "zugzwang" if pair.is_zugzwang else "generic"

        for layer_idx in args.layers:
            eff = patching_effect(
                model, target_probe, clean_run, corrupt_run, layer_idx, pos_idx,
                target_class, readout_layer=final_layer
            )
            per_layer_results[layer_idx].append(eff)
            per_layer_by_group[layer_idx][group].append(eff)

        if (pair_idx + 1) % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (pair_idx + 1) * (len(pairs) - pair_idx - 1)
            print(f"    pair {pair_idx + 1}/{len(pairs)}  "
                  f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

    summary = {layer_idx: _agg(per_layer_results[layer_idx], args.min_denom)
               for layer_idx in args.layers}
    summary_by_group = {
        layer_idx: {
            group: _agg(per_layer_by_group[layer_idx][group], args.min_denom)
            for group in ("zugzwang", "generic")
        }
        for layer_idx in args.layers
    }

    n_zz_total = sum(p.is_zugzwang for p in pairs)
    print(f"\n=== Summary: M5 patching, {len(pairs)} pairs, readout at layer {final_layer}, "
          f"mean |dtz| at branch = {sum(p.dtz_before for p in pairs) / len(pairs):.1f} ===")
    print("(layer 0 = degenerate full-overwrite case, see --layers help; not a real control)")
    header = (f"{'layer':>5} | {'mean':>7} | {'median':>7} | {'q25':>7} | {'q75':>7} | "
              f"{'>0.5':>6} | {'n_ok':>5} | {'n_drop':>6}")
    print(header)
    print("-" * len(header))
    for layer_idx in args.layers:
        s = summary[layer_idx]
        if s["n_valid"] == 0:
            print(f"{layer_idx:>5} | all pairs dropped as unstable ({s['n_dropped_unstable']})")
            continue
        print(f"{layer_idx:>5} | {s['mean']:>7.3f} | {s['median']:>7.3f} | {s['q25']:>7.3f} | "
              f"{s['q75']:>7.3f} | {s['frac_above_0.5']:>6.2f} | {s['n_valid']:>5} | "
              f"{s['n_dropped_unstable']:>6}")

    print(f"\n=== Stratified: zugzwang ({n_zz_total} pairs) vs generic "
          f"({len(pairs) - n_zz_total} pairs) branching positions ===")
    header2 = (f"{'layer':>5} | {'zz median':>9} | {'zz n_ok':>7} | "
               f"{'gen median':>10} | {'gen n_ok':>8}")
    print(header2)
    print("-" * len(header2))
    for layer_idx in args.layers:
        zz, gen = summary_by_group[layer_idx]["zugzwang"], summary_by_group[layer_idx]["generic"]
        zz_med = f"{zz['median']:.3f}" if zz.get("n_valid", 0) > 0 else "n/a"
        gen_med = f"{gen['median']:.3f}" if gen.get("n_valid", 0) > 0 else "n/a"
        print(f"{layer_idx:>5} | {zz_med:>9} | {zz.get('n_valid', 0):>7} | "
              f"{gen_med:>10} | {gen.get('n_valid', 0):>8}")

    with open(args.out, "w") as f:
        json.dump({
            "summary": summary,
            "summary_by_group": summary_by_group,
            "per_pair": {str(l): per_layer_results[l] for l in args.layers},
            "n_pairs": len(pairs),
            "n_zugzwang_pairs": n_zz_total,
            "layers": args.layers,
            "near_dtz_max": args.near_dtz_max,
            "min_denom": args.min_denom,
        }, f, indent=2)
    print(f"\nFull results (incl. raw per-pair values) saved to {args.out}")


if __name__ == "__main__":
    main()
