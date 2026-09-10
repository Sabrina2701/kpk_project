"""
This script generates synthetic chess endgame datasets (KPK or KRK) using Syzygy tablebases and self-play, saving the output into resumable and sharded JSONL files.
Sharded and resumable means that it writes one JSON-lines shard at a time and records progress in 'out/manifest.json'. 
If Colab disconnects mid-run, re-running the same command skips shards already on disk instead of starting over.

Before committing to a full run, this script times the first shard and extrapolates total time, printing a warning if the full run 
would exceed '--time_budget_min'. First I run with a small '--n_games' to sanity-check throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import time

#formatted output filename for each shard
def _shard_path(out_dir: str, shard_idx: int) -> str:
    return os.path.join(out_dir, f"shard_{shard_idx:05d}.jsonl")

#file path for manifest.json inside the output directory
def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")

#reads manifest.json if it exists or returns a default progress tracking dictionary
def _load_manifest(out_dir: str) -> dict:
    path = _manifest_path(out_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_shards": 0, "games_written": 0, "domain": None}

#writes current generation progress back to manifest.json
def _save_manifest(out_dir: str, manifest: dict) -> None:
    with open(_manifest_path(out_dir), "w") as f:
        json.dump(manifest, f, indent=2)

#converts a game record object into a dictionary containing FEN, moves, WDL, DTZ and game outcome
def _record_to_json(rec) -> dict:
    return {
        "start_fen": rec.start_fen,
        "moves": rec.moves_uci,
        "wdl": rec.wdl_after,
        "dtz": rec.dtz_after,
        "outcome": rec.outcome,
    }

#configures command-line options including endgame domain (kpk/krk), total games requested (--n_games), 
#output directory (--out), shard size (--shard_size), Syzygy directory (--tablebase_dir), random seed (--seed) and time limit warning threshold (--time_budget_min)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--n_games", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--shard_size", type=int, default=5000)
    ap.add_argument("--tablebase_dir", type=str, default="./syzygy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_budget_min", type=float, default=60.0)
    args = ap.parse_args()

    #deferred imports so the script can be configured before loading heavy dependencies
    from tablebase import Tablebase
    from selfplay import generate_dataset, SelfPlayConfig

    #creates the output directory and verifies that any existing dataset in that folder matches the specified --domain
    os.makedirs(args.out, exist_ok=True)
    manifest = _load_manifest(args.out)
    if manifest["domain"] not in (None, args.domain):
        raise RuntimeError(
            f"{args.out} already contains a '{manifest['domain']}' dataset; "
            f"use a different --out for '{args.domain}'."
        )
    manifest["domain"] = args.domain

    tb = Tablebase(args.tablebase_dir)
    cfg = SelfPlayConfig()

    #calculates total shards (n_shards), checks completed_shards from the manifest and skips previously completed shards if execution was interrupted
    n_shards = (args.n_games + args.shard_size - 1) // args.shard_size
    start_shard = manifest["completed_shards"]
    if start_shard > 0:
        print(f"Resuming: {start_shard}/{n_shards} shards already on disk, skipping them.")

    for shard_idx in range(start_shard, n_shards):
        shard_seed = args.seed + shard_idx  # distinct, reproducible seed per shard
        remaining = min(args.shard_size, args.n_games - shard_idx * args.shard_size)

        t0 = time.time()
        records = generate_dataset(remaining, tb, seed=shard_seed, cfg=cfg, domain=args.domain)
        elapsed = time.time() - t0

        with open(_shard_path(args.out, shard_idx), "w") as f:
            for rec in records:
                f.write(json.dumps(_record_to_json(rec)) + "\n")

        manifest["completed_shards"] = shard_idx + 1
        manifest["games_written"] = manifest.get("games_written", 0) + remaining
        _save_manifest(args.out, manifest)

        games_per_sec = remaining / max(elapsed, 1e-9)
        eta_min = (args.n_games - manifest["games_written"]) / max(games_per_sec, 1e-9) / 60
        print(
            f"[shard {shard_idx + 1}/{n_shards}] {remaining} games in {elapsed:.1f}s "
            f"({games_per_sec:.1f} games/s) | ETA for remaining shards: {eta_min:.1f} min"
        )
        if shard_idx == start_shard and eta_min > args.time_budget_min:
            print(
                f"WARNING: projected total time ({eta_min:.1f} min) exceeds "
                f"--time_budget_min ({args.time_budget_min}). Consider reducing "
                f"--n_games or profiling Tablebase probing before continuing."
            )

    print(f"Done. {manifest['games_written']} games across {n_shards} shards in {args.out}/")


if __name__ == "__main__":
    main()
