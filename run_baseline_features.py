"""
Trains the same probe types used in M3 (WDL classification) and M4
(DTZ regression, near->far generalization) but on hand-computed features
extracted directly from the board: own/other king file+rank, pawn/rook
file+rank, pairwise Chebyshev distances, distance to promotion rank, side
to move.

"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from probe import (
    OutcomeProbe, DistanceProbe, DistanceProbeMLP, ProbeTrainConfig,
    train_probe, train_distance_probe_ood,
)


def _load_games(data_dir: str):
    games = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".jsonl"):
            with open(os.path.join(data_dir, fname)) as f:
                for line in f:
                    games.append(json.loads(line))
    return games


def extract_hand_features(data_dir: str, domain: str, n_games: int, seed: int = 0):
    """Returns (X, dtz_raw, wdl)
    collect_all_layers (wdl shifted to {0..4}, dtz_raw signed float) so
    the exact same train_probe / train_distance_probe_ood functions apply
    unchanged.

    Feature vector per position (11 dims, then z-scored):
      own_king_file, own_king_rank, other_king_file, other_king_rank,
      special_file, special_rank, king_king_dist, own_to_special_dist,
      other_to_special_dist, dist_to_promotion, side_to_move
    """
    import chess

    rng = random.Random(seed)
    games = _load_games(data_dir)
    rng.shuffle(games)
    games = games[:n_games]

    piece_type = chess.PAWN if domain == "kpk" else chess.ROOK
    feats, dtz_list, wdl_list = [], [], []

    def record(board, dtz_val, wdl_val):
        stm = board.turn
        own_k, other_k = board.king(stm), board.king(not stm)
        squares = board.pieces(piece_type, chess.WHITE) | board.pieces(piece_type, chess.BLACK)
        sp = next(iter(squares)) if squares else None
        if own_k is None or other_k is None or sp is None:
            return
        ok_f, ok_r = chess.square_file(own_k), chess.square_rank(own_k)
        otk_f, otk_r = chess.square_file(other_k), chess.square_rank(other_k)
        sp_f, sp_r = chess.square_file(sp), chess.square_rank(sp)
        promo_dist = (7 - sp_r) if domain == "kpk" else 0  # pawn is always White, promotes at rank 8
        feats.append([
            ok_f, ok_r, otk_f, otk_r, sp_f, sp_r,
            chess.square_distance(own_k, other_k),
            chess.square_distance(own_k, sp),
            chess.square_distance(other_k, sp),
            promo_dist,
            1.0 if stm == chess.WHITE else 0.0,
        ])
        dtz_list.append(dtz_val)
        wdl_list.append(wdl_val)

    for g in games:
        board = chess.Board(g["start_fen"])
        moves, dtz_seq, wdl_seq = g["moves"], g["dtz"], g["wdl"]
        record(board, dtz_seq[0] if dtz_seq else 0, wdl_seq[0] if wdl_seq else 0)
        for i, uci in enumerate(moves):
            board.push_uci(uci)
            record(board, dtz_seq[i] if i < len(dtz_seq) else 0,
                   wdl_seq[i] if i < len(wdl_seq) else 0)

    X = torch.tensor(feats, dtype=torch.float32)
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)  # z-score, helps a linear probe
    dtz_raw = torch.tensor(dtz_list, dtype=torch.float32)
    wdl = torch.tensor(wdl_list, dtype=torch.long) + 2
    return X, dtz_raw, wdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--n_games", type=int, default=4000)
    ap.add_argument("--train_max_dtz", type=int, default=5)
    ap.add_argument("--test_min_dtz", type=int, default=10)
    ap.add_argument("--probe_epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="./baseline_features_results.json")
    args = ap.parse_args()

    print(f"Extracting hand-crafted features from up to {args.n_games} games...")
    X, dtz_raw, wdl = extract_hand_features(args.data_dir, args.domain, args.n_games, args.seed)
    print(f"  {X.shape[0]} positions, {X.shape[1]} hand-crafted features")

    probe_cfg = ProbeTrainConfig(epochs=args.probe_epochs)
    results = {}

    print("\n--- M3-style: WDL classification on hand features ---")
    wdl_probe = OutcomeProbe(X.shape[1], 5)
    _, wdl_metrics = train_probe(wdl_probe, X, wdl, cfg=probe_cfg, class_weighted=True)
    print(f"  acc={wdl_metrics['acc']:.3f}  macro_acc={wdl_metrics['macro_acc']:.3f}")
    results["wdl"] = wdl_metrics

    print(f"\n--- M4-style: near(|dtz|<={args.train_max_dtz}) -> "
          f"far(|dtz|>={args.test_min_dtz}) generalization on hand features ---")
    abs_dtz = dtz_raw.abs()
    train_mask = abs_dtz <= args.train_max_dtz
    test_mask = abs_dtz >= args.test_min_dtz
    print(f"  near: {train_mask.sum().item()}  far: {test_mask.sum().item()}")

    for name, cls in (("linear", DistanceProbe), ("mlp", DistanceProbeMLP)):
        probe = cls(X.shape[1])
        _, m = train_distance_probe_ood(probe, X, dtz_raw, train_mask, test_mask, cfg=probe_cfg)
        gap = m["baseline_mae"] - m["test_mae"]
        print(f"  {name:6s}  test_mae={m['test_mae']:.2f}  baseline_mae={m['baseline_mae']:.2f}  "
              f"indist_mae={m['indist_mae']:.2f}  (beats baseline by {gap:+.2f})")
        results[name] = m

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print("\nCompare these numbers directly against run_probes.py's wdl row and "
          "run_m4.py's linear/mlp rows for the SAME layer -- same probes, same "
          "training procedure, only the input differs.")


if __name__ == "__main__":
    main()
