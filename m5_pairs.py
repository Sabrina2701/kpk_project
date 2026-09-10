"""
Builds minimal clean/corrupt move-sequence pairs for M5 activation patching.

--> pair = (prefix, move_a, move_b) where `prefix` is a real move history
taken from generated self-play games.
--> move_a/move_b are two different legal moves from the position at the end of that prefix, chosen so their
resulting WDL differs. For example move_a preserves a win (WDL=+2), move_b turns it into a draw (0) or a
loss (-2). 
-->clean_tokens = prefix + [move_a]
-->corrupt_tokens = prefix + [move_b]
identical history, identical length, differing only in the final move and its consequence.

"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass


@dataclass
class MinimalPair:
    prefix_uci: list
    move_a_uci: str   # clean: preserves the better outcome
    move_b_uci: str   # corrupt: worse outcome
    wdl_a: int         # WDL for the mover after move_a 
    wdl_b: int         # WDL for the mover after move_b 
    dtz_before: int    # |DTZ| at the branching position before either move
    is_zugzwang: bool  # True if flipping side-to-move at the branching position
                        # flips who's winning (the same test validated in
                        # validate.py's check_critical_sampler_effect).


def _load_games(data_dir: str):
    games = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".jsonl"):
            with open(os.path.join(data_dir, fname)) as f:
                for line in f:
                    games.append(json.loads(line))
    return games


def build_minimal_pairs(data_dir: str, tablebase, n_pairs: int, near_dtz_max: int = 5,
                         seed: int = 0, max_games_scanned: int = 20000,
                         progress_every: int = 2000):
    """Scans real generated games for branching points matching the
    near-conversion + differing-outcome criteria. Stops once
    n_pairs are found or max_games_scanned games have been examined. Prints progress periodically since this can take
    a huge number of games to fill a large n_pairs target."""
    import chess

    def _is_zugzwang(board):
      
        w1 = tablebase.wdl(board)
        white1 = w1 if board.turn == chess.WHITE else -w1
        flipped = board.copy()
        flipped.turn = not flipped.turn
        if not flipped.is_valid():
            return False  
        w2 = tablebase.wdl(flipped)
        white2 = w2 if flipped.turn == chess.WHITE else -w2
        return (white1 > 0) != (white2 > 0)

    rng = random.Random(seed)
    games = _load_games(data_dir)
    rng.shuffle(games)

    pairs = []
    scanned = 0
    t0 = time.time()

    for g in games:
        if len(pairs) >= n_pairs or scanned >= max_games_scanned:
            break
        scanned += 1
        moves = g["moves"]
        if len(moves) < 2:
            continue

        candidate_plies = sorted(rng.sample(range(len(moves)), min(3, len(moves))))
        board = chess.Board(g["start_fen"])

        for ply in range(max(candidate_plies) + 1):
            if ply in candidate_plies and list(board.legal_moves):
                try:
                    dtz_here = tablebase.dtz(board)
                except Exception:
                    dtz_here = None

                if dtz_here is not None and 0 < abs(dtz_here) <= near_dtz_max:
                    legal = list(board.legal_moves)
                    if len(legal) >= 2:
                        scored = []
                        for mv in legal:
                            board.push(mv)
                            
                            wdl_for_mover = -tablebase.wdl(board)
                            board.pop()
                            scored.append((wdl_for_mover, mv))
                        scored.sort(key=lambda t: -t[0])
                        best_wdl, best_mv = scored[0]
                        worst_wdl, worst_mv = scored[-1]
                        if best_wdl != worst_wdl:
                            pairs.append(MinimalPair(
                                prefix_uci=list(moves[:ply]),
                                move_a_uci=best_mv.uci(),
                                move_b_uci=worst_mv.uci(),
                                wdl_a=best_wdl, wdl_b=worst_wdl,
                                dtz_before=abs(dtz_here),
                                is_zugzwang=_is_zugzwang(board),
                            ))
                            if len(pairs) >= n_pairs:
                                break

            if ply < len(moves):
                board.push(chess.Move.from_uci(moves[ply]))

        if progress_every and scanned % progress_every == 0:
            elapsed = time.time() - t0
            n_zz = sum(p.is_zugzwang for p in pairs)
            print(f"    scanned {scanned} games, found {len(pairs)}/{n_pairs} pairs "
                  f"({n_zz} zugzwang, {len(pairs) - n_zz} generic) ({elapsed:.1f}s elapsed)")

    n_zz = sum(p.is_zugzwang for p in pairs)
    print(f"  found {len(pairs)}/{n_pairs} pairs after scanning {scanned} games "
          f"({n_zz} zugzwang, {len(pairs) - n_zz} generic) "
          f"({time.time() - t0:.1f}s)")
    if len(pairs) < n_pairs:
        print(f"  WARNING: ran out of games before reaching the target. Raise "
              f"--max_games_scanned or --near_dtz_max, or lower --n_pairs.")
    return pairs


def pair_to_tokens(pair: MinimalPair):
    """Tokenizes a MinimalPair into (clean_tokens, corrupt_tokens, attn_mask,
    pos_idx) ready for patch.py. 
    pos_idx is the index of the final move token."""
    import torch
    from tokenizer import VOCAB, BOS

    prefix_ids = [VOCAB.stoi[BOS]] + [VOCAB.encode(m) for m in pair.prefix_uci]
    clean_ids = prefix_ids + [VOCAB.encode(pair.move_a_uci)]
    corrupt_ids = prefix_ids + [VOCAB.encode(pair.move_b_uci)]

    clean_tokens = torch.tensor(clean_ids, dtype=torch.long).unsqueeze(0)
    corrupt_tokens = torch.tensor(corrupt_ids, dtype=torch.long).unsqueeze(0)
    attn = torch.ones_like(clean_tokens, dtype=torch.bool)
    pos_idx = clean_tokens.shape[1] - 1
    return clean_tokens, corrupt_tokens, attn, pos_idx
