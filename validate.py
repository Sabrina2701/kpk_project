

from __future__ import annotations

import argparse
import random
import statistics
import sys


def check_vocab_coverage(rng, n_positions=500):
    """Verifies that all legal moves in newly sampled starting KPK and KRK positions (pre-promotion) are present in the tokenizer vocabulary."""
    from selfplay import random_kpk_position, random_krk_position, SelfPlayConfig
    from tokenizer import VOCAB, move_to_uci

    cfg = SelfPlayConfig()
    missing = []
    for _ in range(n_positions):
        board = random_kpk_position(rng, critical=rng.random() < 0.5, cfg=cfg)
        for move in board.legal_moves:
            uci = move_to_uci(move)
            if uci not in VOCAB.stoi:
                missing.append(("kpk", uci, board.fen()))
    for _ in range(n_positions):
        board = random_krk_position(rng)
        for move in board.legal_moves:
            uci = move_to_uci(move)
            if uci not in VOCAB.stoi:
                missing.append(("krk", uci, board.fen()))

    if missing:
        print(f"FAIL vocab_coverage: {len(missing)} legal moves not in vocab.")
        for domain, uci, fen in missing[:10]:
            print(f"    [{domain}] {uci}  (fen: {fen})")
        return False
    print(f"OK   vocab_coverage: all legal moves across {2 * n_positions} sampled positions are tokenizable")
    return True


def check_played_moves_in_vocab(records):
    """Checks all UCI moves actually played across generated games—specifically 
    covering post-promotion moves (e.g., queen diagonal moves) to prevent runtime vocabulary key errors."""
    from tokenizer import VOCAB

    missing = []
    for rec in records:
        for uci in rec.moves_uci:
            if uci not in VOCAB.stoi:
                missing.append((uci, rec.start_fen))

    if missing:
        print(f"FAIL played_moves_in_vocab: {len(missing)} played moves not in vocab.")
        for uci, fen in missing[:10]:
            print(f"    {uci}  (game start: {fen})")
        return False
    n_moves = sum(len(r.moves_uci) for r in records)
    print(f"OK   played_moves_in_vocab: all {n_moves} moves played across {len(records)} games are tokenizable")
    return True


def check_game_legality(records):
    """Replays every stored move sequence on a fresh 'python-chess' board using .push() to confirm that no illegal moves were executed."""
    import chess

    bad_games = 0
    for rec in records:
        board = chess.Board(rec.start_fen)
        try:
            for uci in rec.moves_uci:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"{uci} not legal in {board.fen()}")
                board.push(move)
        except (ValueError, AssertionError) as e:
            bad_games += 1
            if bad_games <= 3:
                print(f"    illegal move in game starting {rec.start_fen}: {e}")

    if bad_games:
        print(f"FAIL game_legality: {bad_games}/{len(records)} games contained an illegal move")
        return False
    print(f"OK   game_legality: all {len(records)} games are legal end-to-end")
    return True


def check_dtz_consistency(records, tb, sample=150):
    """Re-queries the Syzygy tablebase during replay and compares live DTZ/WDL values against the cached generation labels to detect timing or state-sync bugs."""
    import chess

    sampled = records if len(records) <= sample else random.sample(records, sample)
    checked = mismatches = 0
    for rec in sampled:
        board = chess.Board(rec.start_fen)
        for i, uci in enumerate(rec.moves_uci):
            board.push(chess.Move.from_uci(uci))
            if not list(board.legal_moves):
                break
            checked += 1
            if tb.dtz(board) != rec.dtz_after[i]:
                mismatches += 1
            if tb.wdl(board) != rec.wdl_after[i]:
                mismatches += 1

    rate = mismatches / max(checked, 1)
    status = "OK  " if mismatches == 0 else "FAIL"
    print(f"{status} dtz_consistency: {mismatches}/{checked} label mismatches on replay ({rate:.4%})")
    return mismatches == 0


def check_terminal_outcomes(records):
    """Validates that terminal game states adhere to correct outcome conventions (checkmates map to +2 WDL for the winner, stalemates map to 0 WDL)."""
    import chess

    bad = 0
    for rec in records:
        board = chess.Board(rec.start_fen)
        for uci in rec.moves_uci:
            board.push(chess.Move.from_uci(uci))
        if list(board.legal_moves) or not rec.wdl_after:
            continue
        if board.is_checkmate() and rec.wdl_after[-1] != -2:
            bad += 1
        elif board.is_stalemate() and rec.wdl_after[-1] != 0:
            bad += 1

    print(f"{'OK  ' if bad == 0 else 'FAIL'} terminal_outcomes: {bad} mismatches between mate/stalemate and recorded WDL")
    return bad == 0


def check_no_stuck_cycles(records, min_repeats=6, max_period=4, max_stuck_rate=0.01):
    """Detects whether decisive games end in short, repeating move loops (periods of 1–4 plies), 
    ensuring the self-play policy does not get trapped in non-progressing cycles.
"""
    def has_short_cycle(moves):
        n = len(moves)
        for p in range(1, max_period + 1):
            if n < p * min_repeats:
                continue
            tail = moves[-p * min_repeats:]
            chunks = [tuple(tail[i:i + p]) for i in range(0, len(tail), p)]
            if all(c == chunks[0] for c in chunks):
                return p
        return None

    decisive = [r for r in records if r.outcome in ("win", "loss")]
    stuck = [r for r in decisive if has_short_cycle(r.moves_uci) is not None]
    rate = len(stuck) / max(len(decisive), 1)

    status = "OK  " if rate <= max_stuck_rate else "FAIL"
    print(f"{status} no_stuck_cycles: {len(stuck)}/{len(decisive)} decisive games ended in a "
          f"non-progressing move cycle ({rate:.2%}, threshold {max_stuck_rate:.0%})")
    if stuck[:1]:
        print(f"    example: start_fen={stuck[0].start_fen}  tail={stuck[0].moves_uci[-8:]}")
    return rate <= max_stuck_rate


def check_critical_sampler_effect(rng, tb, n=150):
    """Performs a soft informational check measuring whether enabling `p_critical_start` successfully increases the proportion of true zugzwang positions.
"""
    import chess
    from selfplay import random_kpk_position, SelfPlayConfig

    def get_zugzwang_rate(critical_flag):
        cfg = SelfPlayConfig()
        zugzwang_count = 0
        valid_samples = 0

        while valid_samples < n:
            board = random_kpk_position(rng, critical=critical_flag, cfg=cfg)

            w1 = tb.wdl(board)
            white_wdl_1 = w1 if board.turn == chess.WHITE else -w1

            board.turn = not board.turn
            if not board.is_valid():
                continue

            w2 = tb.wdl(board)
            white_wdl_2 = w2 if board.turn == chess.WHITE else -w2

            if (white_wdl_1 > 0) != (white_wdl_2 > 0):
                zugzwang_count += 1
            valid_samples += 1

        return zugzwang_count / n

    rate_easy = get_zugzwang_rate(False)
    rate_hard = get_zugzwang_rate(True)
    print("INFO critical_sampler_effect (soft check, does not fail the run):")
    print(f"    p_critical=0 -> true_zugzwang_rate={rate_easy:.1%}")
    print(f"    p_critical=1 -> true_zugzwang_rate={rate_hard:.1%}")
    if rate_hard <= rate_easy + 0.05:
        print("    WARNING: zugzwang rate barely changes with p_critical -- consider tightening "
              "critical_king_radius in SelfPlayConfig, the bias may be too weak to matter.")


def report_coverage(records):
    """Computes and logs high-level dataset metrics, including total game count, min/mean/max game lengths and outcome distributions."""
    lengths = [len(r.moves_uci) for r in records]
    outcomes: dict = {}
    for r in records:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    print("INFO dataset_summary:")
    print(f"    n_games={len(records)}  length: mean={statistics.mean(lengths):.1f} "
          f"min={min(lengths)} max={max(lengths)}")
    print(f"    outcome distribution: {outcomes}")


def main():
    """Validation pipeline by generating a test sample, running all checks, logging summary statistics and exiting with status code 0 (all hard checks passed) or 1 (any hard check failed).
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tablebase_dir", type=str, default="./syzygy")
    ap.add_argument("--domain", choices=["kpk", "krk"], default="kpk")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tablebase import Tablebase
    from selfplay import generate_dataset, SelfPlayConfig

    rng = random.Random(args.seed)
    tb = Tablebase(args.tablebase_dir)

    print(f"Generating {args.n_games} KPK games for validation...\n")
    records = generate_dataset(args.n_games, tb, seed=args.seed, cfg=SelfPlayConfig(), domain=args.domain)

    results = [
        check_vocab_coverage(rng),
        check_played_moves_in_vocab(records),
        check_game_legality(records),
        check_dtz_consistency(records, tb),
        check_terminal_outcomes(records),
        check_no_stuck_cycles(records),
    ]
    check_critical_sampler_effect(rng, tb)   # soft, informational only
    report_coverage(records)

    print()
    if all(results):
        print("ALL HARD CHECKS PASSED -- safe to launch the full data_gen.py run.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED -- fix the flagged module before generating the full dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
