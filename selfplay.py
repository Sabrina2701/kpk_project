"""
Self-play generation for KPK and, by passing piece_set=('K','R'), KRK.

  - `p_critical` of positions are drawn from a constrained generator that
    keeps the defending king within a few squares of the pawn/attacking
    king (where opposition and key-square play actually matter),
  - the rest are drawn uniformly at random over all legal placements,

and self-play moves are chosen by an epsilon-mixture of random legal moves
and tablebase-optimal moves (`Tablebase.best_move_by_dtz`), so a single game
still passes through both obvious and critical phases rather than only
ever seeing one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class SelfPlayConfig:
    p_critical_start: float = 0.75  # fraction of starting positions sampled near-critical
    epsilon_random_move: float = 0.15  # P(pick a uniformly random legal move instead of tb-optimal)
    max_plies: int = 60
    critical_king_radius: int = 1   # chebyshev distance allowed between defending king and pawn
   


def _kings_adjacent(sq_a: int, sq_b: int) -> bool:
    import chess

    return chess.square_distance(sq_a, sq_b) <= 1


def _chebyshev(sq_a: int, sq_b: int) -> int:
    import chess

    return max(
        abs(chess.square_file(sq_a) - chess.square_file(sq_b)),
        abs(chess.square_rank(sq_a) - chess.square_rank(sq_b)),
    )


def random_kpk_position(rng: random.Random, critical: bool, cfg: SelfPlayConfig):
   
    import chess

    for _ in range(10_000):  # rejection sampling; KPK legal space is tiny so this converges fast
        pawn_sq = rng.randrange(8, 56)  # ranks 2-7 only (0-7 and 56-63 excluded)
        wk_sq = rng.randrange(64)
        if wk_sq == pawn_sq:
            continue

        if critical:
            # defending king sampled within `critical_king_radius` of the pawn
            pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
            df = pf + rng.randint(-cfg.critical_king_radius, cfg.critical_king_radius)
            dr = pr + rng.randint(-cfg.critical_king_radius, cfg.critical_king_radius)
            if not (0 <= df < 8 and 0 <= dr < 8):
                continue
            bk_sq = chess.square(df, dr)
        else:
            bk_sq = rng.randrange(64)

        if bk_sq in (wk_sq, pawn_sq):
            continue
        if _kings_adjacent(wk_sq, bk_sq):
            continue

        turn = rng.choice([chess.WHITE, chess.BLACK])

        board = chess.Board(None)  # empty board
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = turn

        if not board.is_valid():
            continue
      
        return board

    raise RuntimeError("Failed to sample a legal KPK position after 10k tries")


def random_krk_position(rng: random.Random):

    import chess

    for _ in range(10_000):
        wk_sq = rng.randrange(64)
        rook_sq = rng.randrange(64)
        bk_sq = rng.randrange(64)
        if len({wk_sq, rook_sq, bk_sq}) < 3:
            continue
        if _kings_adjacent(wk_sq, bk_sq):
            continue

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.turn = rng.choice([True, False])

        if board.is_valid():
            return board
    raise RuntimeError("Failed to sample a legal KRK position after 10k tries")


@dataclass
class GameRecord:
    start_fen: str
    moves_uci: list = field(default_factory=list)      # ply-by-ply UCI move tokens
    wdl_after: list = field(default_factory=list)       # tablebase WDL after each move, side-to-move POV
    dtz_after: list = field(default_factory=list)       # tablebase DTZ after each move
    outcome: str = "unknown"                            # "win" / "draw" / "loss" / "unfinished" (from start pos, side-to-move POV)


def play_one_game(board, tablebase, rng: random.Random, cfg: SelfPlayConfig) -> GameRecord:
    from tokenizer import move_to_uci
    from tablebase import position_key

    rec = GameRecord(start_fen=board.fen())
    rec.outcome = tablebase.outcome_label(board) if list(board.legal_moves) else "terminal"

    
    visited = {position_key(board)}

    for _ply in range(cfg.max_plies):
        legal = list(board.legal_moves)
        if not legal:
            break  # checkmate or stalemate

        if rng.random() < cfg.epsilon_random_move:
            move = rng.choice(legal)
        else:
            move = tablebase.best_move_by_dtz(board, avoid_positions=visited)
            if move is None:
                move = rng.choice(legal)

        board.push(move)
        visited.add(position_key(board))
        rec.moves_uci.append(move_to_uci(move))

        if list(board.legal_moves):
            rec.wdl_after.append(tablebase.wdl(board))
            rec.dtz_after.append(tablebase.dtz(board))
        else:
          
            rec.wdl_after.append(-2 if board.is_checkmate() else 0)
            rec.dtz_after.append(0)
            break

    return rec


def generate_dataset(
    n_games: int,
    tablebase,
    seed: int = 0,
    cfg: SelfPlayConfig | None = None,
    domain: str = "kpk",
) -> list[GameRecord]:
    cfg = cfg or SelfPlayConfig()
    rng = random.Random(seed)
    records = []
    for i in range(n_games):
        critical = rng.random() < cfg.p_critical_start
        if domain == "kpk":
            board = random_kpk_position(rng, critical=critical, cfg=cfg)
        elif domain == "krk":
            board = random_krk_position(rng)
        else:
            raise ValueError(domain)
        records.append(play_one_game(board, tablebase, rng, cfg))
    return records
