"""
Move tokenizer for bare king endgames (KPK, KRK).


"""

from __future__ import annotations

from dataclasses import dataclass


FILES = "abcdefgh"
RANKS = "12345678"
PROMO_PIECES = ("n", "b", "r", "q")  # underpromotions matter in KPK zugzwang lines

# Special tokens
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = (PAD, BOS, EOS)


def sq_name(rank: int, file: int) -> str:
    return f"{FILES[file]}{RANKS[rank]}"


def in_bounds(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


def _king_moves():
    for r in range(8):
        for f in range(8):
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    if dr == 0 and df == 0:
                        continue
                    r2, f2 = r + dr, f + df
                    if in_bounds(r2, f2):
                        yield sq_name(r, f) + sq_name(r2, f2)


def _rook_moves():
    for r in range(8):
        for f in range(8):
            for f2 in range(8):
                if f2 != f:
                    yield sq_name(r, f) + sq_name(r, f2)
            for r2 in range(8):
                if r2 != r:
                    yield sq_name(r, f) + sq_name(r2, f)


def _bishop_moves():
    # Needed even though no bishop is ever on the board at game start
    for r in range(8):
        for f in range(8):
            for dr in (-1, 1):
                for df in (-1, 1):
                    r2, f2 = r + dr, f + df
                    while in_bounds(r2, f2):
                        yield sq_name(r, f) + sq_name(r2, f2)
                        r2 += dr
                        f2 += df


def _knight_moves():
    
    deltas = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
    for r in range(8):
        for f in range(8):
            for dr, df in deltas:
                r2, f2 = r + dr, f + df
                if in_bounds(r2, f2):
                    yield sq_name(r, f) + sq_name(r2, f2)


def _pawn_moves():
    # Pawns never occupy rank 1 or rank 8 as a source square.
    for r in range(1, 7):
        for f in range(8):
            frm = sq_name(r, f)
            for direction in (+1, -1):  # white and black pushes
                r1 = r + direction
                if not in_bounds(r1, f):
                    continue
                to1 = sq_name(r1, f)
                if r1 in (0, 7):
                    for p in PROMO_PIECES:
                        yield frm + to1 + p
                else:
                    yield frm + to1
                    # double push only from the pawn's own starting rank
                    if r == 1 and direction == +1:
                        r2 = r + 2
                        if not in_bounds(r2, f):
                            continue
                        to2 = sq_name(r2, f)
                        yield frm + to2
                    if r == 6 and direction == -1:
                        r2 = r - 2
                        if not in_bounds(r2, f):
                            continue
                        to2 = sq_name(r2, f)
                        yield frm + to2


def generate_move_geometry() -> list[str]:
    """All UCI-style move strings reachable by K, R or P on an empty board.

    Deterministic order (king, then rook, then pawn, each row-major) so the
    resulting vocabulary is reproducible across runs/machines without
    needing to persist a saved mapping: convenient for M6 merging, where
    both specialists must be built from the exact same tokenizer version.
    """
    seen = []
    seen_set = set()
    for gen in (_king_moves, _rook_moves, _bishop_moves, _knight_moves, _pawn_moves):
        for uci in gen():
            if uci not in seen_set:
                seen_set.add(uci)
                seen.append(uci)
    return seen


@dataclass(frozen=True)
class Vocab:
    stoi: dict
    itos: list

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, uci: str) -> int:
        try:
            return self.stoi[uci]
        except KeyError as e:
            raise KeyError(
                f"Move '{uci}' is not in the tokenizer vocabulary. "
                "This should never happen for legal K/R/P moves generated "
                "by python-chess in a bare-king endgame; if it does, the "
                "move likely involves a piece type outside {K,R,P} (check "
                "the FEN) or a capture (which cannot occur in KPK/KRK)."
            ) from e

    def decode(self, idx: int) -> str:
        return self.itos[idx]


def build_vocab() -> Vocab:
    tokens = list(SPECIAL_TOKENS) + generate_move_geometry()
    stoi = {t: i for i, t in enumerate(tokens)}
    return Vocab(stoi=stoi, itos=tokens)



VOCAB = build_vocab()


def move_to_uci(move) -> str:
    """Adapter for a python-chess Move object -> our UCI string.
    """
    return move.uci()


def uci_to_move(uci: str, board):
    """Adapter for our UCI string -> a python-chess Move object bound to
    `board` (needed because python-chess's Move.from_uci is board-agnostic
    but push() wants a Move consistent with the position)."""
    import chess  

    return chess.Move.from_uci(uci)
