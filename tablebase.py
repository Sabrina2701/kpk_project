

from __future__ import annotations

from functools import lru_cache


def position_key(board) -> str:
    # board+turn+castling+ep only; strip halfmove/fullmove clocks.
    return board.fen().split(" ")[0] + " " + board.fen().split(" ")[1]


_cache_key = position_key  # internal alias, kept for readability below


class Tablebase:
    def __init__(self, directory: str, cache_size: int = 200_000):
        import chess.syzygy  

        self._tb = chess.syzygy.open_tablebase(directory)
       
        self._probe_wdl_cached = lru_cache(maxsize=cache_size)(self._probe_wdl_raw)
        self._probe_dtz_cached = lru_cache(maxsize=cache_size)(self._probe_dtz_raw)

    def _probe_wdl_raw(self, key: str):
        import chess

        board = chess.Board(key)
        return self._tb.probe_wdl(board)

    def _probe_dtz_raw(self, key: str):
        import chess

        board = chess.Board(key)
        return self._tb.probe_dtz(board)

    def wdl(self, board) -> int:
        """Win/Draw/Loss from the side-to-move's perspective.
        +2 win, +1 cursed win (irrelevant <=5 men but kept for API parity),
        0 draw, -1 blessed loss, -2 loss."""
        return self._probe_wdl_cached(_cache_key(board))

    def dtz(self, board) -> int:
        """Distance to zeroing (mate/pawn-move/capture) in plies, signed
        from the side-to-move's perspective. This is the ground-truth
        target for M3 (distance-to-outcome probing)."""
        return self._probe_dtz_cached(_cache_key(board))

    def outcome_label(self, board) -> str:
        w = self.wdl(board)
        if w > 0:
            return "win"
        if w < 0:
            return "loss"
        return "draw"

    def best_move_by_dtz(self, board, avoid_positions=None):
        """Return a tablebase-optimal move.

        avoid_positions : optional set of position_key(board) strings
                           already visited in the current game.
        Returns None if the position is already terminal (no legal moves).
        """
        legal = list(board.legal_moves)
        if not legal:
            return None

        def score(move):
            board.push(move)
            try:
                opp_wdl = self.wdl(board)   
                opp_dtz = self.dtz(board)
            finally:
                board.pop()
            our_result = -opp_wdl  
            if our_result > 0:
                urgency = -abs(opp_dtz)   # winning: prefer small |dtz| (win fast)
            elif our_result < 0:
                urgency = abs(opp_dtz)    # losing: prefer large |dtz| 
            else:
                urgency = -abs(opp_dtz)   # drawn: doesn't affect correctness
            return (our_result, urgency)

        scored = [(score(m), m) for m in legal]
        best_score = max(s for s, _ in scored)
        best_moves = [m for s, m in scored if s == best_score]

        if avoid_positions and len(best_moves) > 1:
            for m in best_moves:
                board.push(m)
                key = position_key(board)
                board.pop()
                if key not in avoid_positions:
                    return m
       

        return best_moves[0]
