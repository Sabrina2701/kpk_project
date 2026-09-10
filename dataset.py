"""
This PyTorch file defines a dataset class (KPKDataset) and a batching function (collate_fn) to load and process 
chess endgames (KPK or KRK) stored in JSONL shard files.
KPKDataset reads the JSONL shards written by data_gen.py and produces for every game:

  tokens        : LongTensor [T]      BOS + move tokens + EOS (padded outside)
  next_tokens   : LongTensor [T]      tokens shifted by one (M1 LM target, -100 at pad)
  king_sq       : LongTensor [T, 2]   (own_king_sq, other_king_sq) after each ply -- M2 target
  special_sq    : LongTensor [T]      pawn (or rook) square after each ply       -- M2 target
  dtz           : LongTensor [T]      signed DTZ after each ply                  -- M3 target
  wdl           : LongTensor [T]      WDL {-2,-1,0,1,2} after each ply           -- M3 target (coarse)

Board state at each ply is reconstructed by replaying the stored move list
from start_fen with python-chess.
"""

from __future__ import annotations  #for notes in the latest python versions

import json
import os

import torch
from torch.utils.data import Dataset   #the standard class for generating a dataset in pytorch

from tokenizer import VOCAB, PAD, BOS, EOS

#Extracts board square indices (0–63) for active pieces at any given position
def _piece_squares(board, domain: str): 
    import chess   #we need the python-chess library to be upload dinamically

    stm = board.turn  #identify whose turn it is
    own_king = board.king(stm) #retrieves the square index for the king of the active player 
    other_king = board.king(not stm) #and the opponent
    if domain == "kpk": 
        piece_type = chess.PAWN
    elif domain == "krk": 
        piece_type = chess.ROOK
    else:
        raise ValueError(domain)
    squares = board.pieces(piece_type, chess.WHITE) | board.pieces(piece_type, chess.BLACK)
    special_sq = next(iter(squares)) if squares else -1 #returns the square index of the tracked piece; -1 if it was captured
    return own_king, other_king, special_sq


#Handles loading game data from disk and extracting move sequences with targets
class KPKDataset(Dataset):
    #Finds all .jsonl files in shard_dir, raises FileNotFoundError if none are found and loads game entries into self.games
    def __init__(self, shard_dir: str, domain: str = "kpk", max_len: int = 64):
        self.domain = domain
        self.max_len = max_len
        self.games = []
        shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".jsonl"))
        if not shard_files:
            raise FileNotFoundError(
                f"No .jsonl shards found in {shard_dir}. Run data_gen.py first."
            )
        for fname in shard_files:
            with open(os.path.join(shard_dir, fname)) as f:
                for line in f:
                    self.games.append(json.loads(line))

    #Total number of loaded games
    def __len__(self):
        return len(self.games)

    #Processes a single game at the specified index
    def __getitem__(self, idx: int):
        import chess

        g = self.games[idx]
        board = chess.Board(g["start_fen"])
        moves = g["moves"][: self.max_len - 2]  # room for BOS/EOS

       
        #Starts the sequence with BOS token and records starting piece positions, DTZ (Distance to Zero) and WDL (Win/Draw/Loss) metrics
        tokens = [VOCAB.stoi[BOS]]
        own_king_seq, other_king_seq, special_seq = [], [], []
        dtz_seq, wdl_seq = [], []

        #ply 0: the start position itself before any move
        ok, otk, sp = _piece_squares(board, self.domain)
        own_king_seq.append(ok)
        other_king_seq.append(otk)
        special_seq.append(sp)
        dtz_seq.append(g["dtz"][0] if g["dtz"] else 0)
        wdl_seq.append(g["wdl"][0] if g["wdl"] else 0)

        #Encodes each move, updates the board with board.push_uci()
        #and logs piece squares, DTZ and WDL values after every move
        for i, uci in enumerate(moves):
            tokens.append(VOCAB.encode(uci))
            board.push_uci(uci)
            ok, otk, sp = _piece_squares(board, self.domain)
            own_king_seq.append(ok)
            other_king_seq.append(otk)
            special_seq.append(sp)
            dtz_seq.append(g["dtz"][i] if i < len(g["dtz"]) else 0)
            wdl_seq.append(g["wdl"][i] if i < len(g["wdl"]) else 0)

        #Appends the EOS token and duplicates the final board metrics to keep sequence lengths uniform
        tokens.append(VOCAB.stoi[EOS])
        own_king_seq.append(own_king_seq[-1])
        other_king_seq.append(other_king_seq[-1])
        special_seq.append(special_seq[-1])
        dtz_seq.append(dtz_seq[-1])
        wdl_seq.append(wdl_seq[-1])

        #Returns a dictionary of PyTorch LongTensor objects containing tokens and target metadata
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "own_king": torch.tensor(own_king_seq, dtype=torch.long),
            "other_king": torch.tensor(other_king_seq, dtype=torch.long),
            "special_sq": torch.tensor(special_seq, dtype=torch.long),
            "dtz": torch.tensor(dtz_seq, dtype=torch.long),
            "wdl": torch.tensor(wdl_seq, dtype=torch.long),
        }

#Merges individual game samples into padded batches suitable for nn training
def collate_fn(batch, pad_id: int = None):
    pad_id = VOCAB.stoi[PAD] if pad_id is None else pad_id
    max_len = max(x["tokens"].shape[0] for x in batch) #picks the longest sequence in the current batch

    #Helper function that applies right-side padding to shorter tensor sequences
    def pad(seq, value):
        return torch.nn.functional.pad(seq, (0, max_len - seq.shape[0]), value=value)

    tokens = torch.stack([pad(x["tokens"], pad_id) for x in batch])

    #Creates language modeling targets by shifting tokens one position to the right and filling padding positions with -100
    next_tokens = tokens[:, 1:].clone()
    next_tokens = torch.nn.functional.pad(next_tokens, (0, 1), value=-100)
    next_tokens[tokens == pad_id] = -100  #ignore_index for CE loss

    #stacked tensors for token sequences, loss masks, piece positions and endgame evaluation metrics
    out = {
        "tokens": tokens,
        "next_tokens": next_tokens,
        "attn_mask": (tokens != pad_id),  #boolean attention mask marking real tokens as True and padding tokens as False
        "own_king": torch.stack([pad(x["own_king"], -1) for x in batch]),
        "other_king": torch.stack([pad(x["other_king"], -1) for x in batch]),
        "special_sq": torch.stack([pad(x["special_sq"], -1) for x in batch]),
        "dtz": torch.stack([pad(x["dtz"], 0) for x in batch]),
        "wdl": torch.stack([pad(x["wdl"], 0) for x in batch]),
    }
    return out
