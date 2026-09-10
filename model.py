"""
Minimal GPT-style move predictor. Kept deliberately hand-rolled (rather than nn.TransformerEncoder) for one
reason: M5 (activation patching) needs to read and overwrite the residual
stream at a specific (layer, position) before the next block consumes it.


`forward(..., return_hidden=True)` returns the residual stream after every
block: these are exactly the vectors M2/M3 probes are trained on and
exactly the vectors M5 patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import VOCAB


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB.size
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    max_len: int = 64
    dropout: float = 0.1
    aux_heads: bool = False  # if True, adds wdl/dtz-bucket auxiliary prediction heads
                              
    n_wdl_classes: int = 5
    n_dtz_bucket_classes: int = 6


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_head, T, head_dim]

        # Build one explicit boolean mask combining causality + key padding
        # (True = allowed to attend) rather than relying on is_causal=True
        
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # [T,T]
        combined = causal[None, None, :, :]  # [1,1,T,T]
        if attn_mask is not None:
            key_padding = attn_mask[:, None, None, :]  # [B,1,1,T], True = real token
            combined = combined & key_padding  # broadcasts to [B,1,T,T]

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MoveGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.aux_heads:
            self.wdl_head = nn.Linear(cfg.d_model, cfg.n_wdl_classes)
            self.dtz_head = nn.Linear(cfg.d_model, cfg.n_dtz_bucket_classes)

    def aux_logits(self, hidden_last: torch.Tensor):
        
        if not self.cfg.aux_heads:
            raise RuntimeError(
                "This model was built with aux_heads=False -- no auxiliary heads exist. "
                "Build it with GPTConfig(aux_heads=True) to use aux_logits()."
            )
        return self.wdl_head(hidden_last), self.dtz_head(hidden_last)

    def forward(self, idx, attn_mask=None, return_hidden=False, patch=None):
        """
        idx        : LongTensor [B, T] token ids
        attn_mask  : BoolTensor [B, T], True = real token, False = pad
        return_hidden : if True, also return the residual stream after
                         every block, shape list of n_layer tensors [B,T,d_model]
        patch      : optional dict {layer_idx: (batch_idx, pos_idx, vector)}
                     -- after block `layer_idx`, overwrite
                     hidden[batch_idx, pos_idx, :] = vector.
                     
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states = []
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, attn_mask=attn_mask)
            if patch is not None and layer_idx in patch:
                batch_idx, pos_idx, vector = patch[layer_idx]
                x = x.clone()
                x[batch_idx, pos_idx, :] = vector
            if return_hidden:
                hidden_states.append(x)

        logits = self.head(self.ln_f(x))
        if return_hidden:
            return logits, hidden_states
        return logits

    @torch.no_grad()
    def hidden_at(self, idx, layer_idx: int, attn_mask=None):
        
        _, hs = self.forward(idx, attn_mask=attn_mask, return_hidden=True)
        return hs[layer_idx]
