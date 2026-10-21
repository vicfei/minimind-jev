"""White-box transformer encoder for typed decisions.

Encoder-only (this is not a language model — no causal mask, no decoding):
embedding -> N pre-LN blocks (hand-rolled MHA on F.scaled_dot_product_attention,
GELU FFN) -> masked mean pool -> a single logit = P(passage answers query).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import V3Config


class Block(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, d)
        self.n_heads = h
        self.head_dim = d // h
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        # keep_mask: (B, L) bool, True where the token is real (not padding)
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = keep_mask[:, None, None, :]  # broadcast over heads and query positions
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(B, L, d)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, keep_mask)
        return self.ln_f(x)


class DecisionModel(nn.Module):
    """(query, question, passage) -> one logit; sigmoid gives P(passage answers query)."""

    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, ids: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(ids, keep_mask)
        pooled = (h * keep_mask.unsqueeze(-1)).sum(1) / keep_mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class MLMModel(nn.Module):
    """Pretraining wrapper: the same encoder with a vocab head (dropped after MLM)."""

    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.encoder(ids, keep_mask))


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
