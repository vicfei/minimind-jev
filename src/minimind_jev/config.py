"""v0.3 model configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class V3Config:
    # 6L / d384 / FFN 4x lands at ~13.9M params (emb 3.07M + pos 0.20M + 6 x 1.77M + head)
    # — the "6-layer / d384 / ~14M" spec from the roadmap.
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6          # head dim 64
    d_ff: int = 1536          # 4x expansion
    ctx: int = 512            # matches bge-small-zh's 512 — keeps v0.1 vs v0.3 comparable
    pad_id: int = 0
    mask_token: str = "[MASK]"

    def dict(self) -> dict:
        return asdict(self)
