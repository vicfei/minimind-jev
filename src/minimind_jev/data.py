"""Data plumbing for v0.3: unlabeled corpus lines (MLM) and labeled pairs (BCE).

Pair text format is identical to v0.1's (query \n question \n passage) so the only
difference between v0.1 and v0.3 runs is the encoder: frozen pretrained bge-small-zh
vs our own from-scratch transformer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

QUESTION = "仅凭该段落能回答该查询吗？"


def load_splits(root: Path, splits: tuple[str, ...] = ("train", "val")) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        for line in (root / "data" / f"{split}.jsonl").read_text().splitlines():
            rows.append(json.loads(line))
    return rows


def corpus_lines(rows: list[dict]) -> list[str]:
    """Unique queries and passages — the unlabeled MLM corpus (~20MB from our splits)."""
    seen: set[str] = set()
    lines: list[str] = []
    for r in rows:
        for text in [r["query"], r["positive"], *r["negatives"]]:
            if text not in seen:
                seen.add(text)
                lines.append(text)
    return lines


def pair_examples(rows: list[dict]) -> tuple[list[str], list[float]]:
    texts, labels = [], []
    for r in rows:
        texts.append(f"{r['query']}\n{QUESTION}\n{r['positive']}"); labels.append(1.0)
        for neg in r["negatives"]:
            texts.append(f"{r['query']}\n{QUESTION}\n{neg}"); labels.append(0.0)
    return texts, labels


def encode_batch(tokenizer, texts: list[str], ctx: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and pad every sequence to exactly ctx width (uniform shapes stack cleanly)."""
    enc = tokenizer.encode_batch([t[: ctx * 4] for t in texts])  # cheap char pre-truncation
    seqs = [e.ids[:ctx] for e in enc]
    padded = torch.full((len(seqs), ctx), pad_id, dtype=torch.long)
    keep = torch.zeros((len(seqs), ctx), dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        keep[i, : len(s)] = True
    return padded, keep


def mask_mlm(ids: torch.Tensor, mask_id: int, p: float = 0.15,
             generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """15% token masking; returns (masked_ids, targets) where unmasked targets are -100."""
    probs = torch.rand(ids.shape, generator=generator)
    selected = probs < p
    targets = ids.clone()
    targets[~selected] = -100
    masked = ids.clone()
    masked[selected] = mask_id
    return masked, targets
