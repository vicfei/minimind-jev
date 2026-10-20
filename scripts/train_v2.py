#!/usr/bin/env python3
"""Train the v0.2 head: frozen bge-small-zh encoder + a 4-way head on teacher soft labels.

Distillation data comes from scripts/distill_generate.py (deepseek-v4-flash grading pairs
into the pattern library's 4-level rubric). The head outputs logits over the four levels;
training minimizes soft cross-entropy against the teacher distribution. Temperature is
fitted on the validation split. Ranking score at eval time is the expected level weight
(0, 1/3, 2/3, 1) — see scripts/eval_export_v2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

LEVELS = ["irrelevant", "related", "partial", "answers"]
LEVEL_WEIGHTS = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])


class HeadV2(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, levels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, levels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def encode(model, texts: list[str], batch: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch):
        emb = model.encode(texts[i : i + batch], normalize_embeddings=True)
        out.append(torch.tensor(emb, dtype=torch.float32))
    return torch.cat(out)


QUESTION = "仅凭该段落能回答该查询吗？"


def load_distilled(distilled_path: Path, split: list[dict]) -> tuple[list[str], torch.Tensor, torch.Tensor, int]:
    """Join a train/val split with teacher distributions -> (texts, soft targets, hard labels, n_missing)."""
    by_key: dict[tuple, dict] = {}
    for line in distilled_path.read_text().splitlines():
        r = json.loads(line)
        by_key[(r["qid"], r["role"], r["idx"])] = r
    texts, targets, hard, missing = [], [], [], 0
    for d in split:
        pairs = [("positive", -1, d["positive"], 1.0)]
        pairs += [(f"negative", i, neg, 0.0) for i, neg in enumerate(d["negatives"])]
        for role, idx, passage, label in pairs:
            r = by_key.get((d["qid"], role, idx))
            if r is None:
                missing += 1
                continue
            texts.append(f"{d['query']}\n{QUESTION}\n{passage}")
            targets.append([r["dist"][k] for k in LEVELS])
            hard.append(label)
    return texts, torch.tensor(targets, dtype=torch.float32), torch.tensor(hard), missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distilled", default="data/distilled.jsonl")
    ap.add_argument("--allow-missing", action="store_true",
                    help="train on the pairs that have teacher labels even if some are missing")
    ap.add_argument("--out", default="model/head_v2.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent.parent
    train = [json.loads(l) for l in (root / "data/train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (root / "data/val.jsonl").read_text().splitlines()]
    distilled_path = root / args.distilled
    if not distilled_path.exists():
        raise SystemExit(f"error: {distilled_path} not found — run scripts/distill_generate.py first (SG side)")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    dim = enc.get_sentence_embedding_dimension()

    tr_texts, ytr, _, tr_miss = load_distilled(distilled_path, train)
    va_texts, yva, hva, va_miss = load_distilled(distilled_path, val)
    for name, miss in (("train", tr_miss), ("val", va_miss)):
        if miss:
            msg = f"{name}: {miss} pairs lack teacher labels"
            if args.allow_missing:
                print(f"warning: {msg} (skipping them)")
            else:
                raise SystemExit(f"error: {msg} — regenerate distilled data or pass --allow-missing")
    print(f"encoder: bge-small-zh-v1.5 (frozen, dim={dim}); "
          f"train {len(tr_texts)} pairs / val {len(va_texts)} pairs (4-level soft targets)")
    print("encoding train …"); Xtr = encode(enc, tr_texts)
    print("encoding val …"); Xva = encode(enc, va_texts); del enc

    head = HeadV2(dim)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(Xtr)
    bs = 512
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logp = torch.log_softmax(head(Xtr[idx]), dim=-1)
            loss = -(ytr[idx] * logp).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); seen += len(idx)
        head.eval()
        with torch.no_grad():
            logp_va = torch.log_softmax(head(Xva), dim=-1)
            va_loss = -(yva * logp_va).sum(-1).mean().item()
            exp_va = (torch.softmax(head(Xva), dim=-1) * LEVEL_WEIGHTS).sum(-1)
            acc = (((exp_va > 0.5).float() == hva).float().mean().item())
        print(f"epoch {epoch+1}: train_loss={tot/seen:.4f} val_loss={va_loss:.4f} val_acc@0.5={acc:.4f}")

    # temperature scaling on val soft-NLL
    with torch.no_grad():
        logits = head(Xva)
    T = nn.Parameter(torch.ones(1) * 1.0)
    topt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        topt.zero_grad()
        loss = -(yva * torch.log_softmax(logits / T, dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    topt.step(closure)
    temp = float(T.item())
    print(f"temperature (fitted on val): {temp:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "dim": dim, "levels": LEVELS,
                "level_weights": LEVEL_WEIGHTS.tolist(), "temp": temp}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
