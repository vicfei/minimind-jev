#!/usr/bin/env python3
"""Train the v0.1 decision head: frozen bge-small-zh encoder + MLP, CPU-friendly.

The model is a cross-encoder: encode "query [SEP] passage" once with the frozen
Chinese encoder, then a small from-scratch MLP predicts P(passage answers query).
Temperature is fitted on the validation split (calibration ethos from the library).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

QUESTION = "仅凭该段落能回答该查询吗？"


class DecisionHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def encode(model, texts: list[str], batch: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch):
        emb = model.encode(texts[i : i + batch], normalize_embeddings=True)
        out.append(torch.tensor(emb, dtype=torch.float32))
    return torch.cat(out)


def pairs_of(split: list[dict]) -> tuple[list[str], list[float]]:
    texts, labels = [], []
    for d in split:
        texts.append(f"{d['query']}\n{QUESTION}\n{d['positive']}"); labels.append(1.0)
        for neg in d["negatives"]:
            texts.append(f"{d['query']}\n{QUESTION}\n{neg}"); labels.append(0.0)
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model/head.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent.parent
    train = [json.loads(l) for l in (root / "data/train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (root / "data/val.jsonl").read_text().splitlines()]

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    dim = enc.get_sentence_embedding_dimension()
    print(f"encoder: bge-small-zh-v1.5 (frozen, dim={dim})")

    print("encoding train …")
    Xtr_texts, ytr = pairs_of(train)
    Xtr = encode(enc, Xtr_texts)
    ytr = torch.tensor(ytr)
    print("encoding val …")
    Xva_texts, yva = pairs_of(val)
    Xva = encode(enc, Xva_texts)
    yva = torch.tensor(yva)
    del enc

    head = DecisionHead(dim)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    bs = 512
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            logits = head(Xtr[idx])
            loss = lossf(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); seen += len(idx)
        head.eval()
        with torch.no_grad():
            va_logits = head(Xva)
            va_loss = lossf(va_logits, yva).item()
            acc = ((va_logits > 0).float() == yva).float().mean().item()
        print(f"epoch {epoch+1}: train_loss={tot/seen:.4f} val_loss={va_loss:.4f} val_acc={acc:.4f}")

    # temperature scaling on validation (calibration)
    with torch.no_grad():
        logits = head(Xva)
    T = nn.Parameter(torch.ones(1) * 1.0)
    topt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        topt.zero_grad()
        loss = lossf(logits / T, yva)
        loss.backward()
        return loss

    topt.step(closure)
    temp = float(T.item())
    print(f"temperature (fitted on val): {temp:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "dim": dim, "temp": temp}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
