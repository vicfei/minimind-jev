#!/usr/bin/env python3
"""Evaluate minimind-jev on the benchmark sample and export logs compatible with
jev-rag-zh-benchmark's analyze.py — so the three-way table drops out unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train import DecisionHead, QUESTION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/home/ubuntu/jev-rag-zh-benchmark/data/s1.jsonl")
    ap.add_argument("--checkpoint", default="model/head.pt")
    ap.add_argument("--out", default="logs/s1.minimind-jev.jsonl")
    args = ap.parse_args()

    sample = [json.loads(l) for l in Path(args.sample).read_text().splitlines()]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    head = DecisionHead(ckpt["dim"])
    head.load_state_dict(ckpt["state"])
    head.eval()
    temp = ckpt.get("temp", 1.0)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    texts, refs = [], []
    for s in sample:
        for p in s["pool"]:
            texts.append(f"{s['query']}\n{QUESTION}\n{p['text']}")
            refs.append((s["qid"], p["pid"]))

    embs = []
    for i in range(0, len(texts), 64):
        embs.append(torch.tensor(
            enc.encode(texts[i : i + 64], normalize_embeddings=True), dtype=torch.float32))
    X = torch.cat(embs)
    with torch.no_grad():
        probs = torch.sigmoid(head(X) / temp).tolist()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for (qid, pid), p in zip(refs, probs):
            rec = {
                "ts": 0, "variant": "minimind-jev", "qid": qid, "pid": pid,
                "status": 200, "latency_s": 0.0,
                "model": "minimind-jev-v0.1 (bge-small-zh + head)",
                "answer": {"answers": {"type": "noul", "noul": round(p, 6)}},
                "usage": {"input_tokens": 0, "cost": 0.0},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(refs)} rows)")


if __name__ == "__main__":
    main()
