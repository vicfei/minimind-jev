#!/usr/bin/env python3
"""Evaluate minimind-jev v0.3 (from-scratch transformer) on the benchmark sample,
exporting benchmark-compatible logs (variant minimind-jev-v3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minimind_jev.config import V3Config  # noqa: E402
from minimind_jev.data import QUESTION, encode_batch  # noqa: E402
from minimind_jev.model import DecisionModel  # noqa: E402
from minimind_jev.tokenizer import load_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/home/ubuntu/jev-rag-zh-benchmark/data/s1.jsonl")
    ap.add_argument("--checkpoint", default="model/v3.pt")
    ap.add_argument("--out", default="logs/s1.minimind-jev-v3.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    ckpt = torch.load(root / args.checkpoint, map_location="cpu")
    cfg = V3Config(**ckpt["cfg"])
    model = DecisionModel(cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()
    temp = ckpt.get("temp", 1.0)
    tokenizer = load_tokenizer(root / ckpt["tokenizer"])
    pad_id = ckpt.get("pad_id", 0)

    sample = [json.loads(l) for l in Path(args.sample).read_text().splitlines()]
    texts, refs = [], []
    for s in sample:
        for p in s["pool"]:
            texts.append(f"{s['query']}\n{QUESTION}\n{p['text']}")
            refs.append((s["qid"], p["pid"]))

    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            ids, keep = encode_batch(tokenizer, texts[i : i + args.batch], cfg.ctx, pad_id)
            logits = model(ids, keep)
            scores.extend(torch.sigmoid(logits / temp).tolist())

    out = root / args.out
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for (qid, pid), p in zip(refs, scores):
            rec = {
                "ts": 0, "variant": "minimind-jev-v3", "qid": qid, "pid": pid,
                "status": 200, "latency_s": 0.0,
                "model": "minimind-jev-v0.3 (from-scratch 6L/d384 transformer)",
                "answer": {"answers": {"type": "noul", "noul": round(p, 6)}},
                "usage": {"input_tokens": 0, "cost": 0.0},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(refs)} rows)")


if __name__ == "__main__":
    main()
