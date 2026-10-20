#!/usr/bin/env python3
"""Evaluate minimind-jev v0.2 (4-level distilled head) on the benchmark sample.

Ranking score = expected level weight under the head's softmax (temperature-scaled);
exports logs with the same envelope as the benchmark's score-zh variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_v2 import LEVEL_WEIGHTS, HeadV2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/home/ubuntu/jev-rag-zh-benchmark/data/s1.jsonl")
    ap.add_argument("--checkpoint", default="model/head_v2.pt")
    ap.add_argument("--out", default="logs/s1.minimind-jev-v2.jsonl")
    args = ap.parse_args()

    sample = [json.loads(l) for l in Path(args.sample).read_text().splitlines()]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    head = HeadV2(ckpt["dim"])
    head.load_state_dict(ckpt["state"])
    head.eval()
    temp = ckpt.get("temp", 1.0)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    texts, refs = [], []
    for s in sample:
        for p in s["pool"]:
            texts.append(f"{s['query']}\n仅凭该段落能回答该查询吗？\n{p['text']}")
            refs.append((s["qid"], p["pid"]))

    embs = []
    for i in range(0, len(texts), 64):
        embs.append(torch.tensor(
            enc.encode(texts[i : i + 64], normalize_embeddings=True), dtype=torch.float32))
    X = torch.cat(embs)
    with torch.no_grad():
        probs = torch.softmax(head(X) / temp, dim=-1)
        scores = (probs * torch.tensor(ckpt.get("level_weights", LEVEL_WEIGHTS.tolist()))).sum(-1).tolist()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.unlink(missing_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for (qid, pid), v in zip(refs, scores):
            rec = {
                "ts": 0, "variant": "minimind-jev-v2", "qid": qid, "pid": pid,
                "status": 200, "latency_s": 0.0,
                "model": "minimind-jev-v0.2 (bge-small-zh + 4-level distilled head)",
                "answer": {"answers": {"type": "score", "score": round(v, 6)}},
                "usage": {"input_tokens": 0, "cost": 0.0},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(refs)} rows)")


if __name__ == "__main__":
    main()
