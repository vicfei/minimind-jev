#!/usr/bin/env python3
"""Build typed-question training data from T2Reranking dev (same source as jev-rag-zh-benchmark).

Output: data/train.jsonl / data/val.jsonl, one (query, passage, label) per line under the
frozen question wording from awesome-jev-prompts' noul-zh variant. Hard binary labels —
the dataset's own positives/hard-negatives. No teacher, no soft labels in v0.1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RAW = Path("/home/ubuntu/jev-rag-zh-benchmark/data/raw/t2reranking_dev.parquet")
QUESTION = {"answers": {"type": "noul", "instructions": "仅凭该段落能回答该查询吗？"}}
MAX_CHARS = 3000
POOL_NEGATIVES = 9


def load_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def build(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(rows):
        positives = row.get("positive") or []
        negatives = row.get("negative") or []
        if not positives or len(negatives) < POOL_NEGATIVES:
            continue
        pos = positives[0]
        negs = negatives[:POOL_NEGATIVES]
        if len(pos) > MAX_CHARS or any(len(t) > MAX_CHARS for t in negs):
            continue
        qid = f"q{i:05d}"
        out.append({
            "qid": qid,
            "query": row["query"],
            "positive": pos,
            "negatives": negs,
        })
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    rows = load_rows(RAW)
    data = build(rows, args.seed)
    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in split) + "\n",
            encoding="utf-8",
        )
        n_pos = len(split)
        n_pairs = sum(1 + len(d["negatives"]) for d in split)
        print(f"{name}: {n_pos} queries / {n_pairs} pairs -> {path}")


if __name__ == "__main__":
    main()
