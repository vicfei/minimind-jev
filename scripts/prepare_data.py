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


def resolve_raw(explicit: str | None) -> Path:
    """Explicit path > local benchmark copy > download from HuggingFace."""
    import os
    import urllib.request

    candidates = [Path(explicit)] if explicit else [
        RAW,
        Path(os.environ.get("MINIMIND_JEV_PARQUET", "")) if os.environ.get("MINIMIND_JEV_PARQUET") else None,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    dest = Path(__file__).resolve().parent.parent / "data" / "raw" / "t2reranking_dev.parquet"
    if not dest.exists():
        import os

        base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{base}/api/datasets/C-MTEB/T2Reranking/parquet/default/dev/0.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} …")
        req = urllib.request.Request(url, headers={"User-Agent": "minimind-jev/0.1"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return dest


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
    ap.add_argument("--parquet", default=None, help="path to t2reranking_dev.parquet (else auto-resolved)")
    ap.add_argument("--exclude-sample",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "eval_sample.jsonl"),
                    help="benchmark eval sample; its queries are excluded from train/val (no leakage). "
                         "In-repo copy ships with the repo. Pass '' to disable.")
    args = ap.parse_args()

    rows = load_rows(resolve_raw(args.parquet))
    data = build(rows, args.seed)

    excluded = 0
    if args.exclude_sample:
        exclude_path = Path(args.exclude_sample)
        if not exclude_path.exists():
            raise SystemExit(
                f"error: --exclude-sample '{args.exclude_sample}' not found — refusing to build a "
                "leaky split. Use the in-repo data/eval_sample.jsonl, pass a valid path, "
                "or pass '' to disable exclusion explicitly.")
        bench_queries = {json.loads(l)["query"] for l in exclude_path.read_text().splitlines()}
        before = len(data)
        data = [d for d in data if d["query"] not in bench_queries]
        excluded = before - len(data)
        print(f"excluded {excluded} queries (present in the benchmark eval sample — no leakage)")
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
