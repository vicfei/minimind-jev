#!/usr/bin/env python3
"""Train the v0.3 byte-level BPE tokenizer on the task corpus (queries + passages)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minimind_jev.config import V3Config  # noqa: E402
from minimind_jev.data import corpus_lines, load_splits  # noqa: E402
from minimind_jev.tokenizer import train_bpe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=V3Config().vocab_size)
    ap.add_argument("--out", default="model/tokenizer.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = load_splits(root)
    lines = corpus_lines(rows)
    chars = sum(len(l) for l in lines)
    print(f"corpus: {len(lines)} unique texts, {chars:,} chars")
    tok = train_bpe(lines, args.vocab, root / args.out)
    print(f"vocab size: {tok.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
