#!/usr/bin/env python3
"""Export a trained v0.3 checkpoint to ONNX (chapter 7 artifact).

Usage: python scripts/export_onnx.py --checkpoint model/v3-lite.pt --out model/v3-lite.onnx
Verify numerics afterwards with --verify (needs onnxruntime): ONNX logits must match
torch within 1e-4 on the same batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minimind_jev.config import V3Config  # noqa: E402
from minimind_jev.data import QUESTION, encode_batch  # noqa: E402
from minimind_jev.model import DecisionModel  # noqa: E402
from minimind_jev.tokenizer import load_tokenizer  # noqa: E402


def build_model(ckpt: dict) -> tuple[DecisionModel, V3Config]:
    cfg = V3Config(**ckpt["cfg"])
    model = DecisionModel(cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model/v3-lite.pt")
    ap.add_argument("--out", default="model/v3-lite.onnx")
    ap.add_argument("--verify", action="store_true", help="compare ONNX vs torch logits on a sample batch")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    ckpt = torch.load(root / args.checkpoint, map_location="cpu")
    model, cfg = build_model(ckpt)

    ids = torch.randint(4, cfg.vocab_size, (2, 64))
    keep = torch.ones(2, 64, dtype=torch.bool)
    out_path = root / args.out
    torch.onnx.export(
        model, (ids, keep), str(out_path),
        input_names=["ids", "keep"], output_names=["logit"],
        dynamic_axes={"ids": {0: "batch", 1: "seq"}, "keep": {0: "batch", 1: "seq"}},
        opset_version=18,  # Split(num_outputs) needs >=18 for onnxruntime to load
    )
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if args.verify:
        import onnxruntime as ort  # optional dependency, chapter-7 verification only

        tokenizer = load_tokenizer(root / ckpt["tokenizer"])
        pad_id = ckpt.get("pad_id", 0)
        texts = [f"量子计算的基本原理\n{QUESTION}\n量子计算利用量子叠加与纠缠对信息编码。",
                 f"量子计算的基本原理\n{QUESTION}\n今日股市大涨，科技板块领涨。"]
        v_ids, v_keep = encode_batch(tokenizer, texts, cfg.ctx, pad_id)
        with torch.no_grad():
            ref = model(v_ids, v_keep)
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        got = torch.tensor(sess.run(None, {"ids": v_ids.numpy(), "keep": v_keep.numpy()})[0])
        diff = float((ref - got).abs().max())
        print(f"verify: torch {ref.tolist()} vs onnx {got.tolist()} | max|Δ|={diff:.2e}")
        assert diff < 1e-4, "ONNX numerics diverged"
        print("verify: PASS (<1e-4)")


if __name__ == "__main__":
    main()
