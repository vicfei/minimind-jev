#!/usr/bin/env python3
"""Train minimind-jev v0.3 from scratch: optional short MLM pretraining on the task
corpus, then binary-BCE decision fine-tuning (the v0.2 lesson: when gold labels exist
for the metric, direct supervision is the default), then temperature calibration.

Laptop RTX 4060 (8GB) budget: tokenizer ~1 min (separate script) + MLM ~15-25 min +
fine-tune ~10-20 min — comfortably inside the "single GPU, under 2h" minimind anchor.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minimind_jev.config import V3Config  # noqa: E402
from minimind_jev.data import corpus_lines, encode_batch, load_splits, mask_mlm, pair_examples  # noqa: E402
from minimind_jev.model import DecisionModel, MLMModel, param_count  # noqa: E402
from minimind_jev.tokenizer import load_tokenizer  # noqa: E402


def batches(n: int, bs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            yield perm[i : i + bs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="model/tokenizer.json")
    ap.add_argument("--mlm-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--mlm-batch", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mlm-lr", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model/v3.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).resolve().parent.parent
    tok_path = root / args.tokenizer
    if not tok_path.exists():
        raise SystemExit(f"error: {tok_path} not found — run scripts/train_tokenizer.py first")
    tokenizer = load_tokenizer(tok_path)

    cfg = V3Config()
    cfg.vocab_size = tokenizer.get_vocab_size()  # actual trained vocab (may be < 8K on small corpora)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    mask_id = tokenizer.token_to_id("[MASK]") or 1
    print(f"device: {device} | vocab {cfg.vocab_size} | ctx {cfg.ctx} | "
          f"{cfg.n_layers}L d{cfg.d_model} ff{cfg.d_ff}")

    train_rows = load_splits(root, ("train",))
    val_rows = load_splits(root, ("val",))

    # ---- stage 1: MLM pretraining on the unlabeled corpus ----
    model = DecisionModel(cfg).to(device)
    if args.mlm_epochs > 0:
        corpus = corpus_lines(load_splits(root))
        mlm = MLMModel(cfg).to(device)
        # share the encoder so pretraining weights carry straight into the decision model
        mlm.encoder.load_state_dict(model.encoder.state_dict())
        opt = torch.optim.AdamW(mlm.parameters(), lr=args.mlm_lr, weight_decay=0.01)
        t0 = time.time()
        step_iter = batches(len(corpus), args.mlm_batch, args.seed)
        steps_per_epoch = (len(corpus) + args.mlm_batch - 1) // args.mlm_batch
        for epoch in range(args.mlm_epochs):
            mlm.train()
            tot, seen = 0.0, 0
            for _ in range(steps_per_epoch):
                idx = next(step_iter)
                texts = [corpus[i] for i in idx.tolist()]
                ids, keep = encode_batch(tokenizer, texts, cfg.ctx, pad_id)
                masked, targets = mask_mlm(ids, mask_id)
                logits = mlm(masked.to(device), keep.to(device))
                loss = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size), targets.to(device).view(-1), ignore_index=-100)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx); seen += len(idx)
            print(f"mlm epoch {epoch + 1}: loss={tot / seen:.4f} ({time.time() - t0:.0f}s elapsed)")
        model.encoder.load_state_dict(mlm.encoder.state_dict())
        del mlm

    # ---- stage 2: binary-BCE decision fine-tuning ----
    print(f"model params: {param_count(model):,}")
    Xtr_texts, ytr = pair_examples(train_rows)
    Xva_texts, yva = pair_examples(val_rows)
    print(f"train {len(Xtr_texts)} pairs / val {len(Xva_texts)} pairs")
    # pre-encode once (ids fixed across epochs, all padded to ctx width so batches stack)
    def encode_all(texts, bs=128):
        ids_l, keep_l = [], []
        for i in range(0, len(texts), bs):
            ids, keep = encode_batch(tokenizer, texts[i : i + bs], cfg.ctx, pad_id)
            ids_l.append(ids.to(device)); keep_l.append(keep.to(device))
        return torch.cat(ids_l), torch.cat(keep_l)

    tr_ids, tr_keep = encode_all(Xtr_texts)
    ytr_t = torch.tensor(ytr, device=device)
    va_ids, va_keep = encode_all(Xva_texts)
    yva_t = torch.tensor(yva, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        tot, seen = 0.0, 0
        for idx in batches(len(ytr_t), args.batch, args.seed + epoch):
            logits = model(tr_ids[idx], tr_keep[idx])
            loss = lossf(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); seen += len(idx)
        model.eval()
        with torch.no_grad():
            va_logits = torch.cat([model(va_ids[i : i + 256], va_keep[i : i + 256])
                                   for i in range(0, len(yva_t), 256)])
            va_loss = lossf(va_logits, yva_t).item()
            acc = ((va_logits > 0).float() == yva_t).float().mean().item()
        print(f"epoch {epoch + 1}: train_loss={tot / seen:.4f} val_loss={va_loss:.4f} val_acc={acc:.4f}")

    # ---- stage 3: temperature on val ----
    with torch.no_grad():
        logits = va_logits
    T = nn.Parameter(torch.ones(1))
    topt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        topt.zero_grad()
        loss = lossf(logits / T, yva_t)
        loss.backward()
        return loss

    topt.step(closure)
    print(f"temperature (fitted on val): {float(T.item()):.3f}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg.dict(), "temp": float(T.item()),
                "tokenizer": str(args.tokenizer), "pad_id": pad_id}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
