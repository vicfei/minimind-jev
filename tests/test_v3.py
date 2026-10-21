"""v0.3 tests: tiny configs, CPU-only, no network, no encoder downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minimind_jev.config import V3Config  # noqa: E402
from minimind_jev.data import encode_batch, mask_mlm  # noqa: E402
from minimind_jev.model import DecisionModel, MLMModel, param_count  # noqa: E402
from minimind_jev.tokenizer import train_bpe  # noqa: E402


def tiny_cfg() -> V3Config:
    return V3Config(vocab_size=256, d_model=32, n_layers=2, n_heads=4, d_ff=64, ctx=16)


def test_forward_shapes():
    cfg = tiny_cfg()
    ids = torch.randint(4, 200, (3, 12))
    keep = torch.ones(3, 12, dtype=torch.bool)
    dm = DecisionModel(cfg)
    mlm = MLMModel(cfg)
    assert dm(ids, keep).shape == (3,)
    assert mlm(ids, keep).shape == (3, 12, cfg.vocab_size)


def test_padding_invariance():
    """Mean-pool over the keep mask: trailing pads must not move the logit."""
    torch.manual_seed(0)
    dm = DecisionModel(tiny_cfg()).eval()
    ids = torch.tensor([[5, 6, 7]])
    keep = torch.tensor([[True, True, True]])
    ids2 = torch.tensor([[5, 6, 7, 0, 0]])
    keep2 = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        a = dm(ids, keep).item()
        b = dm(ids2, keep2).item()
    assert abs(a - b) < 1e-5


def test_default_param_count_near_14m():
    n = param_count(DecisionModel(V3Config()))
    assert 12_000_000 < n < 16_000_000, n


def test_mask_mlm_targets():
    ids = torch.randint(4, 200, (2, 20))
    masked, targets = mask_mlm(ids, mask_id=1, p=0.5, generator=torch.Generator().manual_seed(0))
    selected = targets != -100
    assert selected.float().mean().item() > 0.2  # ~50% requested, tolerate variance
    assert (masked[selected] == 1).all()
    assert (targets[~selected] == -100).all()
    assert (masked[~selected] == ids[~selected]).all()


def test_tokenizer_roundtrip_and_encode(tmp_path):
    corpus = ["量子计算利用量子叠加与纠缠", "股市今天大幅上涨", "检索增强生成系统"] * 20
    tok = train_bpe(corpus, vocab_size=300, out_path=tmp_path / "tok.json")
    s = "中文分词测试句子。"
    ids = tok.encode(s).ids
    assert len(ids) > 0
    assert tok.decode(ids) == s  # byte-level BPE round-trips exactly
    texts = ["短句", "这一句相对长一些" * 8]
    padded, keep = encode_batch(tok, texts, ctx=16, pad_id=0)
    assert padded.shape == (2, 16) and keep.dtype == torch.bool
    assert keep[0].sum().item() <= keep[1].sum().item()
    assert (padded[~keep] == 0).all()


def test_bce_step_reduces_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    dm = DecisionModel(cfg)
    opt = torch.optim.AdamW(dm.parameters(), lr=3e-3)
    ids = torch.randint(4, 200, (8, 12))
    keep = torch.ones(8, 12, dtype=torch.bool)
    y = torch.tensor([1.0, 0.0] * 4)
    lossf = torch.nn.BCEWithLogitsLoss()
    loss0 = lossf(dm(ids, keep), y).item()
    for _ in range(30):
        loss = lossf(dm(ids, keep), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert lossf(dm(ids, keep), y).item() < loss0
