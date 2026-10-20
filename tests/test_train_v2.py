"""Tests for the v0.2 4-level head (synthetic tensors only, no encoder download)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_v2 import LEVEL_WEIGHTS, HeadV2  # noqa: E402


def test_head_v2_forward_shape():
    head = HeadV2(dim=32)
    out = head(torch.randn(7, 32))
    assert out.shape == (7, 4)


def test_head_v2_soft_ce_step_reduces_loss():
    torch.manual_seed(0)
    head = HeadV2(dim=16)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    X = torch.randn(64, 16)
    # teacher that puts all mass on "answers" for half the rows, "irrelevant" for the rest
    y = torch.zeros(64, 4)
    y[:32, 3] = 1.0
    y[32:, 0] = 1.0
    loss0 = -(y * torch.log_softmax(head(X), dim=-1)).sum(-1).mean().item()
    for _ in range(50):
        loss = -(y * torch.log_softmax(head(X), dim=-1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    loss1 = -(y * torch.log_softmax(head(X), dim=-1)).sum(-1).mean().item()
    assert loss1 < loss0


def test_level_weights_monotone():
    w = LEVEL_WEIGHTS.tolist()
    assert w == sorted(w) and w[0] == 0.0 and w[-1] == 1.0
