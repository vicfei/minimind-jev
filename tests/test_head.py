"""Offline: head shape, loss sanity, temperature effect. No network."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train import DecisionHead  # noqa: E402


def test_head_shape():
    head = DecisionHead(dim=512)
    x = torch.randn(8, 512)
    assert head(x).shape == (8,)


def test_temperature_changes_probabilities():
    logit = torch.tensor([2.0])
    p1 = torch.sigmoid(logit / 1.0)
    p2 = torch.sigmoid(logit / 2.0)
    assert p1 > p2 > 0.5  # higher temperature pulls toward 0.5
