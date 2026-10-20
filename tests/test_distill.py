"""Tests for the v0.2 distillation path (no network, no encoder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from distill_generate import LEVELS, load_pairs, parse_teacher  # noqa: E402


def test_parse_teacher_clean():
    d = parse_teacher('{"irrelevant": 0.1, "related": 0.2, "partial": 0.3, "answers": 0.4}')
    assert d == {"irrelevant": 0.1, "related": 0.2, "partial": 0.3, "answers": 0.4}


def test_parse_teacher_fenced_and_wrapped():
    text = '好的，评分如下：\n```json\n{"irrelevant": 0.0, "related": 0.1, "partial": 0.2, "answers": 0.7}\n```\n以上。'
    d = parse_teacher(text)
    assert d is not None and abs(d["answers"] - 0.7) < 1e-6


def test_parse_teacher_renormalizes():
    d = parse_teacher('{"irrelevant": 2, "related": 2, "partial": 4, "answers": 2}')
    assert d is not None and abs(sum(d.values()) - 1.0) < 1e-6
    assert abs(d["partial"] - 0.4) < 1e-6


def test_parse_teacher_rejects_bad():
    assert parse_teacher("no json here") is None
    assert parse_teacher('{"irrelevant": 1.0}') is None  # wrong keys
    assert parse_teacher('{"irrelevant": -1, "related": 0.5, "partial": 0.5, "answers": 0.5}') is None
    assert parse_teacher('{"irrelevant": 0, "related": 0, "partial": 0, "answers": 0}') is None


def test_load_pairs_enumerates_pos_and_negs(tmp_path):
    (tmp_path / "data").mkdir()
    row = {"qid": "q00001", "query": "q", "positive": "p", "negatives": ["n1", "n2"]}
    (tmp_path / "data" / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    (tmp_path / "data" / "val.jsonl").write_text("")
    tasks = load_pairs(tmp_path, ["train"])
    assert len(tasks) == 3
    assert tasks[0]["role"] == "positive" and tasks[0]["hard"] == 1
    assert [t["idx"] for t in tasks[1:]] == [0, 1]
    assert all(t["hard"] == 0 for t in tasks[1:])
    assert set(tasks[0]) >= {"qid", "role", "idx", "query", "passage", "hard"}
    assert LEVELS == ["irrelevant", "related", "partial", "answers"]
