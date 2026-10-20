"""Tests for the v0.2 distillation path (no network, no encoder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from distill_generate import LEVELS, load_pools, parse_teacher  # noqa: E402


def _dist(v: str = "answers") -> dict:
    base = {"irrelevant": 0.1, "related": 0.2, "partial": 0.3, "answers": 0.4}
    return base


def test_parse_teacher_array_clean():
    arr = [{"irrelevant": 0.1, "related": 0.2, "partial": 0.3, "answers": 0.4} for _ in range(10)]
    out = parse_teacher(json.dumps(arr, ensure_ascii=False), 10)
    assert out is not None and len(out) == 10
    assert out[0] == {"irrelevant": 0.1, "related": 0.2, "partial": 0.3, "answers": 0.4}


def test_parse_teacher_wrapped_in_prose():
    text = ('好的：\n```json\n'
            + json.dumps([_dist() for _ in range(3)])
            + '\n```\n以上。')
    out = parse_teacher(text, 3)
    assert out is not None and len(out) == 3


def test_parse_teacher_wrong_length_rejected():
    arr = [_dist() for _ in range(9)]
    assert parse_teacher(json.dumps(arr), 10) is None


def test_parse_teacher_salvages_bad_entries():
    arr = [_dist(), {"irrelevant": 1.0}, _dist()]
    out = parse_teacher(json.dumps(arr), 3)
    assert out is not None and out[1] is None and out[0] is not None and out[2] is not None


def test_parse_teacher_renormalizes():
    arr = [{"irrelevant": 2, "related": 2, "partial": 4, "answers": 2}]
    out = parse_teacher(json.dumps(arr), 1)
    assert out is not None and abs(out[0]["partial"] - 0.4) < 1e-6
    assert abs(sum(out[0].values()) - 1.0) < 1e-6


def test_parse_teacher_rejects_garbage():
    assert parse_teacher("no json here", 10) is None
    assert parse_teacher('{"irrelevant": 1.0}', 10) is None  # object, not array


def test_load_pools_groups_query_passages(tmp_path):
    (tmp_path / "data").mkdir()
    row = {"qid": "q00001", "query": "q", "positive": "p", "negatives": ["n1", "n2"]}
    (tmp_path / "data" / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    (tmp_path / "data" / "val.jsonl").write_text("")
    pools = load_pools(tmp_path, ["train"])
    assert len(pools) == 1
    pool = pools[0]
    assert pool["qid"] == "q00001" and len(pool["passages"]) == 3
    assert pool["passages"][0]["role"] == "positive" and pool["passages"][0]["hard"] == 1
    assert [p["idx"] for p in pool["passages"][1:]] == [0, 1]
    assert all(p["hard"] == 0 for p in pool["passages"][1:])
    assert LEVELS == ["irrelevant", "related", "partial", "answers"]
