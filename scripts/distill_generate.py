#!/usr/bin/env python3
"""Generate v0.2 distillation data: a chat teacher grades (query, passage) pairs into the
pattern library's 4-level relevance rubric; each pair gets a soft distribution.

Rubric frozen from awesome-jev-prompts score-patterns ("Relevance rubric (RAG)"), the same
four levels the benchmark's score-zh variant scores with:
  irrelevant — Irrelevant or off-topic / 不相关（离题）
  related    — Topically related but doesn't answer it / 主题相关但未回答问题
  partial    — Partially answers the question / 部分回答了问题
  answers    — Directly and completely answers it / 直接且完整地回答了问题
(issue #2 recorded a validated pilot prompt; the text was never committed — this
re-freezes it verbatim from the library's wording.)

Cost discipline: run ``--dry-run`` (20 pairs) before the full pass; breakers on call
count, parse-failure rate, and cumulative input tokens. Records append-and-flush, so an
aborted run resumes losslessly. The API key comes from ``$DEEPSEEK_API_KEY`` or
``~/.deepseek_key`` — never from this file, never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LEVELS = ["irrelevant", "related", "partial", "answers"]
RUBRIC = {
    "irrelevant": "不相关（离题）：段落与查询无关或纯属跑题。",
    "related": "主题相关但未回答问题：段落与查询同主题，但没有回答查询。",
    "partial": "部分回答了问题：段落对查询给出了部分回答。",
    "answers": "直接且完整地回答了问题：段落直接、完整地回答了查询。",
}
SYSTEM_PROMPT = (
    "你是一个中文检索相关性评分器。给定一个查询和一个段落，按以下四级评分尺判断"
    "段落对查询的回答程度：\n"
    + "\n".join(f"- {k}：{v}" for k, v in RUBRIC.items())
    + '\n\n只输出一个 JSON 对象，四个键为 "irrelevant"、"related"、"partial"、"answers"，'
    "值为该等级的概率（0 到 1 的小数，保留三位，四项之和为 1）。不要输出任何其他文字。"
)
MAX_RETRIES = 2


def load_pairs(root: Path, splits: list[str]) -> list[dict]:
    tasks: list[dict] = []
    for split in splits:
        path = root / "data" / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            d = json.loads(line)
            tasks.append({"qid": d["qid"], "role": "positive", "idx": -1,
                          "query": d["query"], "passage": d["positive"], "hard": 1})
            for i, neg in enumerate(d["negatives"]):
                tasks.append({"qid": d["qid"], "role": "negative", "idx": i,
                              "query": d["query"], "passage": neg, "hard": 0})
    return tasks


def done_keys(out_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["qid"], r["role"], r["idx"]))
    return done


def parse_teacher(text: str) -> dict[str, float] | None:
    """Extract the 4-level distribution from a teacher reply; None if unusable."""
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if set(obj.keys()) != set(LEVELS):
        return None
    try:
        vals = [float(obj[k]) for k in LEVELS]
    except (TypeError, ValueError):
        return None
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        return None
    total = sum(vals)
    return {k: round(v / total, 4) for k, v in zip(LEVELS, vals)}


def call_teacher(task: dict, args: argparse.Namespace) -> tuple[dict | None, int, str]:
    """One teacher call; returns (dist | None, input_tokens, error | '')."""
    body = json.dumps({
        "model": args.model,
        "temperature": 0.2,
        "max_tokens": 80,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"查询：{task['query']}\n\n段落：{task['passage']}"},
        ],
    }).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
    last_err = ""
    for _ in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(f"{args.base_url}/chat/completions", data=body,
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                resp = json.load(r)
            usage = resp.get("usage") or {}
            dist = parse_teacher(resp["choices"][0]["message"]["content"])
            return dist, int(usage.get("prompt_tokens") or 0), "" if dist else "unparseable"
        except Exception as e:  # noqa: BLE001 - network/HTTP errors retry once
            last_err = repr(e)
            time.sleep(2)
    return None, 0, last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", default="data/distilled.jsonl")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--dry-run", action="store_true", help="grade only the first 20 pairs, then stop with projections")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of pairs graded this run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-input-tokens", type=int, default=25_000_000,
                    help="circuit breaker: cumulative teacher input tokens (est. <¥5 at flash pricing with cache)")
    ap.add_argument("--fail-rate-cap", type=float, default=0.02)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not key:
        key_file = Path.home() / ".deepseek-key"
        if key_file.exists():
            key = key_file.read_text().strip()
    if not key:
        sys.exit("error: no API key — set $DEEPSEEK_API_KEY or create ~/.deepseek-key")
    os.environ["DEEPSEEK_API_KEY"] = key

    root = Path(__file__).resolve().parent.parent
    out_path = root / args.out
    tasks = load_pairs(root, args.splits.split(","))
    done = done_keys(out_path)
    todo = [t for t in tasks if (t["qid"], t["role"], t["idx"]) not in done]
    if args.dry_run:
        todo = todo[:20]
    elif args.limit:
        todo = todo[: args.limit]

    print(f"pairs total={len(tasks)} done={len(done)} todo={len(todo)}"
          f"{' (dry run)' if args.dry_run else ''}")
    out_path.parent.mkdir(exist_ok=True)
    failures = calls = input_tokens = 0
    breaker = ""
    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task, result in zip(todo, pool.map(lambda t: call_teacher(t, args), todo)):
            dist, toks, err = result
            calls += 1
            input_tokens += toks
            if dist is None:
                failures += 1
                print(f"FAIL {task['qid']}/{task['role']}{task['idx']}: {err}", file=sys.stderr)
                continue
            rec = {**task, "dist": dist, "teacher": args.model}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if calls >= 200 and failures / calls > args.fail_rate_cap:
                breaker = f"failure rate {failures}/{calls} > {args.fail_rate_cap:.0%}"
                break
            if input_tokens > args.max_input_tokens:
                breaker = f"input-token budget hit ({input_tokens:,} > {args.max_input_tokens:,})"
                break

    ok = calls - failures
    est_cny = input_tokens / 1e6 * 2.0  # rough flash-tier estimate, display only
    print(f"graded ok={ok} failed={failures} calls={calls} input_tokens={input_tokens:,} "
          f"(est. ¥{est_cny:.2f}); output={out_path}")
    if args.dry_run and calls:
        full_est = est_cny / calls * (len(tasks) - len(done))
        print(f"projection for remaining {len(tasks) - len(done)} pairs: "
              f"~{input_tokens / calls * (len(tasks) - len(done)) / 1e6:.1f}M input tokens, est. ¥{full_est:.2f}")
    if breaker:
        sys.exit(f"aborted by circuit breaker: {breaker} — progress kept, rerun to resume")


if __name__ == "__main__":
    main()
