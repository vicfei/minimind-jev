#!/usr/bin/env python3
"""Generate v0.2 distillation data: a chat teacher grades (query, passage) pools into the
pattern library's 4-level rubric; each passage gets a soft distribution.

Rubric frozen from awesome-jev-prompts score-patterns ("Relevance rubric (RAG)"), the same
four levels the benchmark's score-zh variant scores with:
  irrelevant — Irrelevant or off-topic / 不相关（离题）
  related    — Topically related but doesn't answer it / 主题相关但未回答问题
  partial    — Partially answers the question / 部分回答了问题
  answers    — Directly and completely answers it / 直接且完整地回答了问题
(issue #2 recorded a validated pilot prompt; the text was never committed — this
re-freezes it verbatim from the library's wording.)

Calls are pooled: one call grades all 10 passages of a query (relative grading over the
full hard-negative pool, and the per-call overhead — system prompt, reasoning — is
amortized). deepseek-v4-flash is a reasoning model; ``thinking`` is disabled by default
(validated: same JSON shape, ~40x fewer output tokens — pass ``--thinking`` to re-enable
and compare quality). Cost discipline: ``--dry-run`` grades the first 2 pools and prints
a full-run projection; breakers on failure rate and cumulative input/output tokens.
Records append-and-flush, so an aborted run resumes losslessly. The API key comes from
``$DEEPSEEK_API_KEY`` or ``~/.deepseek-key`` — never from this file, never logged.
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
    "你是一个中文检索相关性评分器。给定一个查询和一组编号段落，按以下四级评分尺"
    "独立判断每个段落对查询的回答程度：\n"
    + "\n".join(f"- {k}：{v}" for k, v in RUBRIC.items())
    + '\n\n只输出一个 JSON 数组，长度与段落数相同，第 i 个对象对应段落 i；每个对象四个键为'
    ' "irrelevant"、"related"、"partial"、"answers"，值为该等级的概率（0 到 1 的小数，'
    "保留三位，四项之和为 1）。不要输出任何其他文字。"
)
MAX_RETRIES = 2


def load_pools(root: Path, splits: list[str]) -> list[dict]:
    """One pool per query: its positive + negatives, keyed for resume/join."""
    pools: list[dict] = []
    for split in splits:
        for line in (root / "data" / f"{split}.jsonl").read_text().splitlines():
            d = json.loads(line)
            passages = [{"role": "positive", "idx": -1, "text": d["positive"], "hard": 1}]
            passages += [{"role": "negative", "idx": i, "text": neg, "hard": 0}
                         for i, neg in enumerate(d["negatives"])]
            pools.append({"qid": d["qid"], "query": d["query"], "passages": passages})
    return pools


def done_keys(out_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["qid"], r["role"], r["idx"]))
    return done


def _clean(obj: object) -> dict[str, float] | None:
    if not isinstance(obj, dict) or set(obj.keys()) != set(LEVELS):
        return None
    try:
        vals = [float(obj[k]) for k in LEVELS]
    except (TypeError, ValueError):
        return None
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        return None
    total = sum(vals)
    return {k: round(v / total, 4) for k, v in zip(LEVELS, vals)}


def parse_teacher(text: str, n_expected: int) -> list[dict | None] | None:
    """Extract the per-passage distributions; None if no usable array of the right length."""
    m = re.search(r"\[[^\[\]]*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != n_expected:
        return None
    return [_clean(o) for o in arr]


def call_teacher(pool: dict, args: argparse.Namespace) -> tuple[list[dict | None] | None, int, int, str]:
    """One pooled teacher call; returns (dists | None, input_tokens, output_tokens, error)."""
    passages = "\n\n".join(f"段落{i + 1}：{p['text']}" for i, p in enumerate(pool["passages"]))
    body = json.dumps({
        "model": args.model,
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"查询：{pool['query']}\n\n{passages}"},
        ],
        **({"thinking": {"type": "disabled"}} if not args.thinking else {}),
    }).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
    last_err = ""
    for _ in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(f"{args.base_url}/chat/completions", data=body,
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            usage = resp.get("usage") or {}
            dists = parse_teacher(resp["choices"][0]["message"]["content"], len(pool["passages"]))
            err = "" if dists is not None else "unparseable"
            return (dists, int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0), err)
        except Exception as e:  # noqa: BLE001 - network/HTTP errors retry once
            last_err = repr(e)
            time.sleep(2)
    return None, 0, 0, last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", default="data/distilled.jsonl")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--thinking", action="store_true",
                    help="keep the reasoning channel on (default off: ~40x fewer output tokens, same JSON)")
    ap.add_argument("--dry-run", action="store_true", help="grade only the first 2 pools, then stop with projections")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of pools graded this run")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--max-input-tokens", type=int, default=6_500_000,
                    help="circuit breaker: cumulative teacher input tokens")
    ap.add_argument("--max-output-tokens", type=int, default=1_200_000,
                    help="circuit breaker: cumulative teacher output tokens")
    ap.add_argument("--fail-rate-cap", type=float, default=0.02)
    ap.add_argument("--price-in", type=float, default=1.0, help="¥/M input cache-miss, display only (off-peak; peak doubles)")
    ap.add_argument("--price-out", type=float, default=4.5, help="¥/M output, display only (upper bound; peak doubles)")
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
    pools = load_pools(root, args.splits.split(","))
    done = done_keys(out_path)
    todo = [p for p in pools if not all((p["qid"], q["role"], q["idx"]) in done
                                        for q in p["passages"])]
    if args.dry_run:
        todo = todo[:2]
    elif args.limit:
        todo = todo[: args.limit]

    n_pairs = sum(len(p["passages"]) for p in pools)
    print(f"pools total={len(pools)} (pairs={n_pairs}) done_pairs={n_pairs - sum(len(p['passages']) for p in todo)} "
          f"todo_pools={len(todo)}{' (dry run: first 2 pools)' if args.dry_run else ''}")
    out_path.parent.mkdir(exist_ok=True)
    failures = records = calls = input_tokens = output_tokens = 0
    breaker = ""
    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        for pool, result in zip(todo, pool_exec.map(lambda p: call_teacher(p, args), todo)):
            dists, in_toks, out_toks, err = result
            calls += 1
            input_tokens += in_toks
            output_tokens += out_toks
            if dists is None:
                failures += len(pool["passages"])
                print(f"FAIL pool {pool['qid']}: {err}", file=sys.stderr)
            else:
                for p, dist in zip(pool["passages"], dists):
                    k = (pool["qid"], p["role"], p["idx"])
                    if dist is None or k in done:
                        if dist is None:
                            failures += 1
                        continue
                    rec = {"qid": pool["qid"], "role": p["role"], "idx": p["idx"],
                           "hard": p["hard"], "dist": dist, "teacher": args.model}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(k)
                    records += 1
            out.flush()
            if records + failures >= 200 and failures / (records + failures) > args.fail_rate_cap:
                breaker = f"failure rate {failures}/{records + failures} > {args.fail_rate_cap:.0%}"
                break
            if input_tokens > args.max_input_tokens or output_tokens > args.max_output_tokens:
                breaker = (f"token budget hit (in {input_tokens:,} / out {output_tokens:,})")
                break

    est = input_tokens / 1e6 * args.price_in + output_tokens / 1e6 * args.price_out
    print(f"pools={calls} records={records} failed_records={failures} "
          f"input_tokens={input_tokens:,} output_tokens={output_tokens:,} (est. ¥{est:.2f}); output={out_path}")
    if args.dry_run and calls:
        remaining_pools = len(pools) - len(todo)
        per_pool_in, per_pool_out = input_tokens / calls, output_tokens / calls
        proj = (per_pool_in * remaining_pools / 1e6 * args.price_in
                + per_pool_out * remaining_pools / 1e6 * args.price_out)
        print(f"projection for remaining {remaining_pools} pools: "
              f"~{per_pool_in * remaining_pools / 1e6:.1f}M in + ~{per_pool_out * remaining_pools / 1e6:.1f}M out, est. ¥{proj:.2f}")
    if breaker:
        sys.exit(f"aborted by circuit breaker: {breaker} — progress kept, rerun to resume")


if __name__ == "__main__":
    main()
