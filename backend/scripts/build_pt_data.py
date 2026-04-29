#!/usr/bin/env python3
"""
Build PT (continued-pretraining) data for DeepScout LoRA replay.

Two sources:
  --source domain        Extract web-page bodies from existing
                         `deep_read_url` call logs. Free, but only as
                         much as you've already crawled.
  --source self-replay   Have the SAME base model (qwen2.5-7b-instruct
                         on DashScope, matching what we'll LoRA-tune)
                         generate diverse natural text on a built-in
                         seed-prompt pool. Used to keep the model from
                         forgetting general capabilities during SFT.

Mix into the SFT batch ≈20%. Each row is `{"text": ...}` plus metadata.

Usage:
    # domain (default)
    python scripts/build_pt_data.py

    # self-replay (needs DASHSCOPE_API_KEY)
    DASHSCOPE_API_KEY=sk-... python scripts/build_pt_data.py \\
        --source self-replay --target 1500 --concurrency 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "deepscout_logs"
DEFAULT_OUTPUT_DOMAIN = DEFAULT_LOG_DIR / "pt_data_domain.jsonl"
DEFAULT_OUTPUT_SELF_REPLAY = DEFAULT_LOG_DIR / "pt_data_self_replay.jsonl"

CONTENT_RE = re.compile(r"## 文档内容\s*\n(.*?)\n\s*## 任务", re.DOTALL)
URL_RE = re.compile(r"URL:\s*(\S+)")
TITLE_RE = re.compile(r"标题:\s*(.+)")


def iter_log_records(log_dir: Path) -> Iterator[dict]:
    files = sorted(log_dir.glob("llm_calls_*.jsonl"))
    if not files:
        print(f"[warn] no llm_calls_*.jsonl found under {log_dir}", file=sys.stderr)
        return
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {f.name}:{ln} bad json: {e}", file=sys.stderr)


def is_deep_read(rec: dict) -> bool:
    if rec.get("agent") != "DeepScout":
        return False
    sp = rec.get("system_prompt") or ""
    up = rec.get("user_prompt") or ""
    if "你是一位专业的文档分析师" in sp:
        return True
    return "## 文档内容" in up and "## 任务" in up


def extract_body(user_prompt: str) -> Optional[dict]:
    m = CONTENT_RE.search(user_prompt)
    if not m:
        return None
    content = m.group(1).strip()
    if not content:
        return None
    url_m = URL_RE.search(user_prompt)
    title_m = TITLE_RE.search(user_prompt)
    return {
        "text": content,
        "url": url_m.group(1).strip() if url_m else "",
        "title": title_m.group(1).strip() if title_m else "",
    }


def fingerprint(text: str) -> str:
    # collapse whitespace before hashing so trivially-different copies dedupe
    norm = re.sub(r"\s+", " ", text).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# ============================================================
# Self-replay
# ============================================================

# Seed prompts span the general capabilities we don't want SFT to erode:
# open-domain QA, writing, code, reasoning, summarization, dialogue.
# Keep them open-ended so high-temperature sampling produces diverse text.
SELF_REPLAY_SEEDS: list[tuple[str, str]] = [
    # (category, prompt)
    ("qa_zh", "请用中文详细解释一下「微分方程」在工程实践中的常见应用场景。"),
    ("qa_zh", "用通俗的语言介绍光合作用的完整过程，并举一个生活中的例子。"),
    ("qa_zh", "为什么互联网公司近几年开始重视「数据中台」？请说明利弊。"),
    ("qa_zh", "比较一下 HTTP/1.1、HTTP/2、HTTP/3 的核心差异，重点讲解性能层面的演进。"),
    ("qa_zh", "请介绍中国新能源汽车产业近五年的关键转折点和驱动因素。"),
    ("qa_en", "Explain the difference between Bayesian and frequentist statistics with a concrete example."),
    ("qa_en", "Walk me through how a modern CPU pipeline executes an instruction, including hazards."),
    ("qa_en", "What are the main reasons distributed systems often choose eventual consistency over strong consistency?"),
    ("writing_zh", "以「秋日的图书馆」为题，写一段 300 字左右的散文，要求有具体的感官描写。"),
    ("writing_zh", "为一款主打「专注力」的番茄钟 App 写一段产品介绍文案，约 200 字。"),
    ("writing_zh", "写一封给十年后自己的信，主题是工作与生活的平衡，约 400 字。"),
    ("writing_zh", "以一名宋代茶馆老板的视角，写一段日记，描述一天的见闻。"),
    ("writing_en", "Write a short product blog post (~300 words) introducing a fictional AI tool that helps researchers organize papers."),
    ("writing_en", "Draft a polite email declining a meeting invitation while offering an asynchronous alternative."),
    ("code_py", "用 Python 实现一个支持 LRU 淘汰策略的线程安全缓存类，给出代码并解释关键点。"),
    ("code_py", "Write a Python function that streams a large CSV file and yields rows whose timestamp falls inside a given window. Include error handling."),
    ("code_sql", "给定订单表 orders(id, user_id, amount, created_at)，写一段 SQL 查出每个用户最近 30 天消费总额，并按金额降序排列。"),
    ("code_js", "用 JavaScript 实现一个简单的事件总线（EventBus），支持 on/off/once/emit，并写两个使用示例。"),
    ("code_go", "Show a minimal Go HTTP server with graceful shutdown handling SIGINT/SIGTERM."),
    ("reason_math", "一个袋子里有 3 个红球、2 个白球，不放回地依次抽取两次，第二次抽到红球的概率是多少？请写出推导。"),
    ("reason_math", "Prove that the sum of the first n odd numbers equals n^2, and explain the geometric intuition."),
    ("reason_logic", "三个人 A、B、C 中只有一人说真话：A 说 B 说谎，B 说 C 说谎，C 说 A 和 B 都说谎。请推断谁说真话，并写出推理过程。"),
    ("reason_logic", "You have 12 coins, one of which is heavier or lighter. Using a balance scale at most three times, identify the odd coin. Walk through your strategy."),
    ("summary_zh", "请把以下要点扩写成一段 200 字左右的新闻短讯：央行下调存款准备金率 0.5%，释放长期资金约一万亿元，旨在支持实体经济。"),
    ("summary_en", "Summarize in 3 bullet points the typical tradeoffs between microservices and monolithic architectures."),
    ("dialog_zh", "续写一段两个程序员关于「是否要在小项目里上 Kubernetes」的辩论对话，至少 6 轮。"),
    ("dialog_en", "Continue a dialogue between a junior engineer and a tech lead reviewing a tricky bug in a payment system. At least 6 turns."),
    ("explain_industry", "用结构化的方式介绍光伏产业链：上游（硅料/硅片）、中游（电池片/组件）、下游（系统集成/电站），各环节的代表企业和主要竞争壁垒。"),
    ("explain_industry", "解释半导体晶圆代工行业的商业模式，与 IDM 模式的差异，并讨论先进制程的资本壁垒。"),
    ("explain_industry", "Give a structured overview of the electric vehicle battery industry: cathode/anode/electrolyte/separator, key suppliers, and current technology frontiers."),
    ("explain_concept", "请用类比的方式解释什么是「向量数据库」，以及它与传统关系型数据库的区别。"),
    ("explain_concept", "Explain how RAG (retrieval-augmented generation) works to a product manager who knows ML basics but not LLM internals."),
    ("howto_zh", "如何为一个 10 人的研发团队搭建第一版的可观测性体系？请按优先级给出建议。"),
    ("howto_en", "What are concrete steps a small startup can take to harden their AWS account against common security mistakes?"),
    ("compare_zh", "对比一下 PostgreSQL 和 MySQL 在 OLTP 场景下的优劣，请覆盖一致性、扩展性、生态。"),
    ("compare_en", "Compare React Server Components and traditional SSR. When would you pick each?"),
    ("creative_zh", "为「未来博物馆里的一件 21 世纪互联网文物」写一段博物馆讲解词，约 250 字。"),
    ("creative_en", "Imagine a near-future city where private cars are illegal. Describe a typical morning commute in 250 words."),
    ("translate", "请把下面这段中文翻译成英文，保持专业风格：「该公司在过去三年中通过持续的研发投入，逐步建立起在功率半导体领域的技术壁垒。」"),
    ("translate", "Translate the following English passage into fluent Chinese: 'Although the macroeconomic headwinds persist, leading manufacturers continue to invest in R&D to capture demand from the energy transition.'"),
]


def _self_replay_one(
    client,
    model: str,
    category: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Optional[dict]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return None
        # PT text = prompt + response, separated by a soft break.
        # Including the prompt preserves a tiny bit of instruction-following
        # signal; loss is computed over the whole sequence (no mask) so the
        # model also sees the prompt tokens as natural text.
        text = f"{prompt}\n\n{content}"
        return {
            "text": text,
            "prompt": prompt,
            "category": category,
            "model": model,
            "source": "self-replay",
        }
    except Exception as e:
        return {"_error": str(e)}


def run_self_replay(args) -> int:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("[error] DASHSCOPE_API_KEY not set. Self-replay needs DashScope access.",
              file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("[error] openai SDK not installed. pip install openai", file=sys.stderr)
        return 2

    base_url = os.getenv(
        "DEEPSCOUT_TEACHER_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # Use the SAME model we're going to LoRA-tune so replay matches the
    # base distribution. Override via DEEPSCOUT_REPLAY_MODEL if needed.
    model = os.getenv("DEEPSCOUT_REPLAY_MODEL", "qwen2.5-7b-instruct")
    client = OpenAI(api_key=api_key, base_url=base_url)

    output_path = Path(args.output or DEFAULT_OUTPUT_SELF_REPLAY).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip prompts already written (by fingerprint of text)
    seen: set[str] = set()
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    seen.add(fingerprint(row.get("text", "")))
                except Exception:
                    continue
        print(f"[info] resume mode: {len(seen)} existing samples in {output_path.name}")

    # Build the work queue: round-robin over seeds, with temperature jitter.
    rng = random.Random(args.seed)
    seeds = list(SELF_REPLAY_SEEDS)
    rng.shuffle(seeds)

    target = args.target
    already = len(seen)
    remaining = max(0, target - already)
    if remaining == 0:
        print(f"[ok] target {target} already met. Nothing to do.")
        return 0

    print(f"[info] self-replay target={target} existing={already} need={remaining}")
    print(f"[info] model={model} base_url={base_url}")
    print(f"[info] concurrency={args.concurrency} temp={args.temperature}+/-0.15")

    # Generate (category, prompt, temperature) tuples lazily
    def gen_jobs():
        i = 0
        while True:
            cat, p = seeds[i % len(seeds)]
            t = max(0.3, min(1.2, args.temperature + rng.uniform(-0.15, 0.15)))
            yield cat, p, t
            i += 1

    jobs = gen_jobs()
    write_lock = threading.Lock()
    stats = Counter()
    written = already
    t0 = time.time()

    def submit_batch(executor, n):
        futs = []
        for _ in range(n):
            cat, p, t = next(jobs)
            futs.append(executor.submit(
                _self_replay_one,
                client, model, cat, p, t, args.max_tokens,
            ))
        return futs

    with output_path.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        # Keep a sliding window of in-flight requests until we hit the target.
        in_flight = submit_batch(ex, args.concurrency * 2)
        while written < target and in_flight:
            done, pending = [], []
            for f in in_flight:
                if f.done():
                    done.append(f)
                else:
                    pending.append(f)
            if not done:
                # avoid busy loop
                time.sleep(0.05)
                in_flight = pending + in_flight[len(pending):]
                continue

            for f in done:
                row = f.result()
                stats["api_calls"] += 1
                if not row:
                    stats["empty_response"] += 1
                    continue
                if "_error" in row:
                    stats["api_error"] += 1
                    if stats["api_error"] <= 5:
                        print(f"[warn] api error: {row['_error']}", file=sys.stderr)
                    continue
                if len(row["text"]) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                fp = fingerprint(row["text"])
                if fp in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(fp)
                with write_lock:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    stats["written"] += 1
                    if written % 25 == 0 or written == target:
                        rate = (written - already) / max(1e-6, time.time() - t0)
                        print(f"  [{written}/{target}]  {rate:.2f} rows/s  "
                              f"err={stats['api_error']} dup={stats['duplicate']}")

            # Top up the queue
            need = (args.concurrency * 2) - len(pending)
            if written < target and need > 0:
                pending.extend(submit_batch(ex, need))
            in_flight = pending

        # Drain remaining futures politely
        for f in in_flight:
            f.cancel()

    elapsed = time.time() - t0
    print(f"[ok] self-replay → {output_path}")
    print(f"     wrote {written - already} new rows in {elapsed:.1f}s "
          f"(total in file: {written})")
    for k, v in stats.most_common():
        print(f"     {k:20s} = {v}")
    return 0 if written > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["domain", "self-replay"],
        default="domain",
        help="PT data source. 'domain' = deep_read web bodies; "
             "'self-replay' = generate via DashScope qwen2.5-7b-instruct.",
    )
    ap.add_argument(
        "--log-dir",
        default=os.getenv("DEEPSCOUT_LOG_DIR", str(DEFAULT_LOG_DIR)),
        help="[domain] Directory holding llm_calls_*.jsonl",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default depends on --source.",
    )
    ap.add_argument("--min-chars", type=int, default=500,
                    help="Drop samples shorter than this (default 500)")
    ap.add_argument("--max-chars", type=int, default=32000,
                    help="[domain] Hard truncate documents longer than this")
    # self-replay-only
    ap.add_argument("--target", type=int, default=1500,
                    help="[self-replay] Total rows to produce (default 1500)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="[self-replay] Parallel API calls (default 4)")
    ap.add_argument("--temperature", type=float, default=0.85,
                    help="[self-replay] Base sampling temperature (default 0.85)")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="[self-replay] Max output tokens per call (default 1024)")
    ap.add_argument("--seed", type=int, default=42,
                    help="[self-replay] RNG seed for prompt order/temperature jitter")
    ap.add_argument("--overwrite", action="store_true",
                    help="[self-replay] Overwrite output file instead of resuming")
    args = ap.parse_args()

    if args.source == "self-replay":
        return run_self_replay(args)

    # ---------- domain ----------
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_path = Path(args.output or DEFAULT_OUTPUT_DOMAIN).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen: set[str] = set()
    written = 0
    total_chars = 0

    with output_path.open("w", encoding="utf-8") as out:
        for rec in iter_log_records(log_dir):
            stats["records"] += 1
            if not is_deep_read(rec):
                stats["skip_not_deep_read"] += 1
                continue
            up = rec.get("user_prompt") or ""
            extracted = extract_body(up)
            if not extracted:
                stats["skip_no_content_section"] += 1
                continue

            text = extracted["text"]
            if len(text) < args.min_chars:
                stats["skip_too_short"] += 1
                continue
            if len(text) > args.max_chars:
                text = text[: args.max_chars]
                stats["truncated"] += 1

            fp = fingerprint(text)
            if fp in seen:
                stats["skip_duplicate"] += 1
                continue
            seen.add(fp)

            row = {
                "text": text,
                "url": extracted["url"],
                "title": extracted["title"],
                "source": "deep_read",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            total_chars += len(text)

    stats["written"] = written
    stats["total_chars"] = total_chars
    stats["avg_chars"] = total_chars // written if written else 0

    print(f"[ok] PT domain data → {output_path}")
    print(f"     log_dir = {log_dir}")
    for k, v in stats.most_common():
        print(f"     {k:24s} = {v}")

    if written == 0:
        print(
            "[warn] no PT samples produced. Run real deep_research first to "
            "populate llm_calls_*.jsonl, or relax --min-chars.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
