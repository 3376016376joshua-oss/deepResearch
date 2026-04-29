#!/usr/bin/env python3
"""
DeepScout LoRA adapter 离线评估

读取 export_training_data.py 产出的 eval 集，跑本地 base+adapter 推理，
对每一条样本与 teacher 输出做对比，输出关键指标：

    - JSON 合法率（上线门槛 ≥ 98%）
    - 必填字段命中率（按 prompt_type 分类）
    - extracted_facts 数量分布
    - credibility_score 与 teacher 的 MAE
    - 平均生成时长

输出：
    backend/data/deepscout_logs/eval_report.json
    backend/data/deepscout_logs/eval_predictions.jsonl

Usage:
    python scripts/eval_lora.py
    DEEPSCOUT_EVAL_LIMIT=20 python scripts/eval_lora.py    # 仅评估前 20 条做冒烟
"""

import os
import re
import json
import time
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "deepscout_logs"

EVAL_FILE = Path(os.getenv("DEEPSCOUT_EVAL_FILE", str(DEFAULT_LOG_DIR / "training_data.eval.jsonl")))
REPORT_FILE = Path(os.getenv("DEEPSCOUT_EVAL_REPORT", str(DEFAULT_LOG_DIR / "eval_report.json")))
PRED_FILE = Path(os.getenv("DEEPSCOUT_EVAL_PRED", str(DEFAULT_LOG_DIR / "eval_predictions.jsonl")))
EVAL_LIMIT = int(os.getenv("DEEPSCOUT_EVAL_LIMIT", "0"))  # 0 = 全量
MAX_NEW_TOKENS = int(os.getenv("DEEPSCOUT_LOCAL_MAX_NEW_TOKENS", "2048"))

JSON_VALID_THRESHOLD = 0.98

REQUIRED_FIELDS = {
    "search_analysis": ["extracted_facts", "key_insights"],
    "deep_read": ["summary", "key_facts"],
    "supplementary": ["extracted_facts"],
    "deep_search": ["extracted_facts"],
    "unknown": [],
}


def parse_json_loose(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return None
    return None


def fact_credibility_scores(parsed: dict) -> List[float]:
    facts = parsed.get("extracted_facts") if isinstance(parsed, dict) else None
    if not isinstance(facts, list):
        return []
    out = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        s = f.get("credibility_score")
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if 0.0 <= v <= 1.0:
            out.append(v)
    return out


def load_eval_dataset(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    if limit > 0:
        items = items[:limit]
    return items


def build_local_pipeline():
    """复用 base.py 的加载逻辑，避免重复代码。"""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "app"))
    import logging
    logger = logging.getLogger("eval_lora")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from service.deep_research_v2.agents.base import BaseAgent

    bundle = BaseAgent._load_deepscout_local_model(logger)
    return bundle["model"], bundle["tokenizer"]


def generate_response(model, tokenizer, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    """调用本地 base+adapter 生成响应，返回 (text, latency_ms)。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + "\n\n请仅输出一个合法的JSON对象，不要输出任何额外说明。"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(model, "device"):
        input_ids = input_ids.to(model.device)

    t0 = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    latency = int((time.time() - t0) * 1000)
    gen = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, latency


def main():
    if not EVAL_FILE.exists():
        print(f"Error: eval file not found at {EVAL_FILE}")
        print("Run scripts/export_training_data.py first.")
        return

    samples = load_eval_dataset(EVAL_FILE, EVAL_LIMIT)
    if not samples:
        print(f"Error: no samples in {EVAL_FILE}")
        return

    print(f"Loaded {len(samples)} eval samples from {EVAL_FILE}")
    print("Loading local base+adapter model...")
    model, tokenizer = build_local_pipeline()

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    json_valid = 0
    type_counts = Counter()
    type_required_hits = defaultdict(lambda: [0, 0])  # ptype -> [hits, total]
    fact_counts: List[int] = []
    cred_diffs: List[float] = []
    latencies: List[int] = []

    pred_fp = open(PRED_FILE, "w", encoding="utf-8")

    for idx, sample in enumerate(samples, 1):
        msgs = sample.get("messages", [])
        if len(msgs) < 3:
            continue
        ptype = sample.get("prompt_type", "unknown")
        type_counts[ptype] += 1
        sys_p = msgs[0].get("content", "")
        usr_p = msgs[1].get("content", "")
        teacher_text = msgs[2].get("content", "")
        teacher_parsed = parse_json_loose(teacher_text) or {}

        try:
            pred_text, latency_ms = generate_response(model, tokenizer, sys_p, usr_p)
        except Exception as e:
            print(f"  [{idx}] generation failed: {e}")
            continue
        latencies.append(latency_ms)

        parsed = parse_json_loose(pred_text)
        if isinstance(parsed, dict):
            json_valid += 1

            required = REQUIRED_FIELDS.get(ptype, [])
            if required:
                hits, total = type_required_hits[ptype]
                if all(k in parsed for k in required):
                    hits += 1
                total += 1
                type_required_hits[ptype] = [hits, total]

            facts = parsed.get("extracted_facts")
            if isinstance(facts, list):
                fact_counts.append(len(facts))

            pred_scores = fact_credibility_scores(parsed)
            teacher_scores = fact_credibility_scores(teacher_parsed)
            n = min(len(pred_scores), len(teacher_scores))
            for i in range(n):
                cred_diffs.append(abs(pred_scores[i] - teacher_scores[i]))

        pred_fp.write(json.dumps({
            "idx": idx,
            "prompt_type": ptype,
            "teacher": teacher_text,
            "prediction": pred_text,
            "latency_ms": latency_ms,
            "json_valid": isinstance(parsed, dict),
        }, ensure_ascii=False) + "\n")
        pred_fp.flush()

        if idx % 10 == 0 or idx == len(samples):
            rate = json_valid / idx
            print(f"  [{idx}/{len(samples)}] json_valid_rate={rate:.3f}, avg_latency={statistics.mean(latencies):.0f}ms")

    pred_fp.close()

    n = len(samples)
    json_valid_rate = json_valid / n if n else 0.0

    required_hit_rates = {}
    for ptype, (hits, total) in type_required_hits.items():
        required_hit_rates[ptype] = (hits / total) if total else None

    report = {
        "eval_file": str(EVAL_FILE),
        "n_samples": n,
        "json_valid": json_valid,
        "json_valid_rate": round(json_valid_rate, 4),
        "passes_threshold": json_valid_rate >= JSON_VALID_THRESHOLD,
        "json_valid_threshold": JSON_VALID_THRESHOLD,
        "type_distribution": dict(type_counts),
        "required_field_hit_rate": required_hit_rates,
        "extracted_facts_count": {
            "n": len(fact_counts),
            "mean": round(statistics.mean(fact_counts), 2) if fact_counts else None,
            "median": statistics.median(fact_counts) if fact_counts else None,
            "min": min(fact_counts) if fact_counts else None,
            "max": max(fact_counts) if fact_counts else None,
        },
        "credibility_mae": {
            "n_pairs": len(cred_diffs),
            "mae": round(statistics.mean(cred_diffs), 4) if cred_diffs else None,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 1) if len(latencies) >= 20 else None,
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"Report: {REPORT_FILE}")
    print(f"Predictions: {PRED_FILE}")
    if not report["passes_threshold"]:
        print(f"WARNING: json_valid_rate={json_valid_rate:.3f} < {JSON_VALID_THRESHOLD}, 不达上线门槛")


if __name__ == "__main__":
    main()
