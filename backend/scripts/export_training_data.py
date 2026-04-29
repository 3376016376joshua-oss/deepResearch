#!/usr/bin/env python3
"""
DeepScout 训练数据导出 + 质量过滤

流程：
    raw jsonl logs → 类型识别 → 字段校验 → 语义去重 → 分层划分 → 标准对话格式

输出：
    training_data.jsonl       (全量 train，向后兼容)
    training_data.train.jsonl (90%)
    training_data.eval.jsonl  (10%, 按 prompt 类型分层)
    training_data.stats.json  (过滤统计)

Usage:
    python scripts/export_training_data.py
"""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "deepscout_logs"
DATA_DIR = Path(os.getenv("DEEPSCOUT_LOG_DIR", str(DEFAULT_LOG_DIR)))
OUTPUT_FILE = DATA_DIR / "training_data.jsonl"
TRAIN_FILE = DATA_DIR / "training_data.train.jsonl"
EVAL_FILE = DATA_DIR / "training_data.eval.jsonl"
STATS_FILE = DATA_DIR / "training_data.stats.json"

EVAL_RATIO = float(os.getenv("DEEPSCOUT_EVAL_RATIO", "0.1"))
MAX_RESPONSE_CHARS = int(os.getenv("DEEPSCOUT_MAX_RESPONSE_CHARS", "32000"))  # ~8k tokens
MIN_RESPONSE_CHARS = 50
RANDOM_SEED = 42

SOURCE_TYPE_WHITELIST = {
    "official", "academic", "news", "report", "self_media", "industry"
}

# ============ 类型识别 ============

TYPE_SEARCH_ANALYSIS = "search_analysis"
TYPE_DEEP_READ = "deep_read"
TYPE_SUPPLEMENTARY = "supplementary"
TYPE_DEEP_SEARCH = "deep_search"
TYPE_UNKNOWN = "unknown"


def detect_prompt_type(user_prompt: str) -> str:
    """通过 user_prompt 关键词识别 4 类 scout 调用。"""
    p = user_prompt
    if "深度阅读文档" in p or "## 文档内容" in p:
        return TYPE_DEEP_READ
    if "补充搜索以解决审核发现" in p or "补充搜索关键词" in p:
        return TYPE_SUPPLEMENTARY
    if "追溯原始数据源" in p or "追踪相关线索" in p or "further_tracing_queries" in p:
        return TYPE_DEEP_SEARCH
    if "研究假设" in p and "搜索结果" in p and "extracted_facts" in p:
        return TYPE_SEARCH_ANALYSIS
    return TYPE_UNKNOWN


# ============ 响应校验 ============

REQUIRED_FIELDS = {
    TYPE_SEARCH_ANALYSIS: ["extracted_facts", "key_insights"],
    TYPE_DEEP_READ: ["summary", "key_facts"],
    TYPE_SUPPLEMENTARY: ["extracted_facts"],
    TYPE_DEEP_SEARCH: ["extracted_facts"],
    TYPE_UNKNOWN: [],
}


def parse_json_loose(text: str) -> Optional[dict]:
    """宽松解析：直接 / 代码块 / 最外层 {…}。"""
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


def validate_facts(facts) -> bool:
    """校验 extracted_facts 列表语义合法。"""
    if not isinstance(facts, list) or not facts:
        return False
    for f in facts:
        if not isinstance(f, dict):
            return False
        if not f.get("content"):
            return False
        score = f.get("credibility_score")
        if score is not None:
            try:
                if not (0.0 <= float(score) <= 1.0):
                    return False
            except (TypeError, ValueError):
                return False
        stype = f.get("source_type")
        if stype and stype not in SOURCE_TYPE_WHITELIST:
            return False
    return True


def validate_response(prompt_type: str, response: str) -> tuple[bool, str]:
    """返回 (是否通过, 失败原因)。"""
    if not response or len(response) < MIN_RESPONSE_CHARS:
        return False, "too_short"
    if len(response) > MAX_RESPONSE_CHARS:
        return False, "too_long"

    parsed = parse_json_loose(response)
    if parsed is None:
        return False, "json_invalid"
    if not isinstance(parsed, dict):
        return False, "json_not_object"

    for field in REQUIRED_FIELDS.get(prompt_type, []):
        if field not in parsed:
            return False, f"missing_field:{field}"

    # 对包含 extracted_facts 的类型做深层校验
    if "extracted_facts" in REQUIRED_FIELDS.get(prompt_type, []):
        if not validate_facts(parsed.get("extracted_facts")):
            return False, "invalid_facts"

    return True, "ok"


# ============ 语义去重 ============

def semantic_fingerprint(text: str) -> str:
    """复用 scout._compute_fact_fingerprint 思路：数字 + 中文关键词。"""
    numbers = re.findall(r"\d+\.?\d*", text)
    keywords = re.findall(r"[一-龥]{2,4}", text)[:10]
    sig = f"{','.join(numbers[:5])}|{','.join(keywords)}"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]


# ============ I/O ============

def load_jsonl(path: Path) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def to_training_format(item: dict, prompt_type: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_prompt"]},
            {"role": "assistant", "content": item.get("response", "")},
        ],
        "agent": item.get("agent", ""),
        "model": item.get("model", ""),
        "prompt_type": prompt_type,
    }


def stratified_split(records: list, eval_ratio: float, seed: int) -> tuple[list, list]:
    """按 prompt_type 分层划分。"""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["prompt_type"]].append(r)

    train, eval_ = [], []
    for ptype, items in by_type.items():
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_ratio)) if len(items) >= 10 else 0
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


# ============ 主流程 ============

def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist")
        return

    jsonl_files = [
        f for f in DATA_DIR.glob("*.jsonl")
        if f.name not in {"training_data.jsonl", "training_data.train.jsonl", "training_data.eval.jsonl"}
    ]
    if not jsonl_files:
        print(f"No source JSONL files in {DATA_DIR}")
        return

    print(f"Found {len(jsonl_files)} source file(s)")

    raw = []
    for jf in jsonl_files:
        items = load_jsonl(jf)
        print(f"  {jf.name}: {len(items)}")
        raw.extend(items)

    stats = Counter()
    stats["raw"] = len(raw)
    print(f"\nTotal raw: {len(raw)}")

    # 1. 必备字段 + 错误过滤
    stage1 = []
    for it in raw:
        if "error" in it:
            stats["drop_error"] += 1
            continue
        if not it.get("system_prompt") or not it.get("user_prompt") or not it.get("response"):
            stats["drop_missing_field"] += 1
            continue
        # 仅训练 DeepScout (DataAnalyst 单独训)
        if it.get("agent") and it["agent"] != "DeepScout":
            stats["drop_other_agent"] += 1
            continue
        stage1.append(it)
    print(f"After basic filter: {len(stage1)}")

    # 2. 类型识别 + 响应校验
    stage2 = []
    type_counts = Counter()
    for it in stage1:
        ptype = detect_prompt_type(it["user_prompt"])
        type_counts[ptype] += 1
        if ptype == TYPE_UNKNOWN:
            stats["drop_unknown_type"] += 1
            continue
        ok, reason = validate_response(ptype, it["response"])
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue
        it["_prompt_type"] = ptype
        stage2.append(it)
    print(f"After response validation: {len(stage2)}")
    print(f"Type distribution (raw): {dict(type_counts)}")

    # 3. 语义去重（基于 user_prompt 指纹）
    seen_fp = {}
    stage3 = []
    for it in stage2:
        fp = semantic_fingerprint(it["user_prompt"])
        key = f"{it['_prompt_type']}|{fp}"
        if key in seen_fp:
            stats["drop_semantic_dup"] += 1
            continue
        seen_fp[key] = True
        stage3.append(it)
    print(f"After semantic dedup: {len(stage3)}")

    if not stage3:
        print("No samples remaining after filtering. Aborting.")
        return

    # 4. 转训练格式
    records = [to_training_format(it, it["_prompt_type"]) for it in stage3]

    # 5. 分层 train/eval 划分
    train, evals = stratified_split(records, EVAL_RATIO, RANDOM_SEED)
    print(f"\nFinal: train={len(train)}, eval={len(evals)}")

    final_type_dist = Counter(r["prompt_type"] for r in records)
    print(f"Final type distribution: {dict(final_type_dist)}")

    # 6. 写出
    def dump(records_, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records_:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(records, OUTPUT_FILE)  # 兼容旧 train_lora.py 默认路径
    dump(train, TRAIN_FILE)
    dump(evals, EVAL_FILE)

    stats_payload = {
        "raw_total": stats["raw"],
        "filtered_kept": len(records),
        "train": len(train),
        "eval": len(evals),
        "drops": {k: v for k, v in stats.items() if k.startswith("drop_")},
        "type_distribution_raw": dict(type_counts),
        "type_distribution_final": dict(final_type_dist),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    print(f"\nWrote:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {TRAIN_FILE}")
    print(f"  {EVAL_FILE}")
    print(f"  {STATS_FILE}")


if __name__ == "__main__":
    main()
