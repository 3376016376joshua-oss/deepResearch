#!/usr/bin/env python3
"""
导出训练数据脚本

合并 JSONL 文件，转换为标准对话格式供 Qwen fine-tuning 使用

Usage:
    python scripts/export_training_data.py
"""

import json
from pathlib import Path


def load_jsonl(file_path: Path) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def deduplicate(data: list) -> list:
    """基于 system_prompt + user_prompt 去重"""
    seen = set()
    result = []
    for item in data:
        key = item["system_prompt"] + "||" + item["user_prompt"]
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def convert_to_training_format(item: dict) -> dict:
    """转换为标准对话格式"""
    return {
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_prompt"]},
            {"role": "assistant", "content": item.get("response", "")}
        ],
        "agent": item["agent"],
        "model": item["model"]
    }


def main():
    data_dir = Path("/tmp/deepscout_training_data")
    output_file = data_dir / "training_data.jsonl"

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("Please run the research flow first to collect training data")
        return

    jsonl_files = list(data_dir.glob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if f.name != "training_data.jsonl"]

    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files")

    all_data = []
    for jsonl_file in jsonl_files:
        file_data = load_jsonl(jsonl_file)
        print(f"  {jsonl_file.name}: {len(file_data)} records")
        all_data.extend(file_data)

    print(f"\nTotal: {len(all_data)} records before dedup")

    # 去重
    all_data = deduplicate(all_data)
    print(f"After dedup: {len(all_data)} records")

    # 过滤掉有错误的记录
    valid_data = [item for item in all_data if "error" not in item and "response" in item]
    print(f"After filtering errors: {len(valid_data)} records")

    # 转换为训练格式并保存
    with open(output_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(convert_to_training_format(item), ensure_ascii=False) + "\n")

    print(f"\nTraining data saved to {output_file}")
    print(f"Total training samples: {len(valid_data)}")


if __name__ == "__main__":
    main()
