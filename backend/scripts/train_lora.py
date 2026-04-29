#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct LoRA Fine-tuning for DeepScout

⚠️ This script is **GPU-only** and is intended to run on a cloud platform
(runpod / autodl / colab / 阿里云 PAI / lambda labs ...). The local dev
machine is assumed to have no GPU — it only runs inference with the
adapter downloaded from the cloud.

Cloud workflow:
    1. Upload `training_data.jsonl` from local
       (produced by scripts/export_training_data.py).
    2. On the cloud machine:
        pip install "unsloth>=2024.8" "trl>=0.9.0" "transformers>=4.43" \\
                    datasets peft torch
        export DEEPSCOUT_TRAIN_DATA=/workspace/training_data.jsonl
        export DEEPSCOUT_SFT_PATH=/workspace/deepscout_adapter
        python scripts/train_lora.py
    3. Download the adapter dir back to local
       backend/app/service/deep_research_v2/sft/  (auto-loaded by base.py).

Repo paths are only used as defaults — both DEEPSCOUT_TRAIN_DATA and
DEEPSCOUT_SFT_PATH accept any absolute path, so this script also runs
fine when the repo is not checked out on the cloud machine (just copy
this file + the data file).
"""

import os
from pathlib import Path

# ============ 离线 / 反 telemetry ============
# unsloth 启动时会向 huggingface.co 发匿名统计请求，国内机器（autodl/阿里云）
# 经常 120s 超时直接把进程干掉。模型已在本地时这步纯粹是浪费——直接关掉。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# 兜底 monkey-patch：即使上面环境变量不被新版 unsloth 识别，也强制把
# get_statistics 替换成 no-op，避免 _get_statistics 触发 TimeoutError。
def _disable_unsloth_telemetry():
    try:
        from unsloth.models import _utils as _unsloth_utils
        _unsloth_utils.get_statistics = lambda *a, **kw: None
        _unsloth_utils._get_statistics = lambda *a, **kw: None
    except Exception:
        pass

_disable_unsloth_telemetry()


# ============ 路径配置 ============
# 兼容两种部署布局，自动按优先级查找数据文件：
#   1. 云端扁平：所有 jsonl 与 train_lora.py 同目录（autodl/runpod 常见）
#   2. 本地嵌套：backend/data/deepscout_logs/[distilled/]xxx.jsonl
# 任何路径仍可通过环境变量完全覆盖。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_ROOT = REPO_ROOT / "data" / "deepscout_logs"

# 候选搜索目录，按顺序匹配第一个存在的文件
SEARCH_DIRS = [
    SCRIPT_DIR,                       # 云端扁平布局
    LOCAL_DATA_ROOT / "distilled",    # 本地蒸馏 SFT
    LOCAL_DATA_ROOT,                  # 本地 PT
]


def _resolve_first(filename: str) -> str | None:
    """在 SEARCH_DIRS 中找第一个存在且非空的同名文件。"""
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists() and p.stat().st_size > 0:
            return str(p)
    return None


def _resolve_many(filenames: list[str]) -> list[str]:
    return [r for r in (_resolve_first(n) for n in filenames) if r]


# SFT 主数据文件名（distill_deepscout_data.py 产物）。
# distilled/ 下 deep_read / deep_search / search_analysis / supplementary_search / merged
# 都是同一批数据的不同视图，lora_train.jsonl 已是全量合并版，不要再叠加。
SFT_FILENAMES = ["lora_train.jsonl"]
# PT 续训数据（build_pt_data.py 产物），按比例混入 SFT 防灾难性遗忘
PT_FILENAMES = ["pt_data_domain.jsonl", "pt_data_self_replay.jsonl"]

# 默认输出目录：本地有 backend/app/.../sft 就写回那里（base.py 自动加载）；
# 云端扁平布局下写到脚本同级 deepscout_adapter/
LOCAL_ADAPTER_DIR = REPO_ROOT / "app" / "service" / "deep_research_v2" / "sft"
if LOCAL_DATA_ROOT.exists():
    DEFAULT_OUTPUT_DIR = LOCAL_ADAPTER_DIR
else:
    DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "deepscout_adapter"

# 默认 base model：脚本同级若已下载好就直接用，避免重复下载
_LOCAL_MODEL_DIR = SCRIPT_DIR / "Qwen2.5-7B-Instruct"
DEFAULT_BASE_MODEL = (
    str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else "Qwen/Qwen2.5-7B-Instruct"
)


def _split_paths(env_value: str) -> list[str]:
    return [p.strip() for p in env_value.split(",") if p.strip()]


def _existing_nonempty(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        pp = Path(p)
        if pp.exists() and pp.stat().st_size > 0:
            out.append(str(pp))
    return out


def main():
    model_name = os.getenv("DEEPSCOUT_BASE_MODEL", DEFAULT_BASE_MODEL)
    # SFT/PT 数据路径解析：
    #   - 兼容旧的 DEEPSCOUT_TRAIN_DATA（单文件，且默认禁用 PT 混入）
    #   - 新的 DEEPSCOUT_SFT_DATA / DEEPSCOUT_PT_DATA 为逗号分隔多文件
    legacy_single = os.getenv("DEEPSCOUT_TRAIN_DATA", "").strip()
    if legacy_single:
        sft_files = _split_paths(legacy_single)
        default_pt_ratio = "0.0"
    else:
        sft_env = os.getenv("DEEPSCOUT_SFT_DATA", "").strip()
        sft_files = _split_paths(sft_env) if sft_env else _resolve_many(SFT_FILENAMES)
        default_pt_ratio = "0.2"
    pt_env = os.getenv("DEEPSCOUT_PT_DATA", "").strip()
    pt_files = _split_paths(pt_env) if pt_env else _resolve_many(PT_FILENAMES)
    pt_ratio = float(os.getenv("DEEPSCOUT_PT_RATIO", default_pt_ratio))
    output_dir = Path(os.getenv("DEEPSCOUT_SFT_PATH", str(DEFAULT_OUTPUT_DIR)))
    max_seq_length = int(os.getenv("DEEPSCOUT_MAX_SEQ_LEN", "4096"))
    num_epochs = float(os.getenv("DEEPSCOUT_EPOCHS", "2"))
    lr = float(os.getenv("DEEPSCOUT_LR", "1e-4"))
    eval_ratio = float(os.getenv("DEEPSCOUT_EVAL_RATIO", "0.1"))
    batch_size = int(os.getenv("DEEPSCOUT_BATCH_SIZE", "2"))
    grad_accum = int(os.getenv("DEEPSCOUT_GRAD_ACCUM", "4"))
    use_packing = os.getenv("DEEPSCOUT_PACKING", "1") == "1"
    lora_r = int(os.getenv("DEEPSCOUT_LORA_R", "16"))
    lora_alpha = int(os.getenv("DEEPSCOUT_LORA_ALPHA", str(lora_r * 2)))

    print("=" * 60)
    print("Qwen2.5-7B-Instruct LoRA Fine-tuning (DeepScout)")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("Error: torch not installed. Install a GPU build of PyTorch on the cloud machine.")
        return

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        print("This script is GPU-only and must run on a cloud GPU instance.")
        print("Local (no-GPU) machines should only run inference; see SFT_PLAN.md §Phase 4.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    sft_files = _existing_nonempty(sft_files)
    pt_files = _existing_nonempty(pt_files)
    if not sft_files:
        print("Error: no non-empty SFT data files found.")
        print("  Expected one of: distilled/lora_train.jsonl, training_data.jsonl")
        print("  Or set DEEPSCOUT_SFT_DATA / DEEPSCOUT_TRAIN_DATA explicitly.")
        return
    if pt_ratio <= 0:
        pt_files = []

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: unsloth not installed. pip install unsloth")
        return

    # ---------- 1. 加载模型 ----------
    print(f"\n[1/6] Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Model loaded: {model.num_parameters() / 1e9:.1f}B params")

    # ---------- 2. LoRA ----------
    print("\n[2/6] Adding LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ---------- 3. 数据 ----------
    print("\n[3/6] Loading dataset")
    print(f"   SFT files ({len(sft_files)}):")
    for p in sft_files:
        print(f"     - {p}")
    if pt_files:
        print(f"   PT files ({len(pt_files)}), mix ratio={pt_ratio}:")
        for p in pt_files:
            print(f"     - {p}")
    else:
        print(f"   PT files: none (ratio={pt_ratio})")

    from datasets import load_dataset, interleave_datasets

    def format_sft(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    sft_raw = load_dataset("json", data_files=sft_files, split="train")
    print(f"   SFT raw samples: {len(sft_raw)}")
    sft_drop = [c for c in sft_raw.column_names if c != "text"]
    sft_ds = sft_raw.map(format_sft, remove_columns=sft_drop)

    pt_ds = None
    if pt_files:
        pt_raw = load_dataset("json", data_files=pt_files, split="train")
        print(f"   PT raw samples: {len(pt_raw)}")
        pt_drop = [c for c in pt_raw.column_names if c != "text"]
        # PT 已经是 {"text": ...}，只需要去掉多余列
        pt_ds = pt_raw.remove_columns(pt_drop) if pt_drop else pt_raw

    # train / eval 划分：只在 SFT 上切 eval（PT 不参与评估）
    if eval_ratio > 0 and len(sft_ds) >= 20:
        split = sft_ds.train_test_split(test_size=eval_ratio, seed=42)
        sft_train, eval_ds = split["train"], split["test"]
        print(f"   SFT train: {len(sft_train)} | Eval: {len(eval_ds)}")
    else:
        sft_train, eval_ds = sft_ds, None
        print(f"   SFT train: {len(sft_train)} | Eval: disabled")

    # PT 按比例混入 train 集
    if pt_ds is not None and len(pt_ds) > 0 and pt_ratio > 0:
        train_ds = interleave_datasets(
            [sft_train, pt_ds],
            probabilities=[1.0 - pt_ratio, pt_ratio],
            seed=42,
            stopping_strategy="all_exhausted",
        )
        print(f"   Train (SFT+PT interleaved): {len(train_ds)} "
              f"(SFT={len(sft_train)}, PT={len(pt_ds)}, pt_ratio={pt_ratio})")
    else:
        train_ds = sft_train
        print(f"   Train: {len(train_ds)} (SFT only)")

    # ---------- 4. 训练配置 ----------
    print("\n[4/6] Building trainer")
    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=str(output_dir),  # 训练产物输出目录（checkpoint、最终 adapter）
        per_device_train_batch_size=batch_size,  # 每张卡单步训练 batch size（不含梯度累积）
        per_device_eval_batch_size=batch_size,  # 每张卡评估 batch size
        gradient_accumulation_steps=grad_accum,  # 梯度累积步数；等效总 batch 会乘以该值
        warmup_ratio=0.03,  # 学习率预热比例（总训练步数的 3%）
        num_train_epochs=num_epochs,  # 训练轮数（epoch）
        learning_rate=lr,  # 峰值学习率
        fp16=not torch.cuda.is_bf16_supported(),  # 若不支持 bf16，则启用 fp16 混合精度
        bf16=torch.cuda.is_bf16_supported(),  # 若硬件支持，则优先使用 bf16
        logging_steps=10,  # 每 10 步打印一次训练日志
        save_strategy="steps",  # 按训练步数保存 checkpoint（非按 epoch）
        save_steps=100,  # 每 100 步保存一次 checkpoint
        save_total_limit=3,  # 最多保留 3 个 checkpoint，旧的自动清理
        eval_strategy="steps" if eval_ds is not None else "no",  # 有 eval 集才按步评估，否则关闭评估
        eval_steps=50 if eval_ds is not None else None,  # 开启评估时每 50 步评估一次
        optim="adamw_torch",  # 优化器实现：PyTorch 原生 AdamW
        lr_scheduler_type="cosine",  # 学习率调度：余弦退火
        weight_decay=0.01,  # 权重衰减（L2 正则）
        report_to="none",  # 不上报到 wandb / tensorboard 等外部平台
        # SFT-specific
        max_seq_length=max_seq_length,  # 单条样本最大 token 长度（超出会截断）
        packing=use_packing,  # 将多条短样本打包进同一序列，提高吞吐
        dataset_text_field="text",  # 数据集中用于训练的文本字段名
        neftune_noise_alpha=5.0,  # NEFTune 嵌入噪声强度，提升 SFT 泛化
        seed=42,  # 随机种子，保证切分/采样/训练过程可复现
    )

    print(f"   bsz={batch_size} x grad_accum={grad_accum} | epochs={num_epochs} | lr={lr}")
    print(f"   max_seq_length={max_seq_length} | packing={use_packing}")
    print(f"   precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
    print(f"   neftune=5.0, gradient_checkpointing=unsloth")

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )
    # trl 兼容：旧版用 tokenizer=，新版用 processing_class=
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    # ---------- 5. 训练 ----------
    print("\n[5/6] Training")
    trainer.train()

    # ---------- 6. 保存 ----------
    print("\n[6/6] Saving adapter")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "=" * 60)
    print(f"Done. Adapter saved to: {output_dir}")
    print("=" * 60)
    print("Next step (download to local dev machine):")
    print(f"  rsync -av <cloud>:{output_dir}/ "
          "<local-repo>/backend/app/service/deep_research_v2/sft/")
    print("Then base.py will auto-load it (DEEPSCOUT_USE_LOCAL_SFT=1, default on).")
    print("On a CPU-only local box, also set DEEPSCOUT_WARMUP=0 to avoid blocking startup.")


if __name__ == "__main__":
    main()
