#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct LoRA Fine-tuning

使用 unsloth 加速训练，支持 Qwen2.5-7B-Instruct

Requirements:
    pip install unsloth transformers datasets trl peft torch

Usage:
    python scripts/train_lora.py
"""

import torch
from pathlib import Path


def main():
    # 配置
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    data_file = "/tmp/deepscout_training_data/training_data.jsonl"
    output_dir = "./qwen2.5-7b-deepscout-lora"
    max_seq_length = 8192

    print("=" * 60)
    print("Qwen2.5-7B-Instruct LoRA Fine-tuning")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 检查数据文件
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"Error: Training data not found at {data_file}")
        print("Please run export_training_data.py first")
        return

    # 导入 unsloth
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: unsloth not installed.")
        print("Please install with: pip install unsloth")
        return

    print(f"\n1. Loading model: {model_name}")

    # 1. 加载模型 (4bit 量化)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        trust_remote_code=True
    )

    print(f"   Model loaded: {model.num_parameters() / 1e9:.1f}B parameters")

    # 2. 添加 LoRA adapter
    print("\n2. Configuring LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.05
    )

    print("   LoRA adapters added")

    # 3. 加载训练数据
    print("\n3. Loading training data...")

    from datasets import load_dataset

    def format_data(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"   Loaded {len(dataset)} samples")

    dataset = dataset.map(format_data, remove_columns=["messages", "agent", "model"])

    # 4. 训练配置
    print("\n4. Configuring training...")

    from trl import SFTTrainingArguments, SFTTrainer

    training_args = SFTTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        optim="adamw_torch",
        report_to="none",
    )

    print(f"   Training config: batch_size=2, gradient_accumulation=4, max_steps=500")
    print(f"   Learning rate: 2e-4")
    print(f"   Precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")

    # 5. 开始训练
    print("\n5. Starting training...")
    print("   (This may take a while on large models)")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    trainer.train()

    # 6. 保存 adapter
    print("\n6. Saving model...")
    final_dir = Path(output_dir) / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Model saved to: {final_dir}")
    print(f"{'=' * 60}")
    print("\nTo use the fine-tuned model:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained(")
    print(f"      model_name='{final_dir}'")
    print(f"  )")
    print(f"  FastLanguageModel.for_inference(model)")


if __name__ == "__main__":
    main()
