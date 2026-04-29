# DeepScout LoRA 防灾难性遗忘 · 实现计划

> 起始日期：2026-04-25
> 背景：本机无 GPU，LoRA 训练在云端跑；本地只做推理。为防止 SFT 后基础能力退化，引入 PT replay 数据混合训练。

## 总体方案

**核心实现思路**：HF `Trainer` 的 `labels=-100` 位置默认不计入 loss，所以"PT 算全序列 loss / QA 只算 answer 部分 loss"完全等价于**预先构造每个样本的 labels**，不需要重写 `compute_loss`。

- 弃用 `SFTTrainer` / `SFTConfig`（用户已确认），直接用 `transformers.Trainer`
- 关闭 packing（packing 会跨样本拼接、破坏 label mask 边界）
- 用 `datasets.interleave_datasets([qa_ds, pt_ds], probabilities=[0.8, 0.2])` 混合

**混合比例**：QA : PT = 8 : 2，PT 内部 `通用 self-replay : 领域 deep_read ≈ 1 : 1`。

## PT 数据来源（优先级）

1. **self-replay**（最推荐）：用 base 模型 `qwen2.5-7b-instruct` 自己跑一组通用 prompt 池生成 2–5k 条文本。分布天然对齐 base，不引入新偏置。
2. **领域 deep_read 网页正文**：从 `data/deepscout_logs/llm_calls_*.jsonl` 抽 `## 文档内容` 段，免费且覆盖目标领域。
3. **公开预训练语料切片**（备选）：`Skywork/SkyPile-150B`、`wikimedia/wikipedia` zh、`allenai/c4` en。

## 阶段拆分

### Phase A · 数据准备（本地 CPU）

- **A1 · 领域 PT 抽取** ✅ 已实现
  - `backend/scripts/build_pt_data.py --source domain`
  - 扫日志 → 正则提取网页正文 → md5 指纹去重 → 输出 `pt_data_domain.jsonl`
- **A2 · self-replay** ✅ 已实现
  - `backend/scripts/build_pt_data.py --source self-replay --target 1500`
  - 通过 DashScope 调 `qwen2.5-7b-instruct`，~40 类 seed prompt（QA / 写作 / 代码 / 推理 / 摘要 / 对话 / 行业 / 翻译）
  - 多线程并发 + 温度抖动 + resume（按 fingerprint 跳过已写）
  - 输出 `pt_data_self_replay.jsonl`，每行 `{text, prompt, category, model, source}`
  - 默认目标 1500 条；环境变量：`DASHSCOPE_API_KEY`、`DEEPSCOUT_REPLAY_MODEL`、`DEEPSCOUT_TEACHER_BASE_URL`
- **A3 · QA + PT 联合 tokenize**（待做，云端跑）
  - QA：apply_chat_template，定位 assistant span（重新 tokenize prefix 取长度），prefix 部分 labels=-100
  - PT：按 max_seq_length 切片，labels = input_ids
  - 输出 HF `datasets.save_to_disk` 格式

### Phase B · 训练脚本（云端 GPU）

- **B1 · 改造 train_lora.py**（待做）
  - 弃 SFTTrainer，用 `Trainer` + `TrainingArguments`
  - `packing=False`、`group_by_length=True`、`DataCollatorForSeq2Seq` 动态 padding
  - `interleave_datasets(probabilities=[0.8, 0.2])`
  - 新环境变量：`DEEPSCOUT_PT_DATA`、`DEEPSCOUT_PT_RATIO`（默认 0.2）
- **B2 · PT loss 加权**（可选，仅当 default 效果不理想时上）
  - 自定义 `Trainer.compute_loss`，按 `sample_type` 分别求 loss 再加权

### Phase C · 评估

- **C1 · 防遗忘指标**（待做）
  - `scripts/eval_forgetting.py`：在通用 benchmark slice 上算 base / SFT-only / SFT+replay 三档 perplexity
  - 上线门槛：相对 base perplexity 退化 < 5%
- **C2 · 任务质量**：复用 `scripts/eval_lora.py`（JSON 合法率、字段命中率）

## 当前进度

- ✅ A1 实现并冒烟过
- ✅ A2 实现（self-replay）
- ⬜ A3 联合 tokenize
- ⬜ B1 训练脚本改造
- ⬜ C1 防遗忘评估

## 下一步

跑 A2 攒 PT 数据：
```bash
DASHSCOPE_API_KEY=sk-... python scripts/build_pt_data.py \
    --source self-replay --target 1500 --concurrency 4
```
然后 A3 + B1 一并改，避免训练脚本反复重写。
