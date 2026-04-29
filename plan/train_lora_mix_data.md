# train_lora.py 接入全部生成数据

## 当前数据现状（`backend/data/deepscout_logs/`）

| 文件 | 行数 | 格式 | 用途 |
|---|---|---|---|
| `training_data.jsonl` | 不存在 | — | 旧默认路径，已失效 |
| `distilled/lora_train.jsonl` | 1200 | `{"messages":[...]}` | 蒸馏后的 SFT 主数据 |
| `pt_data_self_replay.jsonl` | 1500 | `{"text":..., "prompt":..., "category":...}` | 自回放 PT，防遗忘 |
| `pt_data_domain.jsonl` | 0（空文件） | `{"text":...}` | 域内 PT，等真实 deep_read log |

## 改动目标

1. 把 SFT 默认源换成 `distilled/lora_train.jsonl`。
2. 自动挂载 PT 数据 (`pt_data_*.jsonl`)，按 ~20% 比例混入训练。
3. 兼容两种 schema：`messages` → 走 chat template；`text` → 直接当原始语料。
4. 空文件 / 不存在文件静默跳过；都不存在则报错。
5. 保留环境变量覆盖，支持云端自定义路径。

## 实现要点

- 新增环境变量：
  - `DEEPSCOUT_SFT_DATA`：逗号分隔多文件，默认 `data/deepscout_logs/distilled/lora_train.jsonl`
  - `DEEPSCOUT_PT_DATA`：逗号分隔多文件，默认 `pt_data_domain.jsonl,pt_data_self_replay.jsonl`
  - `DEEPSCOUT_PT_RATIO`：PT 混入比例，默认 0.2（设 0 关闭）
  - 旧的 `DEEPSCOUT_TRAIN_DATA` 仍兼容（如果设置则等价于 SFT 单文件且禁用 PT，向后兼容旧 README）
- 数据加载：
  - SFT 文件：`load_dataset("json", data_files=[...])` → map 到 `{"text": chat_template(messages)}`
  - PT 文件：`load_dataset("json", data_files=[...])` → map 到 `{"text": row["text"]}`
  - 都只保留 `text` 一列，便于 interleave。
- 混合：
  - PT 比例 > 0 且两边都非空时用 `datasets.interleave_datasets([sft, pt], probabilities=[1-r, r], stopping_strategy="all_exhausted")`
  - 否则只用 SFT。
- 划分：保持现有 `train_test_split(eval_ratio)`，但只在 SFT 上划分 eval（PT 不参与 eval，避免污染指标），即 SFT 先切 train/eval，再把 PT interleave 进 train。

## 风险与注意

- `messages` 格式中已有 system/user/assistant 三段，apply_chat_template 后会拼成完整对话——packing=True 下损失计算覆盖整段（含 prompt token），与 PT 行为一致，符合"replay 不遮 prompt"的设计意图（见 build_pt_data.py 注释）。
- `interleave_datasets` 在 `stopping_strategy="all_exhausted"` 下，少数派会被重采样。若 PT 比 SFT 多很多（1500 vs 1200），用 `first_exhausted` 反而更早结束；这里两者量级接近，`all_exhausted` 更稳。
- eval_ratio 不再作用于 PT，eval 集只看 SFT 任务质量，符合训练目标（评估推理任务而非语料拟合）。

## 文件改动

仅 `backend/scripts/train_lora.py`。其它脚本不动。
