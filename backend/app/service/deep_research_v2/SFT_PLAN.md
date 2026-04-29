# DeepScout Agent SFT 计划

> 目标：对 `agents/scout.py` 中的 `DeepScout` 进行 LoRA 微调，使本地 Qwen2.5-7B-Instruct + adapter 能替代远程 qwen-plus 完成搜索结果分析、深度阅读、补充搜索、递归追溯四类任务。
>
> **运行模式（本机无 GPU）**：
> - **数据准备**：本地（`distill_deepscout_data.py` / `export_training_data.py`，CPU 即可）
> - **LoRA 微调**：**云平台 GPU**（runpod / autodl / colab / 阿里云 PAI 等），跑 `train_lora.py`
> - **推理**：**本地纯 CPU/小显存**，加载 base 权重 + 下载回来的 adapter，由 `agents/base.py` 自动接管
>
> 本机不需要安装 unsloth / bitsandbytes / 任何 GPU-only 依赖，只需 `transformers + peft + torch`（CPU build）。

---

## 进度图例
- ✅ 已完成
- 🟡 部分完成 / 待验证
- ⬜ 未开始

---

## 一、当前状态盘点

### 数据采集（已有）
- `agents/base.py:24` `TARGET_TRAINING_AGENTS = {"DeepScout", "DataAnalyst"}`
- `agents/base.py:_save_training_log()` → `backend/data/deepscout_logs/llm_calls_YYYYMMDD.jsonl`
- 字段：`system_prompt / user_prompt / response / model / temperature / duration_ms / inference_mode`
- 失败请求也会落盘（带 `error` 字段，导出阶段过滤）

### 数据生产（已有，本地 CPU 可跑）
- `scripts/distill_deepscout_data.py`：用 qwen-max 蒸馏 4 类 prompt，目标 ~1200 条。
- `scripts/export_training_data.py`：✅ **已重写为完整质量过滤管线**
  - 类型识别（4 类 scout 调用 + unknown）
  - 字段校验（`extracted_facts` 非空、`credibility_score ∈ [0,1]`、`source_type` 白名单）
  - 长度过滤（50 < chars < 32k）
  - JSON 宽松解析校验
  - 语义去重（数字+中文关键词指纹，复用 `_compute_fact_fingerprint` 思路）
  - 按 prompt_type **分层** 90/10 train/eval 划分
  - 输出统计：`training_data.stats.json`

### 训练（云平台跑）
- `scripts/train_lora.py`：Unsloth + Qwen2.5-7B-Instruct, LoRA r=16/α=32, max_seq=8192。
- ✅ trl API 修复：`SFTConfig` 替代 `SFTTrainingArguments`，`tokenizer=` / `processing_class=` 双兼容
- ✅ eval_dataset 接入，`eval_strategy="steps"`, `eval_steps=50`
- ✅ 超参更新：`num_train_epochs=2`、`lr=1e-4`、`packing=True`、`gradient_checkpointing="unsloth"`、`neftune_noise_alpha=5.0`、cosine scheduler
- ✅ 全部超参/路径走环境变量（`DEEPSCOUT_BASE_MODEL`、`DEEPSCOUT_TRAIN_DATA`、`DEEPSCOUT_SFT_PATH`、`DEEPSCOUT_EPOCHS`、`DEEPSCOUT_LR`、`DEEPSCOUT_EVAL_RATIO`、`DEEPSCOUT_MAX_SEQ_LEN`）
- ✅ **云端无需依赖本仓库结构**：`DEEPSCOUT_TRAIN_DATA` 与 `DEEPSCOUT_SFT_PATH` 可用任意绝对路径

### 推理（本地 CPU/小显存即可）
- `agents/base.py:_call_deepscout_local_model_sync` 已实现 HF base + PEFT adapter 加载。
- 默认 adapter 路径：`backend/app/service/deep_research_v2/sft`
- 自动切换设备：CUDA 可用走 fp16+GPU，否则走 fp32+CPU（CPU 推理慢，仅适合低 QPS 场景）
- 环境变量：`DEEPSCOUT_USE_LOCAL_SFT`（默认 1）、`DEEPSCOUT_SFT_PATH`、`DEEPSCOUT_BASE_MODEL`、`DEEPSCOUT_LOCAL_MAX_NEW_TOKENS`
- 失败自动回退远程 API。
- 类级缓存避免重复加载（`_deepscout_local_bundle`）。

### Scout 的 4 类调用模式（SFT 目标分布）
| # | 方法 | Prompt | 频率 | 上下文 | 识别关键词 |
|---|---|---|---|---|---|
| 1 | `_analyze_search_results` | `SEARCH_ANALYSIS_PROMPT` | 高 | 中 | `研究假设` + `extracted_facts` |
| 2 | `_analyze_supplementary_results` | 内嵌 supplementary | 中 | 中 | `补充搜索以解决审核发现` |
| 3 | `_analyze_deep_search_results` | 内嵌 deep search | 中 | 中 | `追溯原始数据源` / `further_tracing_queries` |
| 4 | `deep_read_url` | `DEEP_READ_PROMPT` | 低 | 长（~12k chars） | `深度阅读文档` / `## 文档内容` |

---

## 二、SFT 计划与进度

### Phase 1 · 数据层（本地 CPU）

1. 🟡 **真实数据采集路径迁移**
   - ✅ `agents/base.py:_save_training_log()` 默认写 `backend/data/deepscout_logs/`（`DEEPSCOUT_LOG_DIR` 可覆盖）
   - ✅ `export_training_data.py` / `train_lora.py` / `distill_deepscout_data.py` 默认路径同步对齐
   - ✅ `.gitignore` 已忽略 `backend/data/deepscout_logs/` 与 `sft/` adapter
   - ⬜ 跑 20–50 个真实研究主题，攒 ≥500 条 qwen-plus / qwen-max 高质量响应

2. ✅ **蒸馏补量**
   - ✅ teacher 升级到 `qwen-max-latest`（`DEEPSCOUT_TEACHER_MODEL` / `DEEPSCOUT_TEACHER_BASE_URL` 可覆盖）
   - ✅ 4 类 prompt 配比改为 **5 : 2 : 2 : 1**（search 600 / supplementary 240 / deep_search 240 / deep_read 120）

3. ✅ **质量过滤**（`export_training_data.py` 已实现）

4. ✅ **划分**：`stratified_split()` 按 prompt_type 分层 90/10
   - ⬜ eval 集再单独留 50 条做人工抽检

### Phase 2 · 训练（云平台 GPU）

> ⚠️ 本机无 GPU。本节所有命令都在云端执行；本地只负责把 `training_data.jsonl` 上传，把 adapter 下载回来。

#### 2.1 把数据传到云端
本地：
```bash
python scripts/export_training_data.py
# 产出：backend/data/deepscout_logs/training_data.jsonl
```
然后 `scp` / `rsync` / 网盘 上传 `training_data.jsonl` 到云机。

#### 2.2 云端环境
```bash
pip install "unsloth>=2024.8" "trl>=0.9.0" "transformers>=4.43" datasets peft torch
# 或用云平台预置的 PyTorch + CUDA 镜像
```

#### 2.3 云端训练
```bash
export DEEPSCOUT_TRAIN_DATA=/workspace/training_data.jsonl
export DEEPSCOUT_SFT_PATH=/workspace/deepscout_adapter   # 输出 adapter 目录
export DEEPSCOUT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export DEEPSCOUT_EPOCHS=2
export DEEPSCOUT_LR=1e-4
python scripts/train_lora.py
```

训练脚本特性：
- `SFTConfig` + `SFTTrainer`
- `eval_dataset` + `eval_strategy="steps"` + `eval_steps=50`
- `packing=True`、`use_gradient_checkpointing="unsloth"`、`neftune_noise_alpha=5.0`、`lr_scheduler_type="cosine"`、`warmup_ratio=0.03`
- 24GB 单卡 + 4bit + packing 通常足够；若 OOM，把 DEEP_READ 截到 6k 或减 bsz

#### 2.4 把 adapter 下载回本地
adapter 目录只有 ~50–200MB（LoRA 增量权重 + tokenizer 配置），下载到：
```
backend/app/service/deep_research_v2/sft/
```
该路径已在 `.gitignore` 中忽略。完成后无需改任何代码，本地推理会自动加载。

如果路径放别处，可在本地 `.env` 设：
```bash
export DEEPSCOUT_SFT_PATH=/abs/path/to/your/adapter
```

### Phase 3 · 评估

1. ✅ **离线指标脚本**：`scripts/eval_lora.py`（**本地 CPU 也能跑，但慢**；建议在云端训练完顺手跑完再下载报告）

2. ⬜ **对照实验**：同 prompt 与远程 qwen-plus 对比；LLM-as-judge（qwen-max 打分）

3. ⬜ **端到端验证**（本地）：
   - `DEEPSCOUT_USE_LOCAL_SFT=1` 跑 1–2 个完整 deep_research
   - 看 `graph.py` 流转是否正常、Critic 是否触发补搜

### Phase 4 · 本地部署（推理 only）

1. ✅ **adapter 导入**：把云端产物拷到 `backend/app/service/deep_research_v2/sft/`，base.py 自动加载
2. ✅ **CPU 兼容**：`_load_deepscout_local_model` 已根据 `torch.cuda.is_available()` 自动选 fp16+GPU 或 fp32+CPU
3. 🟡 **warm-up 策略调整**：CPU 加载 7B 会很慢且占 ~28GB RAM
   - `app_main.py` 已支持 `DEEPSCOUT_WARMUP=0` 关闭
   - 推荐：本机 RAM < 32GB 或想避免冷启动卡住，**显式设 `DEEPSCOUT_WARMUP=0`**，由首次请求触发懒加载
4. ⬜ **CPU 推理优化（可选，强烈建议）**：transformers 在 CPU 上跑 7B 是分钟级延迟。如果本机要常态推理，建议二选一：
   - **GGUF + llama.cpp / ollama**：把 base+adapter 合并后量化成 Q4_K_M，CPU 跑到秒级
   - **保持 `DEEPSCOUT_USE_LOCAL_SFT=0`**，仅用远程 qwen-plus（adapter 留作离线评估）

---

## 三、关键风险

| 风险 | 状态 | 缓解措施 |
|---|---|---|
| ~~`train_lora.py` 当前 import 即报错~~ | ✅ 已修 | trl `SFTConfig` + 双 API 兼容 |
| 真实日志不足，全靠蒸馏 → 风格漂移 | ⬜ 待处理 | 真实 : 合成 ≥ 1 : 2，真实样本训练时权重 ×2 |
| CPU 推理 7B 速度不可接受 | 🟡 已知 | Phase 4.4：合并后量化为 GGUF；或继续走远程 qwen-plus |
| JSON 结构错乱击穿下游 critic/writer | ⬜ 待评估 | 上线门槛：JSON 合法率 ≥ 98%（Phase 3 测） |
| 同模型自蒸馏导致能力天花板 | ⬜ 待处理 | teacher 升级到更强模型（claude-sonnet-4-6 / qwen-max-latest） |
| 云端到本地传输 adapter 路径不一致 | ✅ 已缓解 | `DEEPSCOUT_SFT_PATH` 环境变量覆盖 |

---

## 四、剩余执行顺序

1. **Phase 1.1 · 跑 20–50 个真实研究主题攒数据**（本地）
2. **Phase 1.2 · 跑蒸馏脚本**（本地）：`python scripts/distill_deepscout_data.py`
3. **Phase 2 · 上云训练**：上传 `training_data.jsonl` → 云端 `train_lora.py` → 下载 adapter 到 `backend/app/service/deep_research_v2/sft/`
4. **Phase 3 · 评估**（云端跑完顺手做）：`python scripts/eval_lora.py`
5. **Phase 4 · 本地推理验证**：`DEEPSCOUT_WARMUP=0 DEEPSCOUT_USE_LOCAL_SFT=1` 跑一次 deep_research，OK 后再决定是否做 GGUF 量化
