# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

"""
DeepResearch V2.0 - Agent 基类

所有专家Agent的基类，提供通用的LLM调用、日志记录等功能。
"""

import json
import logging
import asyncio
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from ..state import ResearchState, AgentLog

# 配置要记录训练数据的 Agent
TARGET_TRAINING_AGENTS = {"DeepScout", "DataAnalyst"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')


class BaseAgent(ABC):
    """
    Agent 基类

    所有专家Agent继承此类，实现特定的 process 方法。
    """
    # DeepScout 本地 SFT 模型缓存（避免重复加载）
    _deepscout_local_bundle: Optional[Dict[str, Any]] = None
    _deepscout_local_load_error: Optional[str] = None

    def __init__(
        self,
        name: str,
        role: str,
        llm_api_key: str,
        llm_base_url: str,
        model: str = "qwen-max"
    ):
        self.name = name
        self.role = role
        self.model = model
        self.client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self.logger = logging.getLogger(f"Agent.{name}")

    def _get_deepscout_adapter_path(self) -> Path:
        """
        获取 DeepScout SFT adapter 路径。
        默认路径：backend/app/service/deep_research_v2/sft
        可通过 DEEPSCOUT_SFT_PATH 覆盖。
        """
        default_path = Path(__file__).resolve().parents[1] / "sft"
        path_str = os.getenv("DEEPSCOUT_SFT_PATH", str(default_path))
        return Path(path_str).expanduser().resolve()

    def _should_use_deepscout_local_model(self) -> bool:
        """
        是否启用 DeepScout 本地模型（HF base + 本地 adapter）。
        默认关闭：首发只用 DashScope；GPU 环境显式 DEEPSCOUT_USE_LOCAL_SFT=1 才启用。
        """
        if self.name != "DeepScout":
            return False
        flag = os.getenv("DEEPSCOUT_USE_LOCAL_SFT", "0").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    @classmethod
    def _load_deepscout_local_model(cls, logger: logging.Logger) -> Dict[str, Any]:
        """
        懒加载 DeepScout 本地模型（仅首次调用加载）。
        """
        if cls._deepscout_local_bundle is not None:
            return cls._deepscout_local_bundle
        if cls._deepscout_local_load_error:
            raise RuntimeError(cls._deepscout_local_load_error)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base_model_name = os.getenv("DEEPSCOUT_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            adapter_path = os.getenv("DEEPSCOUT_SFT_PATH", "")
            if not adapter_path:
                default_path = Path(__file__).resolve().parents[1] / "sft"
                adapter_path = str(default_path)

            adapter_dir = Path(adapter_path).expanduser().resolve()
            if not adapter_dir.exists():
                raise FileNotFoundError(f"DeepScout SFT adapter path not found: {adapter_dir}")

            logger.info(f"[DeepScout] Loading base model from HuggingFace: {base_model_name}")
            logger.info(f"[DeepScout] Loading LoRA adapter from: {adapter_dir}")

            use_cuda = torch.cuda.is_available()
            torch_dtype = torch.float16 if use_cuda else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            model.eval()
            if use_cuda:
                model = model.cuda()

            cls._deepscout_local_bundle = {
                "model": model,
                "tokenizer": tokenizer,
                "base_model_name": base_model_name,
                "adapter_dir": str(adapter_dir)
            }
            logger.info("[DeepScout] Local base+adapter model loaded successfully")
            return cls._deepscout_local_bundle
        except Exception as e:
            cls._deepscout_local_load_error = f"Failed to load local DeepScout model: {e}"
            logger.error(cls._deepscout_local_load_error)
            raise

    def _call_deepscout_local_model_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        同步调用 DeepScout 本地模型（在 to_thread 中执行）。
        """
        bundle = self._load_deepscout_local_model(self.logger)
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]

        final_user_prompt = user_prompt
        if json_mode:
            # 对齐原 call_llm 的 json_mode 行为：尽量只输出 JSON 对象
            final_user_prompt = f"{user_prompt}\n\n请仅输出一个合法的JSON对象，不要输出任何额外说明。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 仅生成合理上限，避免本地推理过慢或 OOM
        local_max_new_tokens = int(os.getenv("DEEPSCOUT_LOCAL_MAX_NEW_TOKENS", "2048"))
        max_new_tokens = max(1, min(max_tokens, local_max_new_tokens))

        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        do_sample = temperature > 0.01
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 0.01)
            generate_kwargs["top_p"] = 0.9

        output_ids = model.generate(input_ids, **generate_kwargs)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if json_mode:
            # 尽量返回规范 JSON 字符串，减少下游解析问题
            parsed = self.parse_json_response(output_text)
            if parsed:
                return json.dumps(parsed, ensure_ascii=False)

        return output_text

    @abstractmethod
    async def process(self, state: ResearchState) -> ResearchState:
        """
        处理状态并返回更新后的状态

        Args:
            state: 当前研究状态

        Returns:
            更新后的状态
        """
        pass

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 16000  # 拉满到最大值
    ) -> str:
        """
        调用 LLM

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            json_mode: 是否强制JSON输出
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM 响应文本
        """
        start_time = time.time()

        # 准备日志数据
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }

        try:
            # DeepScout 优先使用本地 HF base + SFT adapter；失败时自动回退远程 API
            if self._should_use_deepscout_local_model():
                adapter_path = self._get_deepscout_adapter_path()
                if adapter_path.exists():
                    try:
                        content = await asyncio.to_thread(
                            self._call_deepscout_local_model_sync,
                            system_prompt,
                            user_prompt,
                            json_mode,
                            temperature,
                            max_tokens
                        )
                        duration = int((time.time() - start_time) * 1000)
                        self.logger.info(
                            f"Local DeepScout model call completed in {duration}ms, response length: {len(content)}"
                        )

                        if self.name in TARGET_TRAINING_AGENTS:
                            log_data["response"] = content
                            log_data["duration_ms"] = duration
                            log_data["inference_mode"] = "local_sft"
                            self._save_training_log(log_data)

                        return content
                    except Exception as local_e:
                        self.logger.warning(
                            f"Local DeepScout inference failed, fallback to API call: {local_e}"
                        )
                else:
                    self.logger.info(
                        f"DeepScout local adapter not found at {adapter_path}, fallback to API model."
                    )

            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )

            content = response.choices[0].message.content
            duration = int((time.time() - start_time) * 1000)

            self.logger.info(f"LLM call completed in {duration}ms, response length: {len(content)}")

            # 记录训练数据
            if self.name in TARGET_TRAINING_AGENTS:
                log_data["response"] = content
                log_data["duration_ms"] = duration
                self._save_training_log(log_data)

            return content

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")

            # 记录失败的请求
            if self.name in TARGET_TRAINING_AGENTS:
                log_data["error"] = str(e)
                self._save_training_log(log_data)

            raise

    def _save_training_log(self, log_data: dict) -> None:
        """保存训练数据到 JSONL 文件

        默认路径：backend/data/deepscout_logs/
        可通过 DEEPSCOUT_LOG_DIR 覆盖。
        """
        try:
            default_log_dir = Path(__file__).resolve().parents[4] / "data" / "deepscout_logs"
            log_dir = Path(os.getenv("DEEPSCOUT_LOG_DIR", str(default_log_dir))).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"llm_calls_{datetime.now().strftime('%Y%m%d')}.jsonl"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

            self.logger.debug(f"Training log saved to {log_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save training log: {e}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """安全解析JSON响应，处理markdown代码块和格式问题"""
        import re

        def fix_escaped_newlines(s: str) -> str:
            """修复过度转义的换行符"""
            # 处理多层转义: \\\\n -> \n, \\n -> \n
            s = s.replace('\\\\\\\\n', '\n')
            s = s.replace('\\\\n', '\n')
            # 处理可能的 \\r\\n
            s = s.replace('\\\\\\\\r', '\r')
            s = s.replace('\\\\r', '\r')
            return s

        def try_parse(s: str) -> Optional[Dict]:
            """尝试解析JSON，包含修复逻辑"""
            # 清理常见问题
            s = s.strip()
            # 移除可能的BOM
            if s.startswith('\ufeff'):
                s = s[1:]

            try:
                result = json.loads(s)
                # 成功解析后，修复值中的转义字符
                return self._fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            # 尝试修复常见JSON问题
            try:
                # 修复无效的JSON转义序列 \[ \] \# 等 (LLM经常产生这种错误)
                # 需要在字符串值中修复，但避免影响已转义的反斜杠
                # \\[ 是有效的(表示 \[ 字面量)，但 \[ 不是有效的JSON转义
                s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', '', s)
                # 移除注释
                s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
                s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
                # 修复尾随逗号
                s = re.sub(r',(\s*[}\]])', r'\1', s)
                # 修复缺少逗号的情况（在 } 或 ] 后面缺少逗号）
                s = re.sub(r'([}\]])(\s*)([{\[])', r'\1,\2\3', s)
                # 修复没有引号的key
                s = re.sub(r'(\{|\,)\s*(\w+)\s*:', r'\1"\2":', s)
                result = json.loads(s)
                return self._fix_escaped_values(result)
            except json.JSONDecodeError:
                pass

            return None

        # 1. 先尝试直接解析
        result = try_parse(response)
        if result:
            self.logger.debug("Direct JSON parse succeeded")
            return result

        # 2. 尝试提取markdown代码块
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(code_block_pattern, response)
        if match:
            result = try_parse(match.group(1))
            if result:
                self.logger.debug("Extracted JSON from code block")
                return result

        # 3. 尝试找到最外层的 {...}
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            result = try_parse(response[start:end+1])
            if result:
                self.logger.debug("Extracted JSON from braces")
                return result

        # 4. 最后尝试用更宽松的方式解析
        try:
            # 使用 ast.literal_eval 作为备选
            import ast
            # 将 true/false/null 转换为 Python 格式
            s = response
            s = re.sub(r'\btrue\b', 'True', s)
            s = re.sub(r'\bfalse\b', 'False', s)
            s = re.sub(r'\bnull\b', 'None', s)
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1:
                result = ast.literal_eval(s[start:end+1])
                if isinstance(result, dict):
                    self.logger.debug("Parsed using ast.literal_eval")
                    return result
        except:
            pass

        self.logger.error(f"JSON parse error, could not extract valid JSON")
        self.logger.warning(f"Raw response (first 800 chars): {response[:800]}")
        return {}

    def _fix_escaped_values(self, obj: Any, key: str = None) -> Any:
        """
        递归修复字典和列表中的转义字符

        注意：对于 'code' 字段，不处理转义，因为代码中的 \n 是有意义的转义序列
        """
        if isinstance(obj, dict):
            return {k: self._fix_escaped_values(v, key=k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_escaped_values(item, key=key) for item in obj]
        elif isinstance(obj, str):
            # 对于代码字段，不进行转义处理
            # 因为代码中的 \n 应该保持为 \n（两个字符），而不是真正的换行
            if key in ('code', 'fixed_code', 'revised_content'):
                return obj

            # 对于其他字段，修复过度转义的换行符
            result = obj
            result = result.replace('\\\\n', '\n')
            result = result.replace('\\n', '\n')
            result = result.replace('\\\\r', '\r')
            result = result.replace('\\r', '\r')
            result = result.replace('\\\\t', '\t')
            result = result.replace('\\t', '\t')
            return result
        else:
            return obj

    def add_message(self, state: ResearchState, event_type: str, content: Any) -> None:
        """
        添加消息到状态（用于SSE流式输出）

        Args:
            state: 研究状态
            event_type: 事件类型
            content: 消息内容
        """
        message = {
            "type": event_type,
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "content": content
        }
        state["messages"].append(message)

        # 如果有消息队列，立即推送（支持实时流式输出）
        if "_message_queue" in state and state["_message_queue"] is not None:
            try:
                state["_message_queue"].put_nowait(message)
                self.logger.info(f"[SSE] Queued event: {event_type} (queue size: {state['_message_queue'].qsize()})")
            except Exception as e:
                self.logger.warning(f"Failed to push message to queue: {e}")
        else:
            self.logger.warning(f"[SSE] No queue available for event: {event_type}")

    def add_log(
        self,
        state: ResearchState,
        action: str,
        input_summary: str,
        output_summary: str,
        duration_ms: int,
        tokens_used: int = 0
    ) -> None:
        """添加执行日志"""
        log = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "action": action,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "tokens_used": tokens_used
        }
        state["logs"].append(log)


class AgentRegistry:
    """Agent 注册表"""

    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """注册Agent"""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """获取Agent"""
        return cls._agents.get(name)

    @classmethod
    def all(cls) -> Dict[str, BaseAgent]:
        """获取所有Agent"""
        return cls._agents.copy()
