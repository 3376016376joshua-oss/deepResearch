#!/usr/bin/env python3
"""
DeepScout 蒸馏数据生成脚本

使用 qwen-max 生成符合 DeepScout 四个 prompt 格式的训练数据：
1. SEARCH_ANALYSIS_PROMPT - 搜索结果分析 (500条)
2. DEEP_READ_PROMPT - 深度阅读分析 (300条)
3. SUPLEMENTARY_SEARCH - 补充搜索分析 (200条)
4. DEEP_SEARCH - 深度递归搜索分析 (200条)

Usage:
    python scripts/distill_deepscout_data.py
"""

import os
import re
import json
import asyncio
import requests
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random


# ============ 读取主题文件 ============
TOPIC_FILE = Path(__file__).parent.parent.parent / "topic.txt"

# 随机种子，保证可复现
RANDOM_SEED = int(os.getenv("DISTILL_RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)


def load_topics_from_file() -> List[Dict]:
    """从 topic.txt 加载主题并 shuffle"""
    if not TOPIC_FILE.exists():
        print(f"Warning: {TOPIC_FILE} not found, using SAMPLE_TOPICS")
        return SAMPLE_TOPICS

    with open(TOPIC_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 方案1：按 JSON 列表解析
    topics = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            topics = [t for t in parsed if isinstance(t, dict)]
        elif isinstance(parsed, dict):
            topics = [parsed]
    except Exception:
        pass

    # 方案2：逐行 JSON 对象（JSONL）
    if not topics:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    topics.append(item)
            except Exception:
                continue

    # 方案3：兼容旧格式（松散对象），回退正则提取
    if not topics:
        # 预处理：替换 JavaScript 布尔值为 Python 格式
        content_py = content.replace(" true", " True").replace(" false", " False").replace(" null", " None")
        topic_pattern = r'\{\s*"query":\s*"[^"]+",\s*"hypotheses":\s*\[.*?\]\s*,\s*"sections":\s*\[.*?\]\s*\}'
        matches = re.findall(topic_pattern, content_py, re.DOTALL)
        import ast
        for match in matches:
            try:
                topic = ast.literal_eval(match)
                if isinstance(topic, dict):
                    topics.append(topic)
            except Exception as e:
                print(f"Warning: Failed to parse topic: {e}")
                continue

    if not topics:
        print(f"Warning: No topics parsed from {TOPIC_FILE}, using SAMPLE_TOPICS")
        return SAMPLE_TOPICS

    # 基础字段校验与修复
    cleaned_topics = []
    for topic in topics:
        query = str(topic.get("query", "")).strip()
        if not query:
            continue
        hypotheses = topic.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []
        sections = topic.get("sections", [])
        if not isinstance(sections, list):
            sections = []
        cleaned_topics.append({
            "query": query,
            "hypotheses": hypotheses,
            "sections": sections
        })

    if not cleaned_topics:
        print(f"Warning: Parsed topics invalid, using SAMPLE_TOPICS")
        return SAMPLE_TOPICS

    topics = cleaned_topics
    random.shuffle(topics)
    print(f"Loaded {len(topics)} topics from {TOPIC_FILE}")
    return topics


# ============ 示例主题（备用） ============
SAMPLE_TOPICS = [
    {
        "query": "车路云一体化发展现状与投资机会",
        "hypotheses": [
            {"id": "h_1", "content": "车路云一体化将在2026年形成千亿级市场", "status": "unverified"},
            {"id": "h_2", "content": "路侧基础设施投资将先于车端带来回报", "status": "unverified"},
        ],
        "sections": [
            {"id": "s1", "title": "政策推动", "description": "车路云一体化试点城市和政策", "search_queries": ["车路云一体化试点城市政策2024"], "status": "pending"},
            {"id": "s2", "title": "技术架构", "description": "路侧设备和云控平台", "search_queries": ["车路云一体化技术架构路侧设备"], "status": "pending"},
            {"id": "s3", "title": "投资机会", "description": "产业链投资标的分析", "search_queries": ["车路云一体化产业链投资机会"], "status": "pending"},
        ]
    },
]


# ============ 配置 ============
OUTPUT_DIR = Path("/tmp/deepscout_training_data/distilled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# qwen-max API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"

# 数据量目标
TARGET_COUNTS = {
    "search_analysis": 500,
    "deep_read": 300,
    "supplementary_search": 200,
    "deep_search": 200,
}

# 数据混合配置（可通过环境变量覆盖）
MIX_MOCK_RATIO = float(os.getenv("MIX_MOCK_RATIO", "0.8"))      # mock 占比
MIX_REAL_RATIO = float(os.getenv("MIX_REAL_RATIO", "0.2"))      # real 占比
NEGATIVE_CASE_RATIO = float(os.getenv("NEGATIVE_CASE_RATIO", "0.1"))  # 负样本注入占比
ENABLE_REAL_SEARCH = os.getenv("ENABLE_REAL_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")

# 修正比例（防止用户配置和不为1）
_mix_total = MIX_MOCK_RATIO + MIX_REAL_RATIO
if _mix_total <= 0:
    MIX_MOCK_RATIO, MIX_REAL_RATIO = 1.0, 0.0
else:
    MIX_MOCK_RATIO = MIX_MOCK_RATIO / _mix_total
    MIX_REAL_RATIO = MIX_REAL_RATIO / _mix_total

# ============ Prompt 模板 ============

SEARCH_ANALYSIS_PROMPT_TEMPLATE = """你是一位资深的研究分析师，擅长从搜索结果中提取关键信息，并验证研究假设。

## 研究问题
{query}

## 当前研究章节
标题: {section_title}
描述: {section_description}

## 研究假设（需要寻找证据支持或反驳）
{hypotheses}

## 搜索结果
{search_results}

## 任务
1. 分析搜索结果，提取结构化信息
2. 寻找支持或反驳研究假设的证据
3. 如果文章引用了数据来源（如"据XX统计"），生成追溯查询

输出JSON格式：
```json
{{
    "extracted_facts": [
        {{
            "content": "提取的事实陈述（要具体、可验证）",
            "source_name": "来源名称",
            "source_url": "来源URL",
            "source_type": "official/academic/news/report/self_media",
            "credibility_score": 0.0-1.0,
            "data_points": [
                {{"name": "指标名", "value": "数值", "unit": "单位", "year": 2024}}
            ],
            "needs_verification": true或false,
            "importance": "high/medium/low",
            "related_hypothesis": "h_1或h_2或null",
            "hypothesis_support": "supports/refutes/neutral"
        }}
    ],
    "hypothesis_evidence": [
        {{
            "hypothesis_id": "h_1",
            "evidence_type": "supports/refutes/inconclusive",
            "evidence_summary": "证据摘要"
        }}
    ],
    "entities_discovered": [
        {{"name": "实体名", "type": "company/person/policy/technology", "relations": ["与XX相关"]}}
    ],
    "key_insights": ["从这些结果中得到的关键洞察"],
    "follow_up_queries": ["需要进一步搜索的关键词"],
    "source_tracing_queries": ["追溯原始数据源的搜索词，如'国家统计局 2024 汽车销量'"],
    "missing_info": ["仍然缺失的信息"],
    "source_quality_assessment": "对整体来源质量的评估"
}}
```

## 评分标准
- 官方来源（政府、央企）: 0.9-1.0
- 学术来源（论文、研究机构）: 0.8-0.95
- 权威媒体（央媒、财经媒体）: 0.7-0.85
- 行业报告（券商、咨询）: 0.7-0.9
- 一般新闻: 0.5-0.7
- 自媒体: 0.2-0.5

请开始分析："""

DEEP_READ_PROMPT_TEMPLATE = """你是一位专业的文档分析师，擅长从长文本中提取关键信息。

## 研究问题
{query}

## 文档来源
URL: {url}
标题: {title}

## 文档内容
{content}

## 任务
深度阅读文档，提取与研究问题相关的所有关键信息。

输出JSON格式：
```json
{{
    "summary": "文档核心内容摘要（200字内）",
    "key_facts": [
        {{
            "content": "关键事实",
            "confidence": 0.0-1.0,
            "page_location": "大概位置描述"
        }}
    ],
    "data_tables": [
        {{
            "title": "数据表标题",
            "headers": ["列1", "列2"],
            "rows": [["值1", "值2"]]
        }}
    ],
    "quotes": ["重要原文引用"],
    "related_entities": ["提到的相关实体"],
    "publication_date": "发布日期（如果能识别）",
    "author_authority": "作者/机构权威性评估"
}}
```"""

SUPPLEMENTARY_SEARCH_PROMPT_TEMPLATE = """你是一位专业的研究分析师，正在补充搜索以解决审核发现的信息缺失问题。

## 原始研究问题
{original_query}

## 补充搜索关键词
{supplementary_query}

## 搜索结果
{search_results}

## 任务
从搜索结果中提取与"{supplementary_query}"直接相关的关键事实和数据。

输出JSON格式：
```json
{{
    "extracted_facts": [
        {{
            "content": "提取的事实陈述",
            "source_name": "来源名称",
            "source_url": "来源URL",
            "source_type": "official/academic/news/report",
            "credibility_score": 0.0-1.0,
            "data_points": [
                {{"name": "指标名", "value": "数值", "unit": "单位"}}
            ]
        }}
    ],
    "key_findings": "本次补充搜索的关键发现"
}}
```"""

DEEP_SEARCH_PROMPT_TEMPLATE = """你是一位专业的研究分析师，正在{search_type_desc}以获取更权威的信息。

## 原始研究问题
{original_query}

## 当前搜索关键词
{search_query}

{hypotheses_text}

## 搜索结果
{search_results}

## 任务
1. 从搜索结果中提取关键事实和数据（特别关注官方来源和权威数据）
2. 如果发现引用了其他权威来源，生成进一步追溯查询

输出JSON格式：
```json
{{
    "extracted_facts": [
        {{
            "content": "提取的事实陈述（要具体、可验证）",
            "source_name": "来源名称",
            "source_url": "来源URL",
            "source_type": "official/academic/news/report",
            "credibility_score": 0.0-1.0,
            "related_hypothesis": "h_1或null",
            "hypothesis_support": "supports/refutes/neutral"
        }}
    ],
    "data_points": [
        {{"name": "指标名", "value": "数值", "unit": "单位", "year": 2024}}
    ],
    "further_tracing_queries": ["如果发现引用了其他权威来源，建议进一步追溯的查询"],
    "source_reliability": "对本次搜索来源可靠性的评估"
}}
```"""


# ============ 模拟搜索结果生成器 ============

def generate_mock_search_results(query: str, count: int = 10) -> List[Dict]:
    """生成模拟搜索结果（实际使用时替换为真实搜索API调用）"""
    sources = [
        {"name": "中华人民共和国交通运输部", "type": "official"},
        {"name": "中国汽车工业协会", "type": "official"},
        {"name": "赛迪顾问", "type": "academic"},
        {"name": "第一财经", "type": "news"},
        {"name": "Wind资讯", "type": "report"},
        {"name": "盖世汽车", "type": "self_media"},
    ]

    results = []
    for i in range(count):
        src = random.choice(sources)
        results.append({
            "title": f"{query}相关研究_{i+1}",
            "url": f"https://example.com/article_{i+1}",
            "site_name": src["name"],
            "date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "summary": f"关于{query}的详细分析报告，涵盖了行业发展、市场规模、技术路线等多个维度的内容..."
        })
    return results


def generate_mock_document_content() -> Dict[str, str]:
    """生成模拟文档内容"""
    return {
        "title": "车路云一体化产业发展报告",
        "url": "https://example.com/report",
        "content": """车路云一体化是智能交通系统的核心发展方向。本报告系统分析了2024年车路云一体化产业的发展现状。

一、政策环境
交通运输部等五部门联合发布《关于开展智能网联汽车"车路云一体化"应用试点工作的通知》，在北京、上海等城市开展试点。

二、市场规模
据赛迪顾问预测，2025年我国车路云一体化市场规模将达到1200亿元，2030年有望突破5000亿元。

三、产业链分析
上游：芯片、传感器、雷达等核心零部件
中游：路侧设备（RSU）、云控平台、通信模块
下游：整车企业、智慧交通运营服务商

四、代表企业
路侧设备：华为、金溢科技、万集科技
云控平台：百度、腾讯、阿里云
整车：比亚迪、华为问界、小鹏汽车"""
    }


NEGATIVE_QUERIES = [
    "宠物用品电商渠道增长率",
    "电竞酒店区域渗透率",
    "连锁咖啡价格战影响",
    "短剧平台付费转化",
    "跨境美妆营销投放ROI",
]


# ============ 数据生成函数 ============

@dataclass
class DistillTask:
    """蒸馏任务"""
    task_type: str  # search_analysis, deep_read, supplementary_search, deep_search
    system_prompt: str
    user_prompt: str
    query: str
    section_title: Optional[str] = None
    section_description: Optional[str] = None
    hypotheses: Optional[str] = None


class DeepScoutDistiller:
    """DeepScout 蒸馏数据生成器"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.enable_real_search = ENABLE_REAL_SEARCH and bool(BOCHA_API_KEY)
        self.generated_count = {
            "search_analysis": 0,
            "deep_read": 0,
            "supplementary_search": 0,
            "deep_search": 0,
        }
        self.data_source_count = {"mock": 0, "real": 0}
        self.negative_injected_count = 0

    def _choose_data_source(self) -> str:
        """按比例选择 mock / real 数据源"""
        if not self.enable_real_search:
            return "mock"
        return "real" if random.random() < MIX_REAL_RATIO else "mock"

    def _fetch_real_search_results_sync(self, query: str, count: int = 10) -> List[Dict]:
        """同步调用博查搜索接口，返回与 mock 同结构的结果"""
        url = "https://api.bocha.cn/v1/web-search"
        payload = {
            "query": query,
            "summary": True,
            "count": count,
            "freshness": "noLimit"
        }
        headers = {
            "Authorization": f"Bearer {BOCHA_API_KEY}",
            "Content-Type": "application/json"
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        # 兼容 bocha 常见返回结构
        result_items = []
        web_pages = (data.get("data") or {}).get("webPages") or {}
        if isinstance(web_pages, dict):
            result_items = web_pages.get("value", []) or []
        if not result_items and isinstance(data.get("data"), list):
            result_items = data.get("data")

        formatted = []
        for item in result_items[:count]:
            if not isinstance(item, dict):
                continue
            formatted.append({
                "title": item.get("name") or item.get("title") or "N/A",
                "url": item.get("url") or "",
                "site_name": item.get("siteName") or item.get("site_name") or "N/A",
                "date": item.get("datePublished") or item.get("date") or "N/A",
                "summary": item.get("summary") or item.get("snippet") or ""
            })
        return formatted

    async def _get_search_results(
        self,
        query: str,
        count: int = 10,
        inject_negative: bool = True,
        track_stats: bool = True
    ) -> Tuple[List[Dict], str]:
        """按 mix 比例获取搜索结果，并可注入负样本"""
        source = self._choose_data_source()
        results: List[Dict] = []

        if source == "real":
            try:
                results = await asyncio.to_thread(self._fetch_real_search_results_sync, query, count)
            except Exception as e:
                print(f"  WARN: real search failed for '{query[:30]}...': {e}, fallback to mock")
                source = "mock"

        if source == "mock" or not results:
            results = generate_mock_search_results(query, count=count)
            source = "mock"

        if track_stats:
            self.data_source_count[source] += 1

        # 注入负样本：混入少量与主题弱相关结果，增强鲁棒性
        if inject_negative and random.random() < NEGATIVE_CASE_RATIO:
            neg_query = random.choice(NEGATIVE_QUERIES)
            neg_results = generate_mock_search_results(neg_query, count=max(1, count // 3))
            tagged_neg = []
            for r in neg_results:
                r2 = dict(r)
                r2["summary"] = f"[低相关样本] {r2.get('summary', '')}"
                tagged_neg.append(r2)
            keep = max(1, int(count * 0.7))
            results = results[:keep] + tagged_neg[: max(1, count - keep)]
            random.shuffle(results)
            self.negative_injected_count += 1

        return results[:count], source

    async def _build_realish_document(self, query: str) -> Dict[str, str]:
        """
        由真实搜索结果拼接长文，作为 deep_read 输入。
        如果真实搜索不可用则回退 mock 文档。
        """
        results, source = await self._get_search_results(query, count=6, inject_negative=False, track_stats=False)
        if source == "mock" and not self.enable_real_search:
            return generate_mock_document_content()

        if not results:
            return generate_mock_document_content()

        title = f"{query}资料汇编"
        url = results[0].get("url", "https://example.com/compiled")
        chunks = []
        for i, r in enumerate(results, start=1):
            chunks.append(
                f"## 资料{i}\n"
                f"标题: {r.get('title', 'N/A')}\n"
                f"来源: {r.get('site_name', 'N/A')}\n"
                f"日期: {r.get('date', 'N/A')}\n"
                f"摘要: {r.get('summary', '')}\n"
            )
        content = "\n\n".join(chunks)
        return {"title": title, "url": url, "content": content}

    def _extract_json_object(self, text: str) -> str:
        """
        从模型输出中提取 JSON 对象字符串。
        兼容 markdown fence、前后解释文本等情况。
        """
        if not text:
            raise ValueError("Empty model response")

        stripped = text.strip()

        # 优先处理 ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            json.loads(candidate)  # 校验
            return candidate

        # 回退：定位第一个 { 与最后一个 }
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end + 1].strip()
            json.loads(candidate)  # 校验
            return candidate

        raise ValueError("No JSON object found in response")

    def _validate_required_keys(self, task_type: str, parsed: Dict[str, Any]) -> bool:
        required = {
            "search_analysis": ["extracted_facts", "hypothesis_evidence", "key_insights", "follow_up_queries"],
            "deep_read": ["summary", "key_facts", "data_tables", "quotes"],
            "supplementary_search": ["extracted_facts", "key_findings"],
            "deep_search": ["extracted_facts", "data_points", "further_tracing_queries", "source_reliability"],
        }
        keys = required.get(task_type, [])
        return all(k in parsed for k in keys)

    async def call_llm(
        self,
        task_type: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3
    ) -> str:
        """调用 LLM，并返回经过清洗验证后的纯 JSON 字符串。"""
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=8192
                )
                raw_text = response.choices[0].message.content or ""
                json_text = self._extract_json_object(raw_text)
                parsed = json.loads(json_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Model response JSON is not an object")
                if not self._validate_required_keys(task_type, parsed):
                    raise ValueError(f"Missing required keys for task_type={task_type}")
                # 统一返回紧凑 JSON，降低训练噪声
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(0.8 * attempt)

        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")

    def format_search_analysis_prompt(
        self,
        query: str,
        section: Dict,
        hypotheses: List[Dict],
        search_results: List[Dict]
    ) -> DistillTask:
        """格式化搜索分析 prompt"""
        hypotheses_text = "\n".join([
            f"- [{h.get('id')}] {h.get('content')} (状态: {h.get('status', 'unverified')})"
            for h in hypotheses
        ]) if hypotheses else "无特定假设"

        search_results_text = []
        for i, r in enumerate(search_results[:15]):
            search_results_text.append(f"""
[{i+1}] {r.get('title', 'N/A')}
URL: {r.get('url', '')}
来源: {r.get('site_name', 'N/A')}
日期: {r.get('date', 'N/A')}
摘要: {r.get('summary', '')[:300]}
""")

        user_prompt = SEARCH_ANALYSIS_PROMPT_TEMPLATE.format(
            query=query,
            section_title=section.get("title", ""),
            section_description=section.get("description", ""),
            hypotheses=hypotheses_text,
            search_results="\n".join(search_results_text)
        )

        return DistillTask(
            task_type="search_analysis",
            system_prompt="你是专业的研究分析师，擅长从搜索结果中提取结构化信息、验证假设并评估来源质量。",
            user_prompt=user_prompt,
            query=query,
            section_title=section.get("title"),
            section_description=section.get("description"),
            hypotheses=hypotheses_text
        )

    def format_deep_read_prompt(
        self,
        query: str,
        document: Dict[str, str]
    ) -> DistillTask:
        """格式化深度阅读 prompt"""
        user_prompt = DEEP_READ_PROMPT_TEMPLATE.format(
            query=query,
            url=document.get("url", ""),
            title=document.get("title", ""),
            content=document.get("content", "")
        )

        return DistillTask(
            task_type="deep_read",
            system_prompt="你是专业的文档分析师。",
            user_prompt=user_prompt,
            query=query
        )

    def format_supplementary_search_prompt(
        self,
        original_query: str,
        supplementary_query: str,
        search_results: List[Dict]
    ) -> DistillTask:
        """格式化补充搜索 prompt"""
        search_results_text = []
        for r in search_results[:8]:
            search_results_text.append(f"标题: {r.get('title', 'N/A')}\n来源: {r.get('site_name', 'N/A')}\n内容: {r.get('summary', '')[:300]}")

        user_prompt = SUPPLEMENTARY_SEARCH_PROMPT_TEMPLATE.format(
            original_query=original_query,
            supplementary_query=supplementary_query,
            search_results="\n".join(search_results_text)
        )

        return DistillTask(
            task_type="supplementary_search",
            system_prompt="你是专业的信息提取专家，擅长从搜索结果中提取结构化信息。",
            user_prompt=user_prompt,
            query=original_query
        )

    def format_deep_search_prompt(
        self,
        original_query: str,
        search_query: str,
        search_type: str,
        hypotheses: List[Dict],
        search_results: List[Dict]
    ) -> DistillTask:
        """格式化深度递归搜索 prompt"""
        search_type_desc = "追溯原始数据源" if search_type == "source_tracing" else "追踪相关线索"

        hypotheses_text = ""
        if hypotheses:
            hypotheses_text = "## 研究假设\n" + "\n".join([
                f"- [{h.get('id')}] {h.get('content')}" for h in hypotheses[:3]
            ])

        search_results_text = []
        for r in search_results[:6]:
            search_results_text.append(f"标题: {r.get('title', 'N/A')}\n来源: {r.get('site_name', 'N/A')}\n内容: {r.get('summary', '')[:300]}")

        user_prompt = DEEP_SEARCH_PROMPT_TEMPLATE.format(
            original_query=original_query,
            search_query=search_query,
            search_type_desc=search_type_desc,
            hypotheses_text=hypotheses_text,
            search_results="\n".join(search_results_text)
        )

        return DistillTask(
            task_type="deep_search",
            system_prompt="你是专业的信息验证专家，擅长从搜索结果中提取权威信息并追溯原始来源。",
            user_prompt=user_prompt,
            query=original_query
        )

    async def generate_search_analysis_data(
        self,
        topics: List[Dict]
    ) -> List[Dict]:
        """生成搜索分析数据"""
        print("\n[1/4] 生成 SEARCH_ANALYSIS 数据...")
        all_tasks: List[Dict[str, Any]] = []

        for topic in topics:
            query = topic["query"]
            hypotheses = topic.get("hypotheses", [])
            sections = topic.get("sections", [])

            for section in sections:
                all_tasks.append({
                    "query": query,
                    "section": section,
                    "hypotheses": hypotheses
                })

        # 打乱顺序并扩展到目标数量（主题不足时循环采样）
        random.shuffle(all_tasks)
        target_n = TARGET_COUNTS["search_analysis"]
        if not all_tasks:
            print("  WARNING: 没有可用的 search_analysis 任务样本")
            return []
        if len(all_tasks) < target_n:
            expanded = []
            while len(expanded) < target_n:
                expanded.extend(random.sample(all_tasks, min(len(all_tasks), target_n - len(expanded))))
            all_tasks = expanded
        else:
            all_tasks = all_tasks[:target_n]

        results = []
        for i, task in enumerate(all_tasks):
            section = task.get("section", {})
            query = task.get("query", "")
            hypotheses = task.get("hypotheses", [])
            section_title = section.get("title", "")
            print(f"  生成 {i+1}/{len(all_tasks)}: {section_title}")
            try:
                section_query = section_title or query
                search_results, data_source = await self._get_search_results(section_query, count=10, inject_negative=True)

                hydrated_task = self.format_search_analysis_prompt(
                    query=query,
                    section=section,
                    hypotheses=hypotheses,
                    search_results=search_results
                )
                response = await self.call_llm(
                    task_type=hydrated_task.task_type,
                    system_prompt=hydrated_task.system_prompt,
                    user_prompt=hydrated_task.user_prompt
                )
                results.append({
                    "task_type": hydrated_task.task_type,
                    "query": hydrated_task.query,
                    "section_title": hydrated_task.section_title,
                    "section_description": hydrated_task.section_description,
                    "hypotheses": hydrated_task.hypotheses,
                    "system_prompt": hydrated_task.system_prompt,
                    "user_prompt": hydrated_task.user_prompt,
                    "response": response,
                    "data_source": data_source,
                    "timestamp": datetime.now().isoformat()
                })
                self.generated_count["search_analysis"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")

        return results

    async def generate_deep_read_data(self, topics: List[Dict]) -> List[Dict]:
        """生成深度阅读数据"""
        print("\n[2/4] 生成 DEEP_READ 数据...")
        results = []

        for i in range(TARGET_COUNTS["deep_read"]):
            topic = random.choice(topics)
            query = topic["query"]
            source = self._choose_data_source()
            if source == "real":
                document = await self._build_realish_document(query)
                # 真实源失败时 _build_realish_document 会回退 mock，这里重新判定
                data_source = "real" if "资料汇编" in document.get("title", "") else "mock"
            else:
                document = generate_mock_document_content()
                data_source = "mock"
            self.data_source_count[data_source] += 1

            task = self.format_deep_read_prompt(query=query, document=document)

            print(f"  生成 {i+1}/{TARGET_COUNTS['deep_read']}")
            try:
                response = await self.call_llm(
                    task_type=task.task_type,
                    system_prompt=task.system_prompt,
                    user_prompt=task.user_prompt
                )
                results.append({
                    "task_type": task.task_type,
                    "query": task.query,
                    "document_title": document["title"],
                    "document_url": document["url"],
                    "system_prompt": task.system_prompt,
                    "user_prompt": task.user_prompt,
                    "response": response,
                    "data_source": data_source,
                    "timestamp": datetime.now().isoformat()
                })
                self.generated_count["deep_read"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")

        return results

    async def generate_supplementary_search_data(self, topics: List[Dict]) -> List[Dict]:
        """生成补充搜索数据"""
        print("\n[3/4] 生成 SUPLEMENTARY_SEARCH 数据...")
        results = []

        for i in range(TARGET_COUNTS["supplementary_search"]):
            topic = random.choice(topics)
            query = topic["query"]
            supplementary_query = f"{query}的关键数据"

            search_results, data_source = await self._get_search_results(
                supplementary_query, count=8, inject_negative=True
            )
            task = self.format_supplementary_search_prompt(
                original_query=query,
                supplementary_query=supplementary_query,
                search_results=search_results
            )

            print(f"  生成 {i+1}/{TARGET_COUNTS['supplementary_search']}")
            try:
                response = await self.call_llm(
                    task_type=task.task_type,
                    system_prompt=task.system_prompt,
                    user_prompt=task.user_prompt
                )
                results.append({
                    "task_type": task.task_type,
                    "query": task.query,
                    "supplementary_query": supplementary_query,
                    "system_prompt": task.system_prompt,
                    "user_prompt": task.user_prompt,
                    "response": response,
                    "data_source": data_source,
                    "timestamp": datetime.now().isoformat()
                })
                self.generated_count["supplementary_search"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")

        return results

    async def generate_deep_search_data(self, topics: List[Dict]) -> List[Dict]:
        """生成深度递归搜索数据"""
        print("\n[4/4] 生成 DEEP_SEARCH 数据...")
        results = []

        for i in range(TARGET_COUNTS["deep_search"]):
            topic = random.choice(topics)
            query = topic["query"]
            hypotheses = topic.get("hypotheses", [])
            search_query = f"{query}的数据来源"
            search_type = random.choice(["source_tracing", "follow_up"])

            search_results, data_source = await self._get_search_results(
                search_query, count=6, inject_negative=True
            )
            task = self.format_deep_search_prompt(
                original_query=query,
                search_query=search_query,
                search_type=search_type,
                hypotheses=hypotheses,
                search_results=search_results
            )

            print(f"  生成 {i+1}/{TARGET_COUNTS['deep_search']}")
            try:
                response = await self.call_llm(
                    task_type=task.task_type,
                    system_prompt=task.system_prompt,
                    user_prompt=task.user_prompt
                )
                results.append({
                    "task_type": task.task_type,
                    "query": task.query,
                    "search_query": search_query,
                    "search_type": search_type,
                    "system_prompt": task.system_prompt,
                    "user_prompt": task.user_prompt,
                    "response": response,
                    "data_source": data_source,
                    "timestamp": datetime.now().isoformat()
                })
                self.generated_count["deep_search"] += 1
            except Exception as e:
                print(f"  ERROR: {e}")

        return results

    async def generate_all(self, topics: List[Dict]) -> Dict[str, List[Dict]]:
        """生成所有类型的数据"""
        print("=" * 60)
        print("DeepScout 蒸馏数据生成")
        print("=" * 60)
        print(f"目标数据量:")
        for k, v in TARGET_COUNTS.items():
            print(f"  - {k}: {v}")
        print(f"主题数量: {len(topics)}")
        print(f"数据混合比例: mock={MIX_MOCK_RATIO:.2f}, real={MIX_REAL_RATIO:.2f}")
        print(f"负样本注入比例: {NEGATIVE_CASE_RATIO:.2f}")
        print(f"真实检索可用: {self.enable_real_search}")
        print("=" * 60)

        all_data = {
            "search_analysis": [],
            "deep_read": [],
            "supplementary_search": [],
            "deep_search": [],
        }

        all_data["search_analysis"] = await self.generate_search_analysis_data(topics)
        all_data["deep_read"] = await self.generate_deep_read_data(topics)
        all_data["supplementary_search"] = await self.generate_supplementary_search_data(topics)
        all_data["deep_search"] = await self.generate_deep_search_data(topics)

        print("\n" + "=" * 60)
        print("生成完成!")
        print("=" * 60)
        print("实际生成数据量:")
        for k, v in self.generated_count.items():
            print(f"  - {k}: {v}")
        print("数据源统计:")
        print(f"  - mock: {self.data_source_count['mock']}")
        print(f"  - real: {self.data_source_count['real']}")
        print(f"负样本注入次数: {self.negative_injected_count}")

        return all_data


def save_distilled_data(all_data: Dict[str, List[Dict]]) -> None:
    """保存蒸馏数据"""
    # 按类型保存
    for task_type, data in all_data.items():
        if data:
            output_file = OUTPUT_DIR / f"{task_type}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"保存: {output_file} ({len(data)} 条)")

    # 合并为统一格式
    merged_file = OUTPUT_DIR / "merged.jsonl"
    with open(merged_file, "w", encoding="utf-8") as f:
        for task_type, data in all_data.items():
            for item in data:
                # 转换为 LoRA 训练格式
                training_item = {
                    "task_type": item["task_type"],
                    "messages": [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_prompt"]},
                        {"role": "assistant", "content": item["response"]}
                    ],
                    "query": item["query"],
                    "timestamp": item["timestamp"]
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")

    print(f"合并保存: {merged_file}")
    print(f"总计: {sum(len(d) for d in all_data.values())} 条")

    # 生成纯训练字段版本（很多 LoRA 训练脚本只读取 messages）
    lora_file = OUTPUT_DIR / "lora_train.jsonl"
    with open(lora_file, "w", encoding="utf-8") as f:
        for task_type, data in all_data.items():
            for item in data:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_prompt"]},
                        {"role": "assistant", "content": item["response"]}
                    ]
                }, ensure_ascii=False) + "\n")
    print(f"LoRA训练文件: {lora_file}")


async def main():
    """主函数"""
    if not DASHSCOPE_API_KEY:
        print("Error: DASHSCOPE_API_KEY 环境变量未设置")
        return

    distiller = DeepScoutDistiller(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        model=MODEL_NAME
    )

    # 从 topic.txt 加载主题（会自动 shuffle）
    topics = load_topics_from_file()

    # 生成数据
    all_data = await distiller.generate_all(topics)

    # 保存数据
    save_distilled_data(all_data)

    print("\n下一步:")
    print("1. 查看生成的数据: ls /tmp/deepscout_training_data/distilled/")
    print("2. 运行 export_training_data.py 转换为 LoRA 训练格式")
    print("3. 运行 train_lora.py 开始微调")


if __name__ == "__main__":
    asyncio.run(main())
