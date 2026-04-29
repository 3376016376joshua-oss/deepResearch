# Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
# 未经授权，禁止转售或仿制。

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from router import document_router, search_router, chat_router, research_router
from router.auth_router import router as auth_router
from router.session_router import router as session_router
from router.knowledge_router import router as knowledge_router
from router.attachment_router import router as attachment_router
from router.memory_router import router as memory_router
from router.database_router import router as database_router
from router.news_router import router as news_router
from core.database import engine, Base
# 导入所有模型以确保它们被注册
from models import (
    User, ChatSession, ChatMessage, ChatAttachment, LongTermMemory,
    KnowledgeBase, Document, IndustryStats, CompanyData, PolicyData,
    ResearchCheckpoint, IndustryNews, BiddingInfo, NewsCollectionTask
)

# 创建所有数据表（如果不存在）
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("应用启动中...")

    # 初始化定时任务调度器并检查数据
    try:
        from service.scheduler_service import init_scheduler_and_check_data
        await init_scheduler_and_check_data()
        logger.info("定时任务调度器启动成功")
    except Exception as e:
        logger.error(f"定时任务调度器启动失败: {e}")

    # DeepScout 本地 SFT adapter warm-up：避免首条请求懒加载延迟
    # 关闭条件（任一满足即跳过）：
    #   - DEEPSCOUT_WARMUP=0
    #   - DEEPSCOUT_USE_LOCAL_SFT=0
    #   - 无 CUDA 且未显式 DEEPSCOUT_WARMUP=1（CPU 加载 7B fp32 ~28GB RAM，分钟级阻塞启动）
    import os
    warmup_env = os.getenv("DEEPSCOUT_WARMUP", "").strip().lower()
    # 默认 0：首发只用 DashScope，避免无 GPU 环境误加载 7B 模型阻塞启动
    use_local_flag = os.getenv("DEEPSCOUT_USE_LOCAL_SFT", "0").strip().lower()
    provider = "local_sft" if use_local_flag in {"1", "true", "yes", "on"} else "dashscope"
    logger.info(f"[startup] LLM provider = {provider}")

    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    if warmup_env in {"1", "true", "yes", "on"}:
        warmup_enabled = True
    elif warmup_env in {"0", "false", "no", "off"}:
        warmup_enabled = False
    else:
        warmup_enabled = _has_cuda()  # 默认：仅在有 GPU 时 warm-up

    if warmup_enabled and use_local_flag not in {"0", "false", "no", "off"}:
        try:
            import asyncio
            from service.deep_research_v2.agents.base import BaseAgent
            await asyncio.to_thread(BaseAgent._load_deepscout_local_model, logger)
            logger.info("DeepScout 本地模型 warm-up 完成")
        except Exception as e:
            logger.warning(f"DeepScout 本地模型 warm-up 失败（运行时将自动回退远程 API）: {e}")
    else:
        logger.info(
            "DeepScout warm-up 跳过（无 CUDA 或被显式关闭），首次请求时懒加载或回退远程 API"
        )

    yield

    # 关闭时执行
    logger.info("应用关闭中...")
    try:
        from service.scheduler_service import get_scheduler_service
        scheduler = get_scheduler_service()
        scheduler.stop()
    except Exception as e:
        logger.error(f"定时任务调度器关闭失败: {e}")


app = FastAPI(
    title="行业信息助手 API",
    description="基于 AI Agent 的行业信息助手系统",
    version="2.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
import os as _os
_cors_origins_env = _os.getenv("CORS_ORIGINS", "*").strip()
_allow_origins = ["*"] if _cors_origins_env in ("", "*") else [
    o.strip() for o in _cors_origins_env.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth_router)
app.include_router(session_router)
app.include_router(knowledge_router)
app.include_router(attachment_router)
app.include_router(memory_router)
app.include_router(database_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(research_router)
app.include_router(news_router)

@app.get("/hello")
async def hello_world():
    """
    Simple hello world endpoint for network verification
    """
    return {
        "status": "success",
        "message": "Hello World! The API is working correctly."
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
