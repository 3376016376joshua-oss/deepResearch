FROM python:3.11.7-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEEPSCOUT_USE_LOCAL_SFT=0 \
    DEEPSCOUT_WARMUP=0

# 系统依赖：curl 用于 healthcheck，build-essential 给可能需要编译的轮子
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 先装依赖（分层缓存：代码改动不会触发 pip install）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码
COPY . /app/

# 非 root 用户
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/hello || exit 1

CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
