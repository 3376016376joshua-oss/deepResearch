# Vercel + Railway 免费档部署方案

> 目标：用免费/低成本 PaaS 跑通 DashScope-only 模式的最小可用 demo，不依赖自建 ECS。
> 前提：已确定**首发只调 DashScope API**，无需 GPU、无需本地 SFT。

---

## 0. 残酷的现实先说在前面

当前 `docker-compose.yml` 编排了 5 个中间件：Postgres / Redis / Milvus(+etcd+minio) / Elasticsearch。

- **Vercel**：只跑无状态函数和静态站，**不能托管这些中间件**。
- **Railway**：免费档现已改为 **$5 试用额度用完即停**（不再有真正"永久免费"），可以跑 Postgres + Redis，但 **Milvus 和 Elasticsearch 跑不动**（内存超限 / 镜像超大）。

所以走 Vercel/Railway 这一档，必须先做**功能裁剪**，确认哪些能力 demo 阶段可以不要：

| 模块 | 依赖 | demo 能否砍掉 |
|---|---|---|
| 用户/会话/任务记录 | Postgres | ❌ 必须保留 |
| 缓存/限流/SSE 状态 | Redis | ⚠️ 可降级为内存缓存（单实例） |
| 向量检索 / RAG | Milvus | ✅ demo 阶段先关，或换 pgvector |
| 全文检索 / 资讯 | Elasticsearch | ✅ demo 阶段先关，或换 Postgres FTS |

**核心建议**：把 Milvus + ES 暂时禁用，向量能力用 **pgvector 扩展**替代（Railway/Supabase 的 Postgres 都支持），全文检索用 Postgres 自带 FTS。这样中间件压缩到 Postgres + Redis 两个，免费档才放得下。

---

## 1. 推荐架构

```
[用户浏览器]
     │
     ├──> Vercel (前端静态站)  ── frontend/dist
     │       │
     │       └── 调 /api/* 走环境变量里的后端域名
     │
     └──> Railway (后端容器)   ── backend/app/Dockerfile
             │
             ├──> Railway Postgres (含 pgvector)
             ├──> Railway Redis（或 Upstash 免费档）
             └──> DashScope API（外部）
```

**为什么这么分**：
- 前端静态资源放 Vercel，CDN 全球加速，**真免费、不限流量**。
- 后端是有状态长连接服务（SSE / WebSocket），放 Railway 比 Vercel Serverless 合适得多（Vercel 函数有 10s/60s 超时，DashScope 流式响应容易被截断）。
- Railway 同项目内 Postgres/Redis 走内网，零延迟。

**备选**：
- 后端也可放 **Render / Fly.io 免费档**，逻辑一样，Railway 配额用完就切。
- Postgres 想长期免费可用 **Supabase**（500MB + pgvector），Redis 用 **Upstash**（10K 命令/天免费）。

---

## 2. 部署前的代码改动（必做）

### 2.1 后端瘦身
- [ ] 在 `backend/app/app_main.py` 里加环境变量开关，跳过 Milvus / ES 初始化：
  - `DISABLE_MILVUS=1` → 跳过向量库连接
  - `DISABLE_ES=1` → 跳过 ES 索引初始化
- [ ] `requirements.txt` 拆成两份：
  - `requirements-prod.txt`：仅保留 fastapi / uvicorn / openai(dashscope) / sqlalchemy / redis / pgvector
  - 剥离 `torch / transformers / peft / bitsandbytes / pymilvus / elasticsearch`（demo 不需要，能把镜像从 ~5GB 砍到 < 500MB）
- [ ] Dockerfile 用 `python:3.11-slim`，多阶段构建，最终镜像目标 < 400MB（Railway 免费档对镜像大小敏感）

### 2.2 配置项收口
- [ ] 所有密钥走环境变量，不写文件：
  - `DASHSCOUT_API_KEY` / `DASHSCOPE_API_KEY`
  - `DATABASE_URL`（Railway 自动注入）
  - `REDIS_URL`（Railway 自动注入）
  - `DEEPSCOUT_USE_LOCAL_SFT=0`、`DEEPSCOUT_WARMUP=0`
  - `DISABLE_MILVUS=1`、`DISABLE_ES=1`
- [ ] CORS 允许 Vercel 前端域名（先用 `*.vercel.app` 通配，正式上线再收紧）

### 2.3 前端
- [ ] `frontend/` 里把后端地址改成 `import.meta.env.VITE_API_BASE`，本地用 `.env.development`，线上 Vercel 配 `VITE_API_BASE=https://<railway-app>.up.railway.app`
- [ ] `vercel.json` 加 SPA fallback（所有未知路径回退到 `index.html`）

---

## 3. Railway 后端部署步骤

1. 注册 https://railway.app（GitHub 登录），绑定信用卡领 $5 试用额度。
2. **New Project → Deploy from GitHub Repo**，选这个仓库，根目录指到 `backend/`。
3. Railway 检测到 Dockerfile 自动构建。如果没识别，手动在 Settings → Build 设置 `Dockerfile Path = backend/app/Dockerfile`、`Build Context = backend/`。
4. 在同一 project 里 **+ New → Database → Add PostgreSQL**，会自动注入 `DATABASE_URL`。
   - 进入 Postgres 服务，Data 标签里跑 `CREATE EXTENSION vector;` 启用 pgvector。
5. **+ New → Database → Add Redis**（或外接 Upstash，配 `REDIS_URL`）。
6. 在后端服务的 **Variables** 里补齐：
   ```
   DASHSCOPE_API_KEY=sk-xxx
   DEEPSCOUT_USE_LOCAL_SFT=0
   DEEPSCOUT_WARMUP=0
   DISABLE_MILVUS=1
   DISABLE_ES=1
   CORS_ORIGINS=https://<your-app>.vercel.app
   PORT=8000
   ```
7. **Settings → Networking → Generate Domain**，得到 `https://<service>.up.railway.app`，记下来。
8. 看 Deploy logs 确认启动成功，访问 `<domain>/health`（如果有）或 `/docs` 验证。

**坑位提醒**：
- Railway 镜像构建有 **4GB 内存上限**，包含 torch 一定 OOM。先确认 `requirements-prod.txt` 干净。
- 默认实例规格 512MB RAM，FastAPI + 一个 worker 够用；如果 OOM 把 gunicorn workers 调成 1。
- Healthcheck 路径要返回 200，否则 Railway 会反复重启。

---

## 4. Vercel 前端部署步骤

1. 注册 https://vercel.com，**Import Git Repository** 选同一个仓库。
2. **Root Directory** 设为 `frontend`，Framework Preset 选 Vite。
3. Build Command：`npm run build`，Output Directory：`dist`。
4. **Environment Variables**：
   ```
   VITE_API_BASE = https://<service>.up.railway.app
   ```
5. Deploy。完成后得到 `https://<your-app>.vercel.app`。
6. 回到 Railway 后端的 `CORS_ORIGINS` 把这个域名填进去，重启后端。

---

## 5. 上线后必做的几件事

- [ ] **DashScope Key 限流保护**：在后端加请求频控（每 IP 每分钟 N 次），防止 demo 链接被人刷爆账单。
- [ ] **DashScope 用量告警**：阿里云控制台设消费阈值短信提醒。
- [ ] **流式响应**：SSE 走通，否则用户等 10s+ 看不到反馈。Railway 默认支持长连接，无需特殊配置。
- [ ] **日志**：Railway 自带日志查看，关键错误打到 stdout 即可。需要长期保存可接 Logtail / Better Stack 免费档。
- [ ] **冷启动**：Railway 实例长时间无流量会被休眠，首次请求慢 5–10s。免费档接受；不接受可上 $5/月 Hobby 计划保活。

---

## 6. 成本预估（个人 demo）

| 项 | 用量 | 月成本 |
|---|---|---|
| Vercel 前端 | Hobby 免费 | ¥0 |
| Railway 后端 + Postgres + Redis | $5 试用额度 | 用完后约 ¥35–70/月 |
| DashScope Qwen API | 看调用量，qwen-plus ~¥0.004/1K tokens | demo 阶段 ¥10–50/月 |
| 域名（可选） | .com 首年 | ¥70/年 |

**真·零成本组合**：Vercel + Render 免费档（750 小时/月）+ Supabase Postgres 免费档 + Upstash Redis 免费档。代价是冷启动更明显、配额更紧。

---

## 7. 后续演进路径

- demo 验证后想恢复 Milvus / ES → 切回阿里云 ECS，用现有 `docker-compose.yml`（参考 `plan/deployment.md`）。
- 想加本地 SFT 推理 → 加一个 GPU worker（autodl/runpod 按小时），主链路不变。
- pgvector 性能不够 → 再迁 Milvus，向量表是独立的，迁移面小。

---

## 8. TODO 清单（按顺序做）

1. [ ] 代码改动：环境变量开关、requirements 拆分、Dockerfile 瘦身、前端 API base
2. [ ] 本地用 `DISABLE_MILVUS=1 DISABLE_ES=1` 跑通后端，确认无中间件依赖也能起
3. [ ] 推一个分支到 GitHub
4. [ ] Railway 部署后端 + Postgres + Redis，拿到域名
5. [ ] Vercel 部署前端，配 `VITE_API_BASE`
6. [ ] 端到端测一次完整对话流
7. [ ] 加限流 + 用量告警
8. [ ] （可选）绑域名 + HTTPS
