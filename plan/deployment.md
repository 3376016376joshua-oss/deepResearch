# Industry Information Assistant 部署方案

> 目标：将本地开发的项目部署到生产/测试服务器，保持当前 Docker Compose 中间件 + FastAPI 后端 + React 前端的架构。

## 已确认的决策（2026-04-28）

1. **服务器**：阿里云 ECS，公网 IP `47.84.206.6`，连接方式 `ssh root@47.84.206.6`（待确认：地域、规格、OS、是否已配 SSH Key、安全组放行端口）。
2. **域名**：暂未就绪 → 先用公网 IP + 自签证书或 HTTP 跑通；正式上线前再绑域名 + Let's Encrypt。
3. **模型路线**：首发**只用 DashScope API**（Qwen，最早实现），**不启用本地 SFT/LoRA**。
   - 所有部署环境强制 `DEEPSCOUT_USE_LOCAL_SFT=0`、`DEEPSCOUT_WARMUP=0`
   - 不需要 GPU 节点，普通 CPU 服务器即可
   - SFT 相关的依赖（torch/transformers/peft/bitsandbytes 等）可以从生产镜像中**剥离**，显著瘦身镜像体积、加快冷启动 → 见 §13.4
   - 本地 SFT 路径保留在代码与训练脚本中，后续切换只需改环境变量 + 加一个 GPU worker，不影响主链路

---

## 0. 系统现状速览

- **后端**：FastAPI（Python 3.11），入口 `backend/app/app_main.py`，已有 `backend/app/Dockerfile`，监听 `:8000`。
- **前端**：React 19 + Vite 7（`frontend/`），`npm run build` 产出静态资源到 `dist/`。
- **中间件**：`docker-compose.yml` 已编排 Postgres、Redis、Milvus(+etcd+minio)、Elasticsearch。
- **可选 GPU**：DeepScout 本地 SFT/LoRA 推理（`DEEPSCOUT_USE_LOCAL_SFT`、`DEEPSCOUT_WARMUP`），约需 ≥ 24GB 显存或 28GB+ RAM。
- **外部依赖**：DashScope（Qwen）、博查搜索、可选 OpenRouter / DocMind / 招投标 API。

---

## 1. 部署形态选择（先选一个）

| 形态 | 适用 | 资源 | 说明 |
|---|---|---|---|
| **A. 单机 Docker Compose** | 内测、Demo、单租户 | 1 台 16C/32G + 200G SSD | 最快，所有组件单机 |
| **B. 单机 + 外部托管 DB** | 准生产 | 1 台 8C/16G + 托管 RDS/Redis | Postgres/Redis 用云托管，Milvus/ES 自托管 |
| **C. K8s 多节点** | 正式生产、多租户 | ≥3 节点 | 后续扩展，先不做，本计划以 A/B 为主 |
| **+ GPU 节点（可选）** | 启用本地 SFT | A10/3090 24G+ | 不启用则 `DEEPSCOUT_USE_LOCAL_SFT=0`，全用 DashScope |

**推荐起步：形态 A**，跑通后再迁移到 B。下面步骤以 A 为主，B 在 §6 标注差异。

---

## 2. 服务器与域名准备

- [ ] 一台 Linux 服务器（Ubuntu 22.04 / Debian 12 / CentOS Stream，x86_64）
  - 至少 8C/16G/200G，启用 Swap 8G
  - 开放端口：`80`、`443`（对外）；`22`（运维）；其余端口全部走内网/回环
- [ ] 域名 + DNS：例如 `app.example.com`（前端） 与 `api.example.com`（后端），均 A 记录指向服务器
- [ ] 安装基础组件：
  - Docker ≥ 24，Docker Compose plugin
  - Nginx（反向代理 + TLS 终止）
  - certbot（Let's Encrypt 免费证书）
  - Node.js 20 LTS（构建前端用，可选放到 CI）
  - Git
- [ ] （可选）GPU 节点额外装：NVIDIA Driver ≥ 535、`nvidia-container-toolkit`

---

## 3. 代码与配置

### 3.1 拉取代码
```bash
sudo mkdir -p /opt/industry && sudo chown $USER /opt/industry
cd /opt/industry
git clone <repo-url> app && cd app
```

### 3.2 后端 `.env`（生产化要点）
基于 `backend/.env` 创建 `backend/.env.prod`，关键修改：
- `POSTGRES_HOST=postgres`、`REDIS_HOST=redis`、`MILVUS_HOST=milvus`、`ELASTICSEARCH_HOST=elasticsearch`（容器互联用 service 名，**不再是 localhost**）
- `POSTGRES_PASSWORD` 改成强口令；同步改 `docker-compose.yml`
- `JWT_SECRET` / `SECRET_KEY` 用 `openssl rand -hex 32` 重新生成
- 填上线上 key：`DASHSCOPE_API_KEY`、`BOCHA_API_KEY` 等（**绝不入库**）
- `DEEPSCOUT_USE_LOCAL_SFT=0`、`DEEPSCOUT_WARMUP=0`（无 GPU 时）
- `LOG_LEVEL=INFO`、`ENV=prod`

### 3.3 前端 `.env.production`
```env
VITE_TITLE=行业咨询助手
VITE_API_BASE=/api/
VITE_API_PROXY=/api/
```
> 生产由 Nginx 同源代理到后端，避免 CORS。

---

## 4. 镜像与编排

### 4.1 在 `docker-compose.yml` 增加 backend 与 frontend 服务
新增 `docker-compose.prod.yml`（与现有 compose 叠加使用）：

```yaml
services:
  backend:
    build:
      context: ./backend/app
      dockerfile: Dockerfile
    container_name: industry_backend
    restart: unless-stopped
    env_file: ./backend/.env.prod
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_healthy }
      milvus:   { condition: service_healthy }
    volumes:
      - ./backend/app/data:/app/data
      - ./backend/app/logs:/app/logs
    networks: [industry_network]
    expose: ["8000"]   # 仅内网暴露

  frontend:
    image: nginx:1.27-alpine
    container_name: industry_frontend
    restart: unless-stopped
    volumes:
      - ./frontend/dist:/usr/share/nginx/html:ro
      - ./docker/nginx/site.conf:/etc/nginx/conf.d/default.conf:ro
    networks: [industry_network]
    expose: ["80"]
```

### 4.2 后端 Dockerfile 小优化（建议改动）
当前 `backend/app/Dockerfile` 直接 `COPY .` 再 `pip install`，每次代码改动都要重装依赖。建议：
1. 先 `COPY requirements.txt` → `pip install`
2. 再 `COPY .`
3. 添加非 root 用户 `appuser`
4. 加 `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`
> 这部分作为 P1 优化，不阻塞首次上线。

### 4.3 前端构建
两种路径任选其一：
- **CI 构建**：在 GitHub Actions / Jenkins 中 `npm ci && npm run build`，把 `dist/` 推到服务器。
- **本机构建**：服务器上 `cd frontend && npm ci && npm run build`，产物自动被 nginx 容器挂载。

---

## 5. 反向代理与 HTTPS

`/etc/nginx/sites-available/industry.conf`（**宿主机** Nginx，而非容器内 nginx；容器 nginx 仅发静态文件）：

```nginx
server {
    listen 80;
    server_name app.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name app.example.com;

    ssl_certificate     /etc/letsencrypt/live/app.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/app.example.com/privkey.pem;

    client_max_body_size 100m;     # 文档上传

    # 前端静态
    location / {
        proxy_pass http://127.0.0.1:8080;   # frontend 容器 host 端口
    }

    # 后端 API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 流式接口（SSE）必备
        proxy_buffering off;
        proxy_read_timeout 600s;
    }
}
```

> 在 `docker-compose.prod.yml` 里把 frontend 端口映射到 `127.0.0.1:8080:80`、backend 映射到 `127.0.0.1:8000:8000`，仅本机回环可访问，由宿主 Nginx 统一对外。

证书：
```bash
sudo certbot --nginx -d app.example.com
```

---

## 6. 形态 B 差异（外部托管 DB）

- 不启动 compose 中的 `postgres`/`redis`，改填托管实例 host/port，凭证写到 `backend/.env.prod`
- Milvus、Elasticsearch 仍自托管（云上 Milvus/ES 成本较高，按需）
- 数据库迁移：用 `alembic upgrade head`（项目依赖已含 Alembic；如尚未引入版本，先生成基线 `alembic init` + `revision --autogenerate`）

---

## 7. 数据初始化

- [ ] Postgres 表：`Base.metadata.create_all` 已在 `app_main.py` 启动时执行，首次启动会自动建表。**生产建议改用 Alembic** 管理，避免裸 `create_all` 漂移（P1）。
- [ ] `docker/init-db/` 中已有初始化脚本 → 首次起 Postgres 时自动执行。
- [ ] Milvus collection：由后端按需创建（确认 `service` 层有初始化逻辑；如无，写一次性脚本）。
- [ ] 种子数据/知识库：从本地 `backend/data/` rsync 上去。

---

## 8. 上线步骤（Runbook）

```bash
# 1. 拉代码
cd /opt/industry/app && git pull

# 2. 构建前端
cd frontend && npm ci && npm run build && cd ..

# 3. 起中间件
docker compose -f docker-compose.yml up -d postgres redis etcd minio milvus elasticsearch

# 4. 等中间件 healthy
./start-services.sh status

# 5. 起后端 + 前端容器
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build backend frontend

# 6. 烟囱测试
curl -f https://app.example.com/api/health
curl -I https://app.example.com/

# 7. 看日志
docker compose logs -f backend
```

回滚：
```bash
git checkout <last-good-sha>
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build backend
```

---

## 9. 运维配置（上线当天必须做）

- [ ] **备份**：`pg_dump` 每日 cron 到 `/backup/pg/$(date +%F).sql.gz`，保留 14 天；MinIO/Milvus 数据卷每周 tar 备份
- [ ] **日志**：把 `backend` 容器日志接 `loki` 或至少 `logrotate`，保留 30 天
- [ ] **监控**：先用 `docker stats` + `node_exporter + Grafana` 看 CPU/内存/磁盘；后端加 `/metrics`（Prometheus FastAPI middleware，P1）
- [ ] **告警**：磁盘 >80%、内存 >85%、容器 restart>3 次/小时
- [ ] **防火墙**：`ufw` 只放 22/80/443，其余 deny；Postgres/Redis/Milvus 端口确认未对外暴露
- [ ] **密钥**：`.env.prod` 权限 `chmod 600`，提交到内网密钥管理（Vault / 1Password / 阿里云 KMS），**不要进 git**
- [ ] **CORS**：`backend/app/app_main.py` 中 `CORSMiddleware` 的 `allow_origins` 改成具体域名
- [ ] **DEBUG**：确认所有 `debug=True`、`reload=True` 关闭

---

## 10. CI/CD（建议，非首发必需）

GitHub Actions 流水线骨架：
1. `lint + test`（后端 pytest，前端 eslint + tsc）
2. `build`：构建前端产物 + 后端 Docker 镜像，推送到镜像仓库（阿里云 ACR / GHCR）
3. `deploy`：SSH 到服务器，`docker compose pull && up -d`

---

## 11. 风险与待确认项

1. **`Base.metadata.create_all`** 在生产首启会建表，但 schema 演进无版本控制 → 上线前定 Alembic 基线。
2. **DeepScout 本地 SFT** 默认 `DEEPSCOUT_USE_LOCAL_SFT=1`，无 GPU 节点必须显式置 0，否则启动会试图加载 7B 模型。
3. **Elasticsearch 8.11** 需要 `vm.max_map_count=262144`（`sysctl -w vm.max_map_count=262144` 并写入 `/etc/sysctl.conf`）。
4. **文件上传/附件存储**：确认 `backend/app/data/` 是否依赖本地磁盘；若多副本部署需切到 MinIO/OSS。
5. **定时任务调度器**（`scheduler_service`）在多副本下会重复触发 → 单副本先跑，扩容时引入分布式锁（Redis）或 leader 选举。
6. **数据库密码**当前 compose 是 `postgres123`，生产必须改。
7. **域名/备案**：国内服务器需备案后才能用 80/443。

---

## 13. 无服务器期间的并行准备（现在就能做）

服务器/域名到位前，下面这些**完全在本地仓库内完成**，到货当天就能 `git pull && up -d` 上线：

### 13.1 写好部署侧文件（仓库内提交）
- [ ] `docker-compose.prod.yml`（按 §4.1 模板）
- [ ] `docker/nginx/site.conf`（容器内 nginx 静态服务配置；监听 80，root `/usr/share/nginx/html`，SPA fallback `try_files $uri /index.html`）
- [ ] `deploy/nginx-host.conf`（宿主机 Nginx 反代模板，按 §5）
- [ ] `deploy/runbook.md`（精简版上线/回滚 cheat-sheet，从 §8 抽出）
- [ ] `.env.prod.example`（不含真值，标注每项含义）

### 13.2 配置模板化
- [ ] 把 `backend/.env` 中所有"占位 key"梳理成 `backend/.env.prod.example`
- [ ] `frontend/.env.production` 提交到仓库，`VITE_API_BASE=/api/`
- [ ] 确认 `app_main.py` 的 `CORSMiddleware` 支持从 env 读 `ALLOWED_ORIGINS`，没有就加上

### 13.3 强制关闭本地 SFT 的代码加固
- [ ] `app_main.py` 启动时，如果 `DEEPSCOUT_USE_LOCAL_SFT` 未显式为 `1`，**默认走 DashScope**，并在日志打印 `[startup] LLM provider = dashscope`
- [ ] 所有走本地 SFT 的入口加一个集中的 `is_local_sft_enabled()` 判断，避免某个分支绕过开关
- [ ] grep 一遍，确保没有"无条件 import torch"出现在主请求路径

### 13.4 生产镜像瘦身（首发只用 DashScope）
现状：`backend/app/Dockerfile` 直接装 `requirements.txt`，把 SFT 用的 torch/transformers/peft 全装进生产镜像。
建议拆 requirements：
- `backend/app/requirements.txt`（生产基线：fastapi、sqlalchemy、redis、pymilvus、openai、dashscope SDK、文档解析、爬虫……）
- `backend/app/requirements-sft.txt`（torch、transformers、peft、bitsandbytes、accelerate、datasets……）
然后 Dockerfile 只 `pip install -r requirements.txt`；GPU 镜像用 `Dockerfile.sft` 叠加安装。
> 收益：镜像从 ~6GB 降到 ~1.5GB，冷启动数十秒级，CI 也快。

### 13.5 数据库版本化（避免 `create_all` 漂移）
- [ ] `cd backend/app && alembic init migrations`
- [ ] 配 `alembic.ini` 用 `POSTGRES_*` 环境变量
- [ ] `alembic revision --autogenerate -m "baseline"` 生成基线
- [ ] `app_main.py` 里把 `Base.metadata.create_all(bind=engine)` 改为只在 `ENV=dev` 时执行；生产走 `alembic upgrade head`

### 13.6 健康检查与可观测性
- [ ] `GET /health`（不依赖 DB，立即返回）+ `GET /ready`（检查 DB/Redis/Milvus 连通）
- [ ] 后端 Dockerfile 加 `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`
- [ ] 接入结构化日志（JSON），方便后续 Loki/ELK

### 13.7 本地全链路演练
在本机用"生产 compose"跑一遍，**完全模拟生产**：
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file backend/.env.prod up -d --build
```
- [ ] 跑通注册/登录、对话、知识库上传、Deep Research（DashScope 路径）、行业资讯定时采集
- [ ] 确认 `DEEPSCOUT_USE_LOCAL_SFT=0` 时启动日志没有任何 torch 加载行为
- [ ] 用 `ab` 或 `hey` 压一下 `/api/chat` 流式接口，估算单机并发上限

### 13.8 选型与采购（异步推进，不阻塞编码）
- [ ] 明确部署区域（国内/海外）→ 决定是否需要 ICP 备案（国内 80/443 必须）
- [ ] 服务器规格：起步 8C/16G/200G SSD（阿里云 ecs.g7 / 腾讯云 S6 / 自建均可）
- [ ] 域名 + SSL（Let's Encrypt 免费即可）
- [ ] 注册并拿到生产 key：DashScope（必填）、博查搜索（必填）、其它按需

### 13.9 上线日清单（到货当天）
1. 服务器装 Docker/Nginx/certbot → 2. `git clone` → 3. 填 `.env.prod` 真值 →
4. `npm ci && npm run build` → 5. `docker compose ... up -d` → 6. certbot 签证书 →
7. 烟囱测试 → 8. 切流。

---

## 12. 当前任务清单（按顺序执行）

- [ ] 选定服务器规格 + 域名，DNS 解析就绪
- [ ] 安装 Docker/Nginx/certbot
- [ ] 在 `docker/` 下新增 `nginx/site.conf`（容器内 nginx 配置）
- [ ] 写 `docker-compose.prod.yml`（§4.1）
- [ ] 准备 `backend/.env.prod`、`frontend/.env.production`
- [ ] 优化后端 Dockerfile（分层、非 root、healthcheck）
- [ ] 引入 Alembic 基线
- [ ] 跑通 §8 的 Runbook，先跑形态 A
- [ ] 配 §9 的备份、日志、监控、防火墙
- [ ] 验收：注册/登录、对话、知识库上传、Deep Research、行业资讯采集
