# ECS 上线 Runbook

> 目标服务器：阿里云 ECS `47.84.206.6`（root SSH）
> 模式：DashScope-only，无 GPU，无本地 SFT
> 预期耗时：首次约 60 分钟（不含 ICP 备案）

照着从上往下敲。每章末尾的「✅ 验收」都通过再进下一章。

---

## 0. 上线前本地检查（在 Mac 上做）

```bash
cd /Users/weijun/Downloads/google\ download/industry_information_assistant

# 0.1 确认本地能跑通完整 compose
cd frontend && npm ci && npm run build && cd ..
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs backend --tail=30 | grep -i "LLM provider"
# 期望看到: [startup] LLM provider = dashscope

curl -fsS http://localhost:8000/hello && echo OK
curl -fsS http://localhost:8080/api/hello && echo OK
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

# 0.2 推到 git（线上从 git 拉，不要 scp）
git status
git add -A && git commit -m "chore: deploy ready"
git push origin main
```

✅ 验收：本地启动日志看到 `LLM provider = dashscope`，两条 curl 都返回 200。

---

## 1. 服务器初始化（首次登录后做一次）

```bash
ssh root@47.84.206.6
```

### 1.1 系统更新 + 基础工具
```bash
apt update && apt upgrade -y
apt install -y curl wget git vim ufw htop
timedatectl set-timezone Asia/Shanghai
```

### 1.2 装 Docker
```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
docker --version
docker compose version
```

### 1.3 ES 必需的内核参数（不设 ES 启动会失败）
```bash
sysctl -w vm.max_map_count=262144
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
```

### 1.4 装 Nginx + certbot（宿主层，做 TLS 终止 + 反代）
```bash
apt install -y nginx certbot python3-certbot-nginx
systemctl enable --now nginx
```

### 1.5 防火墙：只放 22 / 80 / 443
```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status
```

### 1.6 阿里云控制台同步配置
- 安全组放行：22 / 80 / 443（**仅这三个**，别开 8000 / 5432 / 6379 / 19530 / 9200）
- 给实例挂数据盘（≥100G，挂到 `/data`）

✅ 验收：`docker run --rm hello-world` 成功；`sysctl vm.max_map_count` 返回 262144。

---

## 2. 拉代码

```bash
mkdir -p /opt/industry && cd /opt/industry
git clone <你的 git 地址> app
cd app
```

> 如果是私库：先 `ssh-keygen -t ed25519` 在服务器生成 key，把 `~/.ssh/id_ed25519.pub` 加到 GitHub Deploy Keys。

✅ 验收：`ls /opt/industry/app/docker-compose.yml` 存在。

---

## 3. 配置 `.env.prod`

```bash
cd /opt/industry/app
cp backend/.env.prod.example backend/.env.prod
chmod 600 backend/.env.prod
vim backend/.env.prod
```

**必填项**（其他可保留占位）：
- `DASHSCOPE_API_KEY` ← 阿里云百炼真值
- `BOCHA_API_KEY` ← 博查真值
- `POSTGRES_PASSWORD` ← `openssl rand -hex 16` 生成强口令
- `JWT_SECRET_KEY` ← `openssl rand -hex 32` 生成
- `CORS_ORIGINS` ← 暂填 `*`，绑域名后改成具体域名

⚠️ 改完 `POSTGRES_PASSWORD` 后，**必须同步修改 `docker-compose.yml` 里 postgres 服务的 `POSTGRES_PASSWORD`**，否则两边对不上：

```bash
# 用相同的 POSTGRES_PASSWORD 改 compose
sed -i "s/POSTGRES_PASSWORD: postgres123/POSTGRES_PASSWORD: <your-strong-password>/" docker-compose.yml
```

✅ 验收：`grep DASHSCOPE_API_KEY backend/.env.prod` 看到真 key 不是 `REPLACE_ME`。

---

## 4. 构建前端 + 起所有容器

```bash
cd /opt/industry/app

# 4.1 装 Node + 构建前端
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
cd frontend && npm ci && npm run build && cd ..

# 4.2 切到 .env.prod 并起容器
export BACKEND_ENV_FILE=./backend/.env.prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 4.3 等 1-2 分钟让 Milvus / ES 起来
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

期望所有容器 `Up (healthy)`。如果 `industry_backend` 一直在 restart，看日志：
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs backend --tail=80
```

### 4.4 容器内冒烟测试
```bash
curl -fsS http://localhost:8000/hello && echo OK
curl -fsS http://localhost:8080/api/hello && echo OK
```

✅ 验收：两条 curl 都返回 `{"status":"success",...}`，日志里有 `[startup] LLM provider = dashscope`。

---

## 5. 宿主 Nginx 反代（先用 IP 跑通，无 HTTPS）

容器 frontend 监听宿主机 8080，宿主 Nginx 把 80 → 8080：

```bash
cat > /etc/nginx/sites-available/industry.conf <<'EOF'
server {
    listen 80;
    server_name _;          # 暂时接受任意 host

    client_max_body_size 100m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式
        proxy_buffering off;
        proxy_read_timeout 600s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/industry.conf /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

✅ 验收：浏览器打开 `http://47.84.206.6` 看到前端页面，调用 API 走 `/api/...` 正常。

---

## 6. 绑域名 + HTTPS（有域名后再做）

```bash
# 6.1 DNS：域名厂商把 A 记录指向 47.84.206.6（国内服务器需 ICP 备案后才能正常解析 80/443）

# 6.2 改 nginx server_name
sed -i 's/server_name _;/server_name app.example.com;/' /etc/nginx/sites-available/industry.conf
nginx -t && systemctl reload nginx

# 6.3 签证书（自动改 nginx 配置加 443 + 强制跳转）
certbot --nginx -d app.example.com --non-interactive --agree-tos -m you@example.com

# 6.4 收紧 CORS
sed -i 's|CORS_ORIGINS=.*|CORS_ORIGINS=https://app.example.com|' /opt/industry/app/backend/.env.prod
cd /opt/industry/app
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d backend  # 重启后端读新 env
```

certbot 会自动配置每月续期。

✅ 验收：`https://app.example.com` 拿到绿锁，HTTP 自动 301 到 HTTPS。

---

## 7. 上线后必做（30 分钟内完成）

### 7.1 Postgres 每日备份
```bash
mkdir -p /backup/pg
cat > /etc/cron.daily/pg-backup <<'EOF'
#!/bin/bash
docker exec industry_postgres pg_dump -U postgres industry_assistant | gzip > /backup/pg/$(date +\%F).sql.gz
find /backup/pg -name '*.sql.gz' -mtime +14 -delete
EOF
chmod +x /etc/cron.daily/pg-backup
# 立即跑一次验证
/etc/cron.daily/pg-backup && ls -lh /backup/pg/
```

### 7.2 Docker 日志轮转（防止占满磁盘）
```bash
cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "50m", "max-file": "5" }
}
EOF
systemctl restart docker
# 重启 docker 后所有容器会重启，用 compose 拉起来
cd /opt/industry/app
export BACKEND_ENV_FILE=./backend/.env.prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 7.3 阿里云消费告警
控制台 → 费用中心 → 资源包/账单 → 告警，设月度阈值（防 DashScope 被刷爆）。

✅ 验收：`/backup/pg/$(date +%F).sql.gz` 存在且非空。

---

## 8. 常用运维命令

```bash
cd /opt/industry/app
export BACKEND_ENV_FILE=./backend/.env.prod
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

$COMPOSE ps                            # 看状态
$COMPOSE logs -f backend               # 跟踪后端日志
$COMPOSE logs -f --tail=100 backend    # 最近 100 行
$COMPOSE restart backend               # 重启后端
$COMPOSE up -d --build backend         # 改了代码后重建后端
$COMPOSE down                          # 停所有（保留数据卷）
$COMPOSE down -v                       # 停所有 + 删数据卷（⚠️ 会清库）

docker stats                           # 实时资源占用
df -h                                  # 磁盘
docker system prune -af                # 清未用镜像（定期跑）
```

---

## 9. 部署/回滚 Runbook

### 部署新版本
```bash
cd /opt/industry/app
git pull
cd frontend && npm ci && npm run build && cd ..
export BACKEND_ENV_FILE=./backend/.env.prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build backend frontend
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=50 backend
curl -fsS https://app.example.com/api/hello
```

### 回滚到上一版本
```bash
cd /opt/industry/app
git log --oneline -5         # 找上一个 good sha
git checkout <good-sha>
cd frontend && npm ci && npm run build && cd ..
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build backend frontend
```

---

## 10. 故障排查速查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `industry_backend` 反复 restart | DB 连不上 / 缺依赖 | `$COMPOSE logs backend` 看 traceback |
| ES 起不来，`exit 78` | `vm.max_map_count` 没设 | `sysctl vm.max_map_count` 应为 262144 |
| Milvus 启动后 unhealthy | etcd 或 minio 没起 | `$COMPOSE ps` 看依赖 |
| `502 Bad Gateway` 浏览器报 | nginx → frontend 容器掉了 | `docker ps`、`systemctl status nginx` |
| API 慢 / 卡 5 分钟 | 误开了本地 SFT | 确认 `.env.prod` 里 `DEEPSCOUT_USE_LOCAL_SFT=0` |
| 磁盘爆了 | 日志或数据卷 | `du -sh /var/lib/docker/* \| sort -h` |
| DashScope 401 | key 错或没生效 | `$COMPOSE exec backend env \| grep DASHSCOPE` |

---

## 11. 待办（部署完成后再考虑）

- [ ] 引入 Alembic 替换 `Base.metadata.create_all`（schema 第一次要改时再做）
- [ ] 接入 Prometheus + Grafana 监控
- [ ] CI/CD：GitHub Actions 推镜像 + ssh 部署
- [ ] 多副本时引入分布式锁（scheduler_service 防重复触发）
