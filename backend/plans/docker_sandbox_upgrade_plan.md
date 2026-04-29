# CodeWizard 安全沙箱 Docker 化升级方案

> 目标：将 `backend/app/service/deep_research_v2/agents/wizard.py` 中
> `CodeWizard._execute_in_sandbox`（同进程伪沙箱）升级为基于 Docker 的强隔离沙箱，
> 保持对外接口不变，性能开销控制在 P50 < 300ms。

---

## 一、现状与问题

### 当前沙箱位置
- 类：`CodeWizard`（wizard.py:28）
- 入口：`_execute_code`（wizard.py:1014）→ `asyncio.to_thread(_execute_in_sandbox)`（wizard.py:1066）
- 实现：`_execute_in_sandbox`（wizard.py:1088）—— 在 FastAPI 主进程内 `exec()`

### 安全缺陷（必须解决）
| 编号 | 缺陷 | 风险等级 |
|------|------|----------|
| S1 | 同进程 `exec()`，可经元编程逃逸读取主进程内存（API key 等） | 🔴 严重 |
| S2 | 正则黑名单 `FORBIDDEN_PATTERNS`（wizard.py:313）可被字符串拼接、`getattr` 绕过 | 🔴 严重 |
| S3 | 无 CPU / 内存 / 时间限制，恶意循环可拖垮主服务 | 🟠 高 |
| S4 | 无文件系统隔离：`pd.read_csv('/etc/passwd')` 可读敏感文件 | 🟠 高 |
| S5 | 无网络隔离：`pd.read_csv('http://...')` 可外联 | 🟡 中 |
| S6 | matplotlib 全局 `rcParams` 跨次调用污染 | 🟡 中 |

---

## 二、目标架构（方案 B：常驻 Worker 池）

```
┌──────────────────────────────────────────────────────────────┐
│ FastAPI 主进程                                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ CodeWizard                                          │     │
│  │   _execute_code()                                   │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │   DockerSandboxClient   ──── unix socket / HTTP ─┐  │     │
│  └──────────────────────────────────────────────────┼──┘     │
└─────────────────────────────────────────────────────┼────────┘
                                                      │
   ┌──────────────────────────────────────────────────┴────────┐
   │ Docker network=none, read-only rootfs, tmpfs, no-new-priv │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
   │  │ worker1 │  │ worker2 │  │ worker3 │  │ workerN │       │
   │  │ python: │  │ python: │  │ python: │  │ python: │       │
   │  │ pd/np/  │  │ pd/np/  │  │ ...     │  │ ...     │       │
   │  │ plt/sns │  │ plt/sns │  │         │  │         │       │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
   └───────────────────────────────────────────────────────────┘
```

**关键设计点**：
- **Worker 池**：N=4 常驻容器（可配置），冷启动一次后复用
- **任务协议**：JSON over stdin/stdout（worker 内 `while True: code=read(); exec(); write(result)`）
- **每任务隔离**：每次执行后清理 worker 全局状态（重置 `plt`、清空用户变量）；连续 K 次后销毁重建（防止内存泄漏 / 状态残留）
- **强隔离参数**：`--network=none --read-only --tmpfs /tmp:size=64m --memory=512m --cpus=1.0 --pids-limit=64 --security-opt=no-new-privileges --cap-drop=ALL --user=65534:65534`

---

## 三、实施步骤

### Step 1：构建沙箱镜像（新增）
**新文件**：`backend/sandbox/Dockerfile`
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir \
    pandas==2.* numpy==1.* matplotlib==3.* seaborn==0.* \
    wordcloud jieba
RUN useradd -u 65534 -m sandbox && \
    mkdir -p /sandbox && chown sandbox:sandbox /sandbox
USER sandbox
WORKDIR /sandbox
COPY --chown=sandbox:sandbox runner.py /sandbox/runner.py
ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/sandbox/runner.py"]
```

**新文件**：`backend/sandbox/runner.py`（worker 端）
- 循环读取 stdin 一行 JSON：`{"id": "...", "code": "..."}`
- 在隔离的 globals 字典中 `exec()`
- 捕获 stdout/stderr/异常/matplotlib 图像（base64）
- 返回 stdout 一行 JSON：`{"id": "...", "success": ..., "output": ..., "error": ..., "charts": [b64...]}`
- 每次执行后 `plt.close('all')` + 清空用户全局变量 + 重置 `plt.rcParams`

**新文件**：`backend/sandbox/build.sh`
```bash
docker build -t codewizard-sandbox:latest backend/sandbox/
```

### Step 2：实现 Python 端客户端（新增）
**新文件**：`backend/app/service/deep_research_v2/agents/docker_sandbox.py`

类骨架：
```python
class DockerSandboxPool:
    def __init__(self, image: str, pool_size: int = 4, exec_timeout: int = 30):
        ...
    async def start(self): ...   # 预热 pool_size 个容器
    async def stop(self): ...    # 优雅关闭所有 worker
    async def execute(self, code: str) -> dict:
        # 1. 从空闲队列取一个 worker
        # 2. 写 JSON 任务到 worker stdin
        # 3. asyncio.wait_for(read_stdout, timeout=exec_timeout)
        # 4. 解析结果；若超时则 kill 容器并重建一个补回池
        # 5. 归还 worker（达到 max_uses 则销毁重建）

class DockerSandboxClient:
    """单容器执行（方案 A 兜底，便于 debug）"""
    async def execute(self, code: str) -> dict: ...
```

**关键实现要点**：
- 用 `asyncio.create_subprocess_exec("docker", "run", ...)` 直接管理子进程，不引入 `docker-py`（避免守护进程依赖）
- 通过 `--name codewizard-worker-{uuid}` 跟踪容器，启动时清理同名残留
- 超时处理：`asyncio.wait_for` + `docker kill {name}`
- 资源限制全部走命令行参数，避免依赖 daemon 配置

### Step 3：接入 CodeWizard（改造）
修改 `wizard.py`：

```python
# wizard.py 顶部
from .docker_sandbox import DockerSandboxPool

class CodeWizard(BaseAgent):
    # 类级单例池，由 app 启动时注入
    _sandbox_pool: Optional[DockerSandboxPool] = None

    @classmethod
    def set_sandbox_pool(cls, pool): cls._sandbox_pool = pool

    async def _execute_code(self, code: str) -> Dict[str, Any]:
        # 保留 _clean_code、_is_code_safe（作为前置过滤，廉价拦截）
        code = self._clean_code(code)
        if not self._is_code_safe(code):
            return {"success": False, "error": "Code contains forbidden operations", ...}

        if self._sandbox_pool and os.getenv("CODEWIZARD_SANDBOX", "docker") == "docker":
            return await self._sandbox_pool.execute(code)
        # Fallback：原 in-process 沙箱（仅本地开发 / Docker 不可用）
        return await asyncio.to_thread(self._execute_in_sandbox, code)
```

**保留**：`_execute_in_sandbox` 作为 fallback，供本地无 Docker 环境使用。

### Step 4：应用启动接线（改造）
修改 `backend/app/app_main.py`：
- 启动时（lifespan startup）：`pool = DockerSandboxPool(...); await pool.start(); CodeWizard.set_sandbox_pool(pool)`
- 关闭时（lifespan shutdown）：`await pool.stop()`
- 健康检查端点新增 `/health/sandbox`：返回池中可用 worker 数

### Step 5：配置与开关
新增环境变量（写入 `.env.example`）：
```
CODEWIZARD_SANDBOX=docker        # docker | inprocess
CODEWIZARD_SANDBOX_IMAGE=codewizard-sandbox:latest
CODEWIZARD_SANDBOX_POOL_SIZE=4
CODEWIZARD_SANDBOX_TIMEOUT=30
CODEWIZARD_SANDBOX_MEMORY=512m
CODEWIZARD_SANDBOX_CPUS=1.0
CODEWIZARD_SANDBOX_MAX_USES=50    # 每个 worker 最多复用次数
```

### Step 6：测试
新增 `backend/test/test_docker_sandbox.py`：

| 测试用例 | 目的 |
|----------|------|
| `test_normal_chart` | 正常生成 matplotlib 图 → 返回 base64 | 
| `test_fs_isolation` | `open('/etc/passwd')` 应失败 |
| `test_network_isolation` | `urllib.request.urlopen('http://1.1.1.1')` 应失败 |
| `test_memory_limit` | `[0]*10**10` 应被 OOM kill 后返回 error，主进程不挂 |
| `test_cpu_timeout` | `while True: pass` 应在 timeout 后被 kill |
| `test_escape_attempt` | `().__class__.__mro__[-1].__subclasses__()` 即使执行成功也无法影响主进程 |
| `test_state_isolation` | 任务 A 定义 `x=1`，任务 B 读不到 `x` |
| `test_pool_recovery` | kill 一个 worker，池能自动补一个 |
| `test_concurrent` | 并发 8 任务在 pool=4 下能正确串行/排队 |

### Step 7：渐进上线
1. **Phase 1**：合并代码，默认 `CODEWIZARD_SANDBOX=inprocess`，仅 staging 切 docker
2. **Phase 2**：staging 跑 1 周，监控 P50/P99 延迟、超时率、OOM 率
3. **Phase 3**：生产切换 `CODEWIZARD_SANDBOX=docker`，保留 inprocess 一周以便回滚
4. **Phase 4**：稳定后删除 `_execute_in_sandbox` 与相关 fallback 代码

---

## 四、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Docker 守护进程不可用 | 启动时检测，fallback 到 inprocess + 告警 |
| 冷启动延迟高 | 常驻 worker 池预热；考虑 `gVisor` runtime 进一步压缩 |
| Worker 内存泄漏（matplotlib） | 每 worker `max_uses=50` 后销毁重建 |
| 生产环境 Docker-in-Docker | 若 backend 本身在容器内，需挂载 `/var/run/docker.sock`（注意宿主机权限）或改用 sysbox / kata |
| 镜像安全更新 | 镜像构建集成到 CI，定期 rebuild + Trivy 扫描 |

---

## 五、估算

- 镜像构建：0.5d
- DockerSandboxPool 实现 + 协议：1.5d
- CodeWizard 接入 + lifespan 接线：0.5d
- 测试用例：1d
- 灰度 + 调优：1d
- **合计**：约 4.5 工作日

---

## 六、关键文件清单

**新增**：
- `backend/sandbox/Dockerfile`
- `backend/sandbox/runner.py`
- `backend/sandbox/build.sh`
- `backend/app/service/deep_research_v2/agents/docker_sandbox.py`
- `backend/test/test_docker_sandbox.py`
- `backend/plans/docker_sandbox_upgrade_plan.md`（本文档）

**修改**：
- `backend/app/service/deep_research_v2/agents/wizard.py`（替换 `_execute_code` 派发逻辑，保留 `_execute_in_sandbox` 作 fallback）
- `backend/app/app_main.py`（lifespan 启停 pool）
- `backend/requirements.txt`（无需新增依赖；不引入 docker-py）
- `.env.example` / `.env`（新增 `CODEWIZARD_SANDBOX_*` 变量）
