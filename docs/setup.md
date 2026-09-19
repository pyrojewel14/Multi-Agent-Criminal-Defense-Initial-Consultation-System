# 安装与启动

本文说明开发环境与已知 clean-clone 限制。它不是生产部署手册。

## 前置条件

- Python 3.10+ 和 `uv`
- Node.js 20+ 与 npm
- Redis 7+
- 可选：Ollama 或兼容的云端模型接口
- 可选：Docker Engine / Docker Desktop 与 Compose v2

## 1. 安装依赖

```bash
make install
```

该命令用 `uv` 创建 `backend/.venv`，安装后端依赖，并用 `npm ci` 安装前端锁定依赖。

## 2. 准备配置

```bash
cp backend/.env.example backend/.env
```

至少替换 `JWT_SECRET_KEY`。真实 `.env`、API key、token 与密码不得提交到 Git。配置分组包括：

- JWT 与 token 有效期；
- LLM / embedding provider；
- reranker 路径；
- Redis；
- SQLite、Chroma 与 MD5 store；
- 日志等级。

示例文件只包含占位值。需要管理员或律师账号时，应通过受信任的本地初始化流程创建临时账号，并通过无回显输入提供密码；不要把固定密码写入源码、文档或 shell history。

## 3. 模型配置

Ollama 示例：

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b
```

```dotenv
LLM_TYPE=OLLAMA
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_NAME=qwen3.5:0.8b
EMBED_MODEL_TYPE=OLLAMA
TEXT_EMBEDDING_MODEL_NAME=qwen3-embedding:0.6b
```

云端模式应只在本地 `.env` 中填写真实 key，并按照供应商文档确认 base URL、模型权限和数据处理要求。

## 4. 启动

先启动 Redis，再分别启动后端与前端：

```bash
redis-server
make run-backend
make run-frontend
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

默认前端地址是 `http://127.0.0.1:5173`。Vite 开发代理默认指向后端 8000 端口。

## 5. Clean-clone 数据限制

`backend/data/` 被 Git 与 Docker build context 共同排除。clean clone 和镜像不包含：

- `backend/data/law_knowledge/criminal_law_chapters.json`；
- SQLite 数据库；
- Chroma collection 与 MD5 store；
- reranker 或其他模型权重；
- 本地上传资料和派生文件。

应用可能完成基础启动，但真实法条验证与关键词补召回在缺少结构化法条库时不可用。连续 3 次非事实失败后，工作流会进入 degraded 人工审核路径。不要把基础健康检查解释为真实 RAG 已准备完成。

新增运行时代码实际读取的验证库前，需要审计来源、许可、版本、完整性与转换 manifest。旧文件可能仍存在于 Git/远端历史；当前树删除不等于历史清除，任何历史改写都需要单独授权。

## 6. Docker Compose

```bash
make compose-config
make compose-up
curl http://127.0.0.1:8000/health
make compose-down
```

Compose 只包含 backend 与 Redis。前端、Ollama、法条数据、Chroma 内容和模型权重不在镜像内。SQLite、Chroma、MD5 store 与模型缓存使用 `backend-runtime` 命名卷；Redis 使用 `redis-data`。

`make compose-config` 只验证配置解析。只有实际完成镜像 build、容器启动、健康检查和所需外部依赖调用后，才能分别声明这些步骤通过。

## 7. 验证

```bash
make test
npm --prefix frontend run typecheck
npm --prefix frontend run build
make eval
```

更细的风险分组见 [testing.md](testing.md)。不要运行或宣称历史上可能被系统终止的全量 pytest，除非在当前目标环境中取得完整输出。

## 常见问题

- Redis 连接失败：运行 `redis-cli ping`，并检查 `REDIS_HOST`、端口、DB 和密码。
- JWT 配置警告：替换示例密钥并重启后端。
- 模型不可用：确认 provider、base URL、模型名和网络权限。
- Chroma 为空：健康检查不会自动生成索引；需要经过授权的知识上传与 embedding 流程。
- reranker 缺失：链路会保留原召回顺序，但这不证明排序质量。
- Docker 无法访问宿主机模型：检查 `host.docker.internal`、模型监听地址和防火墙暴露范围。
