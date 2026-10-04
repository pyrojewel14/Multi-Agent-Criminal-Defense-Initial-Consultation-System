# 安装与启动

这篇带你配置开发环境、构建法条索引并启动前后端。
完成后可检查 API、打开页面，再用公开合成输入做联调。
只想先看流程，读 [Demo](demo.md)；使用前请看 [能力与限制](limitations.md)。

## 先选一条路线

- **看流程**：安装后端依赖后运行 [确定性 Demo](demo.md)，不需要模型或 Redis。
- **跑真实服务**：按下面 1–6 步操作。默认全量法条检索需要本地 Ollama embedding 和独立索引。
- **用容器**：先理解本地配置，再看本文的 Docker 小节；现有 Compose 还需要补充索引挂载。

下面所有 `make` 命令都从**仓库根目录**执行。需要进入 `backend/` 的命令用括号包裹，执行后不会改变终端的工作目录。

## 1. 安装依赖

| 依赖 | 用途 |
| --- | --- |
| Python 3.10+、`uv` | 后端；Makefile 默认创建 Python 3.12 环境 |
| Node.js 20+、npm | 前端依赖和开发服务器 |
| Redis 7+ | 服务启动时必须连接的缓存与观测投影 |
| Ollama | 下文使用的本地聊天模型和 embedding 服务 |
| Docker、Compose v2 | 可选的容器启动方式 |

```bash
make install
```

预期：创建 `backend/.venv`，安装后端依赖，前端 `npm ci` 成功退出。这里只安装依赖，不下载模型，也不生成索引。

## 2. 准备配置

```bash
cp backend/.env.example backend/.env
```

已有 `.env` 时直接编辑，避免覆盖原配置。至少替换 `JWT_SECRET_KEY`，确认 Redis 地址、端口与密码。JWT 是接口使用的签名访问令牌。

真实密钥、令牌和密码只放本地配置。公开注册只创建普通用户；管理员和律师账号的前置条件见 [接口说明](api.md#权限矩阵)。

## 3. 启动并配置模型

先启动 Ollama 服务；尚未安装下面两个模型时再拉取：

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3-embedding:0.6b
ollama list
```

预期：列表中出现两个模型。将 `backend/.env` 的对应字段设为：

```dotenv
LLM_TYPE=OLLAMA
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL_NAME=qwen3.5:0.8b
EMBED_MODEL_TYPE=OLLAMA
TEXT_EMBEDDING_MODEL_NAME=qwen3-embedding:0.6b
LAW_KNOWLEDGE_PROFILE=full
LAW_FULL_DEMO_ANNOTATIONS=on
```

embedding 把文字转换为向量，用于按语义检索。当前 `full` 索引入口只接受 Ollama embedding；聊天模型与 embedding 配置需分别核对，不能把云聊天配置直接当作全量检索配置。

**先知道这个限制**：上述小模型是默认配置示例。最近记录的本地六例 LawRef 复测未通过；配置正确不保证生成有效最终答案。详见 [模型证据](knowledge/lawref_response_protocol.md#本地小模型与新提示词复测2026-10-01)。

## 4. 构建并连接全量法条索引

法条 JSON 随仓库提供，向量索引需另行生成。首次构建：

```bash
make build-law-index
```

默认输出目录为 `backend/data/law_indexes/full-law`，collection（Chroma 中的一组文档）为 `criminal_law_full`。成功时脚本输出 manifest，即索引版本清单，其中应有：

```json
{
  "document_count": 504,
  "collection": "criminal_law_full",
  "embedding_engine": "ollama",
  "embedding_model": "qwen3-embedding:0.6b",
  "legal_review_completed": false
}
```

以上是需核对的字段，完整输出还含语料 hash、模型 digest、维度与向量 hash。digest 是模型内容的指纹。脚本读回全部文档、来源与向量后才写 manifest：

```text
backend/data/law_indexes/full-law/                 索引目录
backend/data/law_indexes/full-law.manifest.json    相邻清单文件
```

**目录和相邻 manifest 都必须不存在**。已有可用索引就复用；重建时指定新路径，不覆盖旧版本：

```bash
make build-law-index LAW_INDEX_DIR=backend/data/law_indexes/full-law-new
```

将实际目录和输出的 `embedding_model_digest` 写入 `backend/.env`：

```dotenv
# 此相对路径适用于 make run-backend 从 backend/ 启动。
LAW_FULL_INDEX_DIRECTORY=./data/law_indexes/full-law
LAW_FULL_INDEX_COLLECTION=criminal_law_full
LAW_FULL_EMBEDDING_DIGEST=<构建输出中的实际 embedding_model_digest>
```

从其他目录启动或运行评测时，请改用自己环境中的索引**绝对路径**。更换 embedding 模型、digest 或语料版本后，需构建匹配的新索引。维护步骤见 [全量索引说明](knowledge/full_criminal_law.md#构建更新与回滚)。

索引故障时关键词补召回仍会运行。它有候选时，工具记录 `partial_dependency_failure`，模型仍可继续核验；没有候选时记录 `dependency_failure`。这两个工具状态不能直接当作最终 `law_search_status`，路由规则见 [工作流](workflow.md#法条结果与失败窗口)。

默认 full 的混合召回、Qwen 重排、可选 HyDE 和设备配置见 [全量检索说明](knowledge/full_law_retrieval.md#配置与-hyde)。Memory 输入/摘要预算见 [Memory 配置](memory/README.md#配置与预算)，需按实际模型窗口配置。

## 5. 启动服务

分别在三个终端的仓库根目录运行：

```bash
redis-server
```

```bash
make run-backend
```

```bash
make run-frontend
```

另开终端检查：

```bash
redis-cli ping
curl --fail http://127.0.0.1:8000/health
```

预期分别为 `PONG` 和 `{"status":"healthy"}`。前端地址是 `http://127.0.0.1:5173`，接口文档是 `http://127.0.0.1:8000/docs`。

健康接口只返回服务状态，**不会检查模型或索引**。验证检索时，先把 `.env` 中的索引路径改成绝对路径，再从根目录运行：

```bash
backend/.venv/bin/python evaluation/run_full_eval.py live-rag \
  --query-mode semantic --output evaluation/results/full-rag-new.json
```

预期：结果 JSON 列出逐例检索结果和汇总，全部满足固定案例契约时退出码为 0。若失败，先看逐例错误和配置；不要仅凭结果文件存在判定成功。该入口调用真实 embedding 和索引，仍不验证聊天模型或完整咨询闭环。

## 6. 跑验证

```bash
make test
npm --prefix frontend run typecheck
npm --prefix frontend run build
make eval
```

预期：测试报告列出通过/失败数；类型检查和构建退出码为 0；离线评估生成 `evaluation/results/` 下的结果。`make eval` 固定使用六条 `snapshot`，不会评测默认全量检索。分组说明见 [测试](testing.md)，指标解释见 [评估](evaluation.md)。

## 启动时检查哪些数据

应用在数据库与 Redis 初始化前，离线校验所选法条语料及启用的 demo 标注。可单独运行：

```bash
(cd backend && .venv/bin/python -c \
  "from app.knowledge.law_knowledge import preflight_law_knowledge; preflight_law_knowledge(); print('preflight OK')")
```

预期最后输出 `preflight OK`；数据缺失或校验失败时抛出错误。它检查结构、来源和标注契约，不连 Ollama，不校验 Chroma 索引。

默认 `full` 的 505 条记录、504 条有效正文和 38 条演示覆盖标注，以及 `snapshot` 的六条历史回归入口，统一说明在 [法条与检索](rag.md#语料与标注)。数据库、索引、模型权重和上传资料不随仓库分发。

## Docker Compose

```bash
make compose-config
make compose-up
curl --fail http://127.0.0.1:8000/health
make compose-down
```

这些步骤依次检查配置解析、构建并启动、HTTP 响应、停止服务。某一步通过只证明该步骤。

现有 Compose 包含 backend 与 Redis；法条 JSON 进入镜像，前端、Ollama、向量索引和模型权重需要另行准备。`host.docker.internal` 用于访问宿主机的 **Ollama HTTP 服务**，不能读取宿主机上的索引目录。

要在容器里使用宿主机已构建的索引，可创建本地 `docker-compose.override.yml`，同时挂载目录和相邻 manifest：

```yaml
services:
  backend:
    environment:
      LAW_FULL_INDEX_DIRECTORY: /app/law-index/full-law
      LAW_FULL_INDEX_COLLECTION: criminal_law_full
      LAW_FULL_EMBEDDING_DIGEST: "<与宿主机 Ollama 匹配的实际 digest>"
    volumes:
      - ./backend/data/law_indexes:/app/law-index
```

示例采用可写挂载：Chroma 的读取可能更新 SQLite 文件。先停止使用同一索引的本地后端，再供容器使用；此示例不是多进程共享索引方案。改过目录名时同步修改容器内路径。override 文件已被 Git 忽略。

业务 SQLite、LangGraph checkpoint、通用 Chroma、MD5 store 与模型缓存放在 `backend-runtime` 命名卷，Redis 使用 `redis-data`。checkpoint 保存流程状态；保留卷和 checkpoint 文件才能恢复中断位置。详见 [存储与恢复](architecture.md#存储与恢复)。

## 常见问题

| 现象 | 先检查什么 | 处理方式 |
| --- | --- | --- |
| 工具轨迹有 `dependency_failure` / `partial_dependency_failure` | 索引路径、manifest、collection、模型 digest、Ollama 连通性 | 按第 4 步修配置；仅在索引不匹配时用新目录重建 |
| 反复 `duplicate_call`、`agent_timeout`、`max_steps` | LawRef 最终响应与模型配置 | 读 [响应诊断](knowledge/lawref_response_protocol.md)；索引成功不代表模型决策成功 |
| `text_only` 后转人工 | 条文标注是否有覆盖资格 | 这是标注审查路径；补充案件事实不能修复标注缺口 |
| Redis 连接失败，应用无法启动 | `redis-cli ping`、地址、端口、密码 | 修 Redis 配置；它是启动硬依赖 |
| 法条预检失败 | 报错中的文件和字段 | 恢复匹配的公开资产，检查自定义语料/标注路径 |
| 模型不可达 | base URL、模型名、服务状态 | 容器使用宿主机地址，本地使用 loopback 地址 |
| reranker 缺失、超时或繁忙 | `RERANKER_MODEL_PATH` 与 full 专用状态 | full 回退融合顺序；snapshot 回退召回顺序，详见 [全量检索说明](knowledge/full_law_retrieval.md) |

## 边界

这份手册用于开发联调。构建输入是带版本的法律文本与项目标注，标注没有法律专家背书。服务启动、检索成功和确定性测试均不能证明法律适用正确、模型稳定或生产可用；完整说明见 [能力与限制](limitations.md)。
