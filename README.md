# Multi-Agent Criminal Defense Initial Consultation System

一个面向刑事辩护初期咨询场景的 LLM / Agent 应用工程项目。项目用 FastAPI 提供认证、会话、知识库和咨询接口，用 LangGraph 编排多 Agent 工作流，用 ChromaDB、BM25、HyDE 和 rerank 组成法律资料检索链路，并在流程中加入知情同意、PII 脱敏、高风险表达检测和律师审核断点。

本仓库适合作为求职展示项目使用：它不是“法律意见自动生成器”，而是一个可运行、可解释、可被面试追问的 Agent 应用样例。下文所有功能描述都以当前代码为准。

## 项目背景

刑事案件初期咨询常见问题是：用户叙述零散、关键信息缺失、法条引用容易幻觉、自动化系统不能直接越过律师审核给出确定性结论。本项目把初期咨询拆成结构化工作流：

1. 先完成权利义务告知和用户知情同意。
2. 再通过事实挖掘 Agent 把自然语言叙述提取成案件事实字段。
3. 结合 RAG 和本地 JSON 法条知识库召回、验证相关法条。
4. 用覆盖度驱动追问，信息足够后生成风险评估和服务方案草案。
5. 最终进入律师审核或高风险人工介入。

核心代码位置：

- FastAPI 入口：[backend/main.py](backend/main.py)
- LangGraph 工作流：[backend/app/orchestrator/workflow.py](backend/app/orchestrator/workflow.py)
- Agent 节点：[backend/app/agents](backend/app/agents)
- RAG 服务：[backend/app/rag](backend/app/rag)
- 认证与权限：[backend/app/security](backend/app/security)
- v1 路由：[backend/app/v1/router](backend/app/v1/router)

## 核心功能

- 多 Agent 流程：LangGraph 编排咨询、评估、规划与人工审核。
- 知情同意门禁：用户确认告知后才进入事实收集。
- 结构化事实提取：通过 tool calling 提取案情要素。
- 覆盖度追问：事实覆盖率低于 `0.8` 时自动追问。
- 法条检索：RAG 召回并结合本地法条库验证、补充。
- 风险介入：识别高风险表达并触发人工处理。
- 律师审核：支持批准或退回指定流程节点。
- 认证权限：JWT 双令牌、角色鉴权与记录过滤。
- 知识库：支持文档上传、查看、删除及 ChromaDB 入库。
- 会话历史：SQLite 持久化，Redis / 内存缓存工作流状态。

## 技术栈

后端：

- FastAPI、Uvicorn、Pydantic
- SQLAlchemy async ORM、SQLite、Redis
- LangGraph、LangChain
- ChromaDB、BM25Retriever、EnsembleRetriever
- 阿里云百炼 / Ollama 兼容的 LLM 与 embedding 工厂
- PyJWT、passlib、RBAC 依赖注入
- pytest、pytest-asyncio、ruff

## 系统架构

```text
FastAPI backend
  ├─ /api/v1/auth           注册、登录、刷新、当前用户、登出
  ├─ /api/v1/sessions       Agent 会话创建、消息、同意、状态、审核、关闭
  ├─ /api/v1/consultations  咨询历史、消息、分配律师、状态更新
  ├─ /api/v1/knowledge      文档上传、检索库管理
  ├─ /api/v1/users          用户管理
  ├─ /api/v1/lawyer(s)      律师相关接口
  └─ /api/v1/sessions/{id}/ws WebSocket 消息和心跳

LangGraph StateGraph
  Receptionist
    -> FactDigger
    -> WaitForUser -> LawRef -> FactDigger
    -> RiskAssessor -> ServicePlanner -> HumanReview
    -> HumanAlert

Data and retrieval
  ├─ SQLite: users, consultations, consultation_messages
  ├─ Redis: session:{session_id} 状态缓存
  ├─ LangGraph MemorySaver: workflow checkpoint
  ├─ ChromaDB: 文档向量检索
  └─ backend/data/law_knowledge/criminal_law_chapters.json: 法条结构化知识库
```

## Agent 工作流说明

工作流定义在 `ConsultationOrchestrator._build_workflow()`：

```text
START
  -> receptionist
  -> check_consent
       continue -> fact_digger
       end      -> END

fact_digger
  -> check_facts_sufficient
       complete -> risk_assessor -> service_planner -> human_review
       loop     -> wait_for_user -> law_ref -> fact_digger
       alert    -> human_alert -> END
       max_loop -> risk_assessor

human_review
  -> lawyer_decision
       approved     -> END
       revise_facts -> fact_digger
       revise_risk  -> risk_assessor
```

关键控制点：

- `interrupt_after=["receptionist", "wait_for_user", "human_review", "human_alert"]`：这些节点执行后暂停，等待外部用户或律师输入。
- `COVERAGE_THRESHOLD = 0.8`：事实覆盖度达到阈值后进入风险评估。
- `fact_law_loop_count >= 10`：防止 FactDigger / LawRef 无限循环，达到上限后强制进入风险评估。
- `alert_triggered`：高风险表达会从 FactDigger 分流到 HumanAlert。
- `lawyer_decision`：律师审核决定控制最终批准或回退到前序节点。

更详细的节点输入输出见 [docs/architecture.md](docs/architecture.md)。

## RAG 检索流程

RAG 主链路由 `RagService`、`HybridRetriever`、`VectorStoreService` 和 `LawRef` 共同完成：

```text
结构化事实
  -> 拼接 behavior_sequence / consequence 为查询
  -> HyDE 生成假设性回答
  -> ChromaDB 向量检索 + BM25 混合召回
  -> reranker 重排序
  -> 文档摘要
  -> LawRef 从文档中提取法条编号
  -> JSON 法条库精确验证并增强元数据
  -> JSON 关键词检索补召回
  -> LLM 提取结构化法律分析
  -> applied_laws / element_to_law_mapping 写回 state
```

需要注意：当前工作流主要消费的是检索到的 `documents` 和法条验证结果，而不是把 RAG summary 直接作为法律结论。未通过 JSON 法条库验证的 RAG 结果会标记为 `rag_unverified`，事实覆盖度计算会跳过这类结果。

2026-07-13 首次验收保留了“空 Chroma + Ollama 502”的失败记录；修复 loopback 请求误走系统代理并通过现有 service 链路入库后，2026-07-14 五条样例已实际运行 HyDE、Chroma + BM25/Ensemble、去重、reranker 与 JSON 验证子链。两次运行条件、真实 top-k 和仍存在的边界见 [docs/rag.md](docs/rag.md) 和 [demos/rag/](demos/rag/)。

## 本地启动方式

后端：

推荐使用 `uv`：

```bash
cd backend
cp .env.example .env
# 编辑 .env，至少配置 JWT_SECRET_KEY 和 LLM / embedding 相关变量
uv sync
uv run python main.py
```

也可以使用 `requirements.txt`：

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

服务默认启动在 `http://localhost:8000`，接口文档为 `http://localhost:8000/docs`。

Redis：

```bash
redis-server
```

或使用本机服务管理器启动 Redis。后端启动时会执行 `init_redis()`，Redis 不可用会导致启动失败。

测试：

```bash
cd backend
uv sync --extra dev
uv run pytest
```

小规模离线评估：

```bash
cd backend
uv run python ../evaluation/run_eval.py
```

指标定义、当前基线结果和边界见 [docs/evaluation.md](docs/evaluation.md)。

## 环境变量说明

见 [backend/.env.example](backend/.env.example)。常用变量如下：

| 变量 | 说明 | 默认 / 示例 |
| --- | --- | --- |
| `JWT_SECRET_KEY` | JWT 签名密钥，生产环境必须替换 | `CHANGE_ME_IN_PRODUCTION` |
| `JWT_ALGORITHM` | JWT 算法 | `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | access token 有效期 | `15` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | refresh token 有效期 | `7` |
| `LLM_TYPE` | 聊天模型后端 | `ALIYUN` 或 `OLLAMA` |
| `ALIYUN_ACCESS_KEY_SECRET` | 阿里云百炼 API key | 无默认可用值 |
| `ALIYUN_BASE_URL` | OpenAI 兼容接口地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ALIYUN_MODEL_NAME` | 阿里云聊天模型 | `qwen3-max` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_MODEL_NAME` | Ollama 聊天模型 | `qwen3.5:0.8b` |
| `EMBED_MODEL_TYPE` | embedding 后端 | `ALIYUN` 或 `OLLAMA` |
| `ALIYUN_EMBED_MODEL_NAME` | 阿里云 embedding 模型 | `text-embedding-v4` |
| `OLLAMA_EMBED_MODEL_NAME` | `.env.example` 中的 Ollama embedding 名称 | `qwen3-embedding:0.6b` |
| `DATABASE_PATH` | SQLite 数据库路径 | `./data/chat_history.db` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | Redis 连接配置 | `localhost` / `6379` / `3` |
| `RERANKER_MODEL_PATH` | reranker 本地模型路径 | `./data/models/Qwen/Qwen3-Reranker-0.6B` |
| `LOG_LEVEL` | 全局日志级别 | `INFO` |

## API 示例

注册：

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "client1",
    "password": "Passw0rd!",
    "email": "client1@example.com",
    "real_name": "测试用户"
  }'
```

登录：

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "client1", "password": "Passw0rd!"}'
```

创建咨询会话：

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "client1",
    "user_type": "suspect",
    "initial_message": "我想咨询一个刑事案件",
    "source": "web"
  }'
```

确认知情同意：

```bash
curl -X POST http://localhost:8000/api/v1/sessions/$SESSION_ID/confirm-consent \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "'$SESSION_ID'",
    "consent_given": true,
    "consent_timestamp": "2026-07-03T10:00:00Z",
    "consent_version": "v1",
    "identity_info": {"role": "suspect"}
  }'
```

发送案件描述：

```bash
curl -X POST http://localhost:8000/api/v1/sessions/$SESSION_ID/message \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "'$SESSION_ID'",
    "content": "事情发生在杭州，我和对方发生争执，对方先动手，我推了他一下，他摔倒后报警了。",
    "message_type": "text"
  }'
```

律师审核：

```bash
curl -X PUT http://localhost:8000/api/v1/sessions/$SESSION_ID/review \
  -H "Authorization: Bearer $LAWYER_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "revise_facts",
    "feedback": "请补充伤情鉴定和是否取得谅解。"
  }'
```

## 示例输入输出

输入：

```text
我家人因为和别人打架被带走了。事情发生在杭州，对方说受伤了，但我们不清楚有没有鉴定。
```

可能输出：

```text
为了更准确地分析案件，请您补充以下信息：

1. 事件发生的具体时间、地点和参与人员分别是什么？
2. 双方冲突过程中具体有哪些行为？是否有人使用工具？
3. 对方伤情是否已有医院诊断或伤情鉴定？
4. 公安机关目前采取了什么措施，是否已刑事拘留？
```

高风险输入示例：

```text
我想让朋友统一口径，不要把关键事实说出去。
```

可能输出：

```text
为保护您的权益，此部分内容建议直接与律师单独沟通。
```

说明：真实输出取决于配置的 LLM、知识库内容、当前工作流状态和已收集事实。

## 当前限制与后续优化方向

当前限制：

- LangGraph 使用 `MemorySaver` 作为 checkpointer，服务重启后的断点恢复能力有限；Redis 是额外缓存，不等价于完整工作流持久化。
- `Consultation` 表包含事实、法条、风险、报告字段，但当前主流程主要保存消息和部分状态，数据库字段与工作流 state 仍可进一步打通。
- WebSocket 端点在握手前校验 access token 和会话所有权；浏览器客户端通过 `?token=` 传递 token，当前只允许会话所有者的 `client` 连接。
- RAG 检索依赖知识库内容和 embedding / reranker 模型配置；未通过 JSON 法条库验证的结果不会被当作可靠构成要件来源。
- 当前没有数据库 migration 工具，表结构由 SQLAlchemy metadata 在启动时创建。
- 评估已有离线 MVP，但还没有覆盖真实 LLM 输出质量、长期对话一致性和人工审核质量的完整指标体系。
- 后端测试位于 `backend/tests/`，已纳入版本控制，并由受限 GitHub Actions 分组执行。

后续优化：

- 把 LangGraph checkpoint 切换到可持久化后端，并统一 DB / Redis / checkpoint 的状态边界。
- 远端验证并逐步扩大现有受限 CI，再补 Alembic migration、Docker Compose 和端到端冒烟测试。
- 为 RAG 建立标注集，评估 recall@k、rerank 命中率、法条验证通过率、未验证结果占比。
- 增加律师审核操作的审计日志和报告版本管理。
- 补 WebSocket token 传输加固、速率限制和消息持久化。
- 为高风险检测增加更系统的测试样本，降低误报与漏报。

## 文档

- [docs/demo.md](docs/demo.md)：三条标准 case、完整可复现流程、FastAPI curl 路径和演示边界。
- [docs/architecture.md](docs/architecture.md)：LangGraph 工作流、节点职责、条件边和人工介入。
- [docs/rag.md](docs/rag.md)：真实 RAG 调用链、实现状态、fallback、运行证据和简历表述边界。
- [docs/evaluation.md](docs/evaluation.md)：30 条正式离线评估集、指标定义、实际结果和失败边界。
- [docs/testing.md](docs/testing.md)：测试分组、Phase 5 契约映射、受限 CI 和已知限制。
- [docs/api.md](docs/api.md)：核心接口、JWT/RBAC 权限、错误码、Redis/SQLite 边界和可复制 curl。
- [demos/](demos/)：咨询 Demo case、RAG 查询、历史失败记录和 2026-07-14 live 结果。
- [evaluation/](evaluation/)：唯一活动评估入口及旧版 MVP 历史报告。

## 许可证

见 [LICENSE](LICENSE)。
