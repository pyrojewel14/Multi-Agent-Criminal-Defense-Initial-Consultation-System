# Multi-Agent Criminal Defense Initial Consultation System

> 面向刑事辩护初期咨询的多 Agent AI 应用工程原型：用 LangGraph 把知情同意、结构化事实收集、法律资料检索、风险评估与律师审核编排成可中断、可恢复的工作流。

它不是自动出具法律意见的产品，而是一个用于展示 Agent 工作流、RAG 可靠性边界、后端权限与前端交互的可运行工程项目。

## 30 秒看懂项目

| 能力 | 当前实现 | 可核验证据 |
| --- | --- | --- |
| 多 Agent 编排 | 8 个 LangGraph 节点、条件边、`MemorySaver` checkpoint 和 4 个人工/用户中断点 | [workflow.py](backend/app/orchestrator/workflow.py) · [state](backend/app/state/consultation_state.py) |
| 结构化事实 | FactDigger 通过 Tool Calling 提取 10 类案件事实，并按法条构成要件覆盖度继续追问 | [fact_digger.py](backend/app/agents/fact_digger.py) · [extract_case_facts](backend/app/tools/fact_tools.py) |
| RAG 与法条验证 | HyDE、Chroma 向量检索、条件式 BM25 混合召回、reranker、JSON 法条验证与关键词补召回 | [rag_service.py](backend/app/rag/rag_service.py) · [hybrid_retriever.py](backend/app/rag/retrievers/hybrid_retriever.py) · [law_ref.py](backend/app/agents/law_ref.py) |
| Human-in-the-loop | 知情同意门禁、高风险转人工、律师批准或退回事实/风险节点 | [workflow.py](backend/app/orchestrator/workflow.py) · [human_alert.py](backend/app/agents/human_alert.py) |
| 应用后端 | FastAPI、JWT 双令牌、RBAC、SQLite async ORM、Redis 状态缓存、统一错误 envelope | [main.py](backend/main.py) · [JWT/RBAC](backend/app/security) · [v1 routes](backend/app/v1/router) |
| React 前端 | client 注册/登录、双 ID 会话、知情同意、消息与状态面板；lawyer/admin 审核工作台 | [frontend/src](frontend/src) · [前端测试](frontend/src/pages/ClientWorkspace.test.tsx) |
| 小型评估 | 30 条 input-only deterministic baseline，覆盖事实、法条关键词、高风险、追问、拒答和免责声明契约 | [cases.jsonl](evaluation/cases.jsonl) · [run_eval.py](evaluation/run_eval.py) |

## 真实界面

以下截图来自同一条本地虚构案件的真实 FastAPI + Redis + Ollama + Chroma + React 路径，不是 mock 或直接改写 workflow state。`session_id=2a1f37ef-a809-40c2-917a-fe2a9e932e8c`、`consultation_id=9efa9a69-9491-44f3-a066-0cf16e1a8b43` 自然经过结构化事实、RAG 候选法条、风险评估、ServicePlanner 报告草案与 HumanReview；管理员再分配律师，律师从前端调用 workflow review endpoint 批准，client 最终看到完成状态。它证明的是工程链路可运行，不代表模型输出具有法律正确性。

律师批准后的 client 完成态（桌面端 1440 × 900）：

![真实 client 律师批准后桌面视图](assets/screenshots/frontend-fixed-client-approved-desktop.png)

同一完成态（移动端 390 × 844）：

![真实 client 律师批准后移动视图](assets/screenshots/frontend-fixed-client-approved-mobile.png)

早期 client 追问与结构化事实截图仍保留在 [assets/screenshots](assets/screenshots)，用于区分第一轮 MVP 证据与本次自然闭环证据。

## 系统架构

```mermaid
flowchart LR
    User["咨询用户 / 律师 / 管理员"] --> Frontend["React + TypeScript 前端"]
    Frontend -->|"REST + 状态轮询"| API["FastAPI /api/v1"]
    ClientWS["client WebSocket"] -->|"access token + session owner"| API

    API --> Auth["JWT access/refresh + RBAC"]
    API --> Sessions["会话与咨询 service"]
    API --> Knowledge["知识库管理"]
    Sessions --> Graph["LangGraph StateGraph"]
    Sessions --> Redis["Redis session cache"]
    Sessions --> SQLite["SQLite + SQLAlchemy async"]
    Graph --> Checkpoint["MemorySaver checkpoint"]
    Graph --> Agents["Receptionist / FactDigger / WaitForUser / LawRef / RiskAssessor / ServicePlanner / HumanReview / HumanAlert"]
    Knowledge --> Chroma["Chroma persistent collection"]
    Agents --> RAG["HyDE + Chroma / BM25 + reranker"]
    RAG --> Chroma
    RAG --> LawJSON["本地结构化法条 JSON 验证库"]
```

存储边界不是一套强一致事务：LangGraph checkpoint 在进程内，Redis 是会话缓存，SQLite 保存用户、咨询与 HTTP 消息等数据；完整 workflow state 目前不能仅靠 SQLite 恢复。

## LangGraph 工作流

源码拓扑定义在 [`ConsultationOrchestrator._build_workflow()`](backend/app/orchestrator/workflow.py)。

```mermaid
flowchart TD
    Start(["START"]) --> Receptionist["receptionist"]
    Receptionist --> Consent{"check_consent"}
    Consent -->|"未同意"| End(["END"])
    Consent -->|"已同意"| FactDigger["fact_digger"]

    FactDigger --> Facts{"check_facts_sufficient"}
    Facts -->|"alert_triggered"| HumanAlert["human_alert"]
    HumanAlert --> End
    Facts -->|"coverage < 0.8"| WaitForUser["wait_for_user"]
    WaitForUser -->|"外部输入后恢复"| LawRef["law_ref"]
    LawRef --> FactDigger
    Facts -->|"coverage >= 0.8 或循环 >= 10"| RiskAssessor["risk_assessor"]
    RiskAssessor --> ServicePlanner["service_planner"]
    ServicePlanner --> HumanReview["human_review"]

    HumanReview --> Decision{"lawyer_decision"}
    Decision -->|"approved"| End
    Decision -->|"revise_facts"| FactDigger
    Decision -->|"revise_risk"| RiskAssessor
    Decision -->|"尚无有效决定"| HumanReview
```

`receptionist`、`wait_for_user`、`human_review`、`human_alert` 执行后会中断。高风险分支到达 `END` 表示自动图停止，不表示外部律师工单或通知已发送。

## RAG 检索与验证流程

```mermaid
flowchart TD
    Facts["facts_structured + user_id"] --> Query["拼接行为与后果为 query"]
    Query --> Keyword["本地 JSON 关键词补召回"]
    Query --> HyDE["HyDE 假设文本；失败回退原 query"]
    HyDE --> Hybrid["按 user_id / public 范围建立 retriever"]
    Hybrid --> Vector["Chroma 向量召回"]
    Hybrid -. "query < 200 且有语料" .-> BM25["从同范围 Chroma 文档临时构建 BM25"]
    Vector --> Ensemble["向量结果或 Ensemble 融合"]
    BM25 --> Ensemble
    Ensemble --> Dedupe["正文前缀 MD5 去重"]
    Dedupe --> Rerank["reranker；失败保留原顺序"]
    Rerank --> Article["从候选正文提取法条编号"]
    Article --> Verify["本地法条 JSON 精确验证与元数据增强"]
    Verify --> Mark["标记 rag_verified / rag_unverified"]
    Mark --> Merge["与 JSON 关键词结果合并去重"]
    Keyword --> Merge
    Merge --> Extract["LLM 提取结构化法律分析；失败使用候选回退"]
    Extract --> State["applied_laws + element_to_law_mapping"]
    State --> Coverage["rag_unverified 不参与事实覆盖度计算"]
```

内置法条 JSON 是验证与规则补召回数据，不等于已填充的 Chroma 向量库。真实 RAG 还依赖可用的 embedding、已入库文档和本地 reranker；[`demos/rag/results/2026-07-14-ollama.json`](demos/rag/results/2026-07-14-ollama.json) 保存了一次 5 条查询的 live 子链记录，仅作运行证据，不是质量指标。

## 快速启动

### 前置条件

- Python `>=3.10`；统一 Make 入口默认用 `uv` 创建 Python 3.12 环境。
- Node.js、npm、Redis 7+。
- Ollama 本地模型，或可用的阿里云百炼兼容接口。

### Make 入口

```bash
cp backend/.env.example backend/.env
# 修改 JWT_SECRET_KEY，并选择 Ollama 或阿里云聊天 / embedding 配置

make install
redis-server              # 终端 1；当前后端启动硬依赖
make run-backend          # 终端 2：http://127.0.0.1:8000
make run-frontend         # 终端 3：http://127.0.0.1:5173
```

验证入口：

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}

make test                 # 定向后端回归 + 全部前端测试，不是完整 pytest
make eval                 # 30 条 deterministic baseline
```

### 通用 `uv` 后端环境

```bash
cd backend
cp .env.example .env
# 至少替换 JWT_SECRET_KEY，并配置 LLM / embedding
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python main.py
```

前端可单独运行：

```bash
cd frontend
npm ci
npm run dev
```

配置模板见 [backend/.env.example](backend/.env.example)，统一命令见 [Makefile](Makefile)。

### 本地演示角色初始化

公开注册始终只能创建 `client`。本地演示如需 `admin` / `lawyer`，先用受控 CLI 初始化管理员，再由管理员调用既有 API 创建律师与分配案件；CLI 不接受命令行明文密码、要求显式本地确认，并拒绝 `APP_ENV=production`：

```bash
cd backend
read -s PHASE9_ADMIN_PASSWORD && export PHASE9_ADMIN_PASSWORD
.venv/bin/python -m examples.phase9_demo_admin \
  --confirm-local-demo \
  --username phase9_admin \
  --password-env PHASE9_ADMIN_PASSWORD
unset PHASE9_ADMIN_PASSWORD
```

管理员登录后使用 `POST /api/v1/lawyers/` 创建临时律师账号，再用 `POST /api/v1/consultations/assign` 提交 `consultation_id` 与 `lawyer_id`。两步均受 admin JWT 保护；演示密码只放在当前终端环境或无回显输入中，不写入仓库。律师主要审核动作使用 workflow `session_id` 调用 `PUT /api/v1/sessions/{session_id}/review`，不能用 SQLite-only 报告更新替代 LangGraph 恢复。

### 模型、数据与 Docker 边界

- Ollama 默认示例使用 `qwen3.5:0.8b` 和 `qwen3-embedding:0.6b`；阿里云模式需自行提供 key，不能把真实密钥写回仓库。
- SQLite 表由 lifespan 创建；Redis 连接失败会阻止后端启动。
- Chroma 初始可以为空，需通过知识库链路实际入库；健康检查不准备向量数据，也不调用 LLM。
- reranker 权重不随 Git 或 Docker 镜像分发；模型缺失或加载失败时保留原召回顺序。
- [docker-compose.yml](docker-compose.yml) 只包含 backend + Redis。Ollama / 阿里云、reranker 权重和前端均在 Compose 外。
- `docker compose config --quiet` 已验证；镜像 build、Compose up 和容器健康检查尚未完整验证，不能把配置校验视为部署成功。

## API 最小闭环

基础地址为 `http://127.0.0.1:8000/api/v1`，OpenAPI 页面为 `http://127.0.0.1:8000/docs`。

### 1. 注册与登录

公开注册只创建 `client`，不能通过请求体自助创建 `lawyer` 或 `admin`。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"client1","password":"Passw0rd!","email":"client1@example.com","real_name":"测试用户"}'

curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"client1","password":"Passw0rd!"}'
```

认证路由运行时成功响应使用 envelope：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "access_token": "<access_token>",
    "refresh_token": "<refresh_token>",
    "token_type": "bearer",
    "expires_in": 900,
    "user": {"id": "<user_id>", "role": "client"}
  }
}
```

把登录返回的 token 写入当前终端：

```bash
export ACCESS_TOKEN='<access_token>'
```

### 2. 创建 workflow 会话

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"user_type":"suspect","source":"readme"}'
```

`/sessions` 路由直接返回其 Pydantic response model，不包在 `data` 中：

```json
{
  "session_id": "<workflow_session_id>",
  "consultation_id": "<sqlite_consultation_id>",
  "welcome_message": "<权利义务告知>",
  "current_agent": "Receptionist",
  "created_at": "<ISO-8601>"
}
```

两个 ID 不能混用：

- `session_id`：LangGraph、Redis 和 `/sessions/{session_id}` 实时 workflow 路由使用。
- `consultation_id`：SQLite 咨询记录、历史和律师分配等数据库路由使用。

### 3. 知情同意与消息

```bash
export SESSION_ID='<workflow_session_id>'

curl -X POST "http://127.0.0.1:8000/api/v1/sessions/$SESSION_ID/confirm-consent" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"consent_given\":true,\"consent_timestamp\":\"2026-07-16T08:00:00Z\",\"consent_version\":\"v1\",\"identity_info\":{\"role\":\"suspect\"}}"

curl -X POST "http://127.0.0.1:8000/api/v1/sessions/$SESSION_ID/message" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"content\":\"事情发生在杭州，我和对方发生争执，对方先动手，我推了他一下。\",\"message_type\":\"text\"}"
```

核心权限边界：client 只能访问自己的会话；lawyer 只能访问已分配会话；admin 可管理用户、律师和知识库。律师审核使用 `PUT /api/v1/sessions/{session_id}/review`，可选择 `approved`、`revise_facts` 或 `revise_risk`。运行时错误统一为：

```json
{"error":{"code":"FORBIDDEN","message":"无权访问此会话"}}
```

部分历史成功路由仍返回 `{code,message,data}`，workflow `/sessions` 路由返回裸 response model；前端客户端同时兼容两种成功格式。自动生成的 422 OpenAPI schema 也可能仍显示 FastAPI 默认 `detail`，但运行时由全局处理器转换为 `error` envelope。

## Demo 与评估证据

### 可审计 Demo 资产

- [三条咨询 case 的数据契约](demos/consultation/README.md) 与 [普通伤害 case](demos/consultation/cases/ordinary_assault.json)：用于展示普通追问、事实不足和高风险转人工控制路径，不是真实案件。
- [确定性咨询 Demo runner](backend/examples/demo_complete.py) 与 [对应测试](backend/tests/demo/test_demo_complete.py)：固定节点输出会标记 `data_source=demo_fixture`，不冒充真实 RAG。
- [五条 RAG query](demos/rag/queries.json)、[历史失败记录](demos/rag/results/2026-07-13.json) 与 [Ollama live 记录](demos/rag/results/2026-07-14-ollama.json)：保留成功与失败条件，不能当作召回率或稳定性统计。

运行普通咨询 Demo：

```bash
cd backend
.venv/bin/python -m examples.demo_complete --case ordinary_assault
```

### 30 条 deterministic baseline

当前唯一活动评估入口是 [evaluation/run_eval.py](evaluation/run_eval.py)，输入为 [30 条固定 JSONL case](evaluation/cases.jsonl)，模式为 `offline-deterministic-baseline`。

| 指标 | 已有结果 |
| --- | ---: |
| 事实字段抽取覆盖 | 74 / 77（96.1%） |
| 法条关键词 hit@5 | 36 / 36（100.0%） |
| 高风险触发 | 30 / 30（100.0%） |
| 追问触发 | 29 / 30（96.7%） |
| 拒答触发 | 29 / 30（96.7%） |
| 免责声明 | 30 / 30（100.0%） |

这组结果只描述固定小样例中的确定性规则基线。runner 不调用真实 LLM、Ollama、Chroma、向量检索或 reranker，因此不能外推真实 LLM/RAG 准确率、开放输入表现或生产稳定性。历史 8 条 MVP 使用不同输入契约，也不与这组结果直接比较。

## 限制、安全边界与改进方向

- 本项目是工程原型：输出用于初步信息整理和律师审核前草案，不替代执业律师意见，不能承诺法律适用、量刑、程序或案件结果。30 条评估是固定 input-only deterministic baseline，不调用真实 LLM、Ollama、Chroma 或 reranker；一次同进程的 Ollama/Chroma 自然链只证明工程路径曾跑通，不代表法律正确性或生产稳定性。
- 事实覆盖低于 `0.8` 会追问，循环最多 10 次；这能限制无限等待，却不能证明短输入、待鉴定证据或多人多行为叙述已经查明。`FactDigger` 对已覆盖的特定要件做保守规则，但尚无事件图、跨主体一致性或证据来源校验。[事实覆盖实现](backend/app/agents/fact_digger.py)
- RAG 候选先经本地法条 JSON 验证；`rag_unverified` 不参与覆盖度计算，但“已验证”也不等于个案结论。向量库、embedding、关键词回退、编号抽取和 rerank 仍可能漏召回、误召回或排序不当。[LawRef 验证路径](backend/app/agents/law_ref.py)
- 模型输出可能空缺、不稳定或含占位；RiskAssessor 解析失败会保留“待评估/待确认”默认结构，而非给出可信风险结论。人工审核仍是必要断点。[风险降级](backend/app/agents/risk_assessor.py) · [审核断点](backend/app/orchestrator/workflow.py)
- 已有免责声明、知情同意门禁、PII 正则脱敏、高风险转人工和律师审核路径：FactDigger 在写入 `facts_raw` 前脱敏；高风险规则会进入 HumanAlert，HTTP/WS 暴露告警。但规则可能误报/漏报，脱敏并未构成端到端数据治理，HumanAlert 没有已证实的外部工单/通知闭环；通用“拒答”目前只在离线 baseline 中实现，不是主运行时统一策略。[安全过滤](backend/app/security/sensitive_filter.py) · [HumanAlert](backend/app/agents/human_alert.py) · [离线拒答基线](evaluation/run_eval.py)
- `MemorySaver`、进程内活跃队列、Redis 与 SQLite 不是强一致状态系统。律师批准的 workflow 完成态不会强一致回写 SQLite consultation；服务重启后不能由 Redis/SQLite 继续执行原图。当前没有完整容器 build/up/health、线上监控、容量或成本证据，不能表述为生产部署能力。[状态同步](backend/app/v1/service/consultation_service.py)

优先改进顺序是：先用持久化 checkpoint 和一致性回写补齐恢复能力，并把拒答/人工转交/模型失败处理做成共享运行时策略；再以更大、版本化的评估集验证 rerank、检索和事件图；最后在现有 client owner、assigned lawyer、admin 校验基础上继续细化案件/字段/动作级授权，补 access token 即时撤销、角色或禁用变化的及时生效、审核轨迹与最小权限，并建设脱敏可观测性、成本统计及前端对草案、降级和不可恢复状态的明确展示。

## 可迁移场景

下列是这套“结构化抽取 → 检索验证 → 条件路由 → 人工审核”架构可适配的方向，不是本仓库已经交付的产品：

- 企业知识库问答：将法条 JSON 验证替换为企业制度、产品或技术规范校验。
- 智能客服预审：用结构化 Tool Calling 收集工单要素，缺项时持续追问。
- 工单分流：把风险条件边替换为优先级、部门或 SLA 路由。
- 合规审核：对检索证据与规则库做双层验证，并保留人工批准门禁。
- 内部流程助手：复用 checkpoint、中断恢复、RBAC 和审计数据边界。

## 后续计划

1. 使用持久化 LangGraph checkpointer，并明确 SQLite、Redis 与 workflow state 的一致性及最终审核回写策略。
2. 完善知识库文档解析、向量库初始化与带版本的 RAG 评估集。
3. 增加 Alembic、容器 build/up smoke、前端部署与更细的失败场景测试。
4. 将 deterministic baseline 与固定模型/提示词/索引版本的 live LLM/RAG 评估分开报告。

## 代码入口

- [FastAPI 入口](backend/main.py)
- [LangGraph 工作流](backend/app/orchestrator/workflow.py)
- [Agent 节点](backend/app/agents)
- [RAG 服务](backend/app/rag)
- [API 路由](backend/app/v1/router)
- [React 前端](frontend/src)
- [Demo 资产](demos/README.md)
- [离线评估](evaluation/README.md)
- [部署配置](backend/.env.example)
- [LICENSE](LICENSE)

## License

本项目按 [MIT License](LICENSE) 开源。
