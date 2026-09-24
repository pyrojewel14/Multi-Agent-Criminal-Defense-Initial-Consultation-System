# 系统架构与状态边界

本文概述当前仓库的组件关系。详细节点行为见 [workflow.md](workflow.md)，部署限制见 [setup.md](setup.md)，法条来源边界见 [rag.md](rag.md)。

本项目对外定位为 **Workflow with LLM Nodes / 受控 Agentic Workflow**。LangGraph 的条件边、循环、中断与审核共同约束模型节点；`HumanReview`、`HumanAlert`、`WaitForUser` 是人工/控制节点，不是独立推理 Agent。仓库名称是历史名称，不用于推断有多少自主 Agent。

## 组件

```text
React / TypeScript
        |
        v
FastAPI routes ---- JWT / RBAC
        |
        v
ConsultationService
        |
        v
LangGraph StateGraph
  receptionist
  fact_intake -> law_ref -> fact_digger
  wait_for_user / human_alert
  risk_assessor -> service_planner -> human_review
        |
        +--> LangGraph checkpointer（默认 MemorySaver，进程内）
        +--> Redis optional projection/cache
        +--> SQLite users / consultations / messages（业务审计）
        +--> LLM / embedding / Chroma / reranker（外部或本地依赖）
```

前端通过 FastAPI 访问会话与审核接口。服务层负责认证资源边界、LangGraph checkpoint 恢复、同 session 命令串行化、成功结果幂等重放以及 lifecycle application command 的 SQLite 审计同步。Agent 共享 `ConsultationState`，条件边决定下一节点。

## LangGraph 节点

当前注册 9 个执行节点：

1. `receptionist`
2. `fact_intake`
3. `law_ref`
4. `fact_digger`
5. `wait_for_user`
6. `risk_assessor`
7. `service_planner`
8. `human_review`
9. `human_alert`

`fact_intake` 和 `fact_digger` 都属于 FactDigger 角色。拆分的目的不是增加一个业务 Agent，而是保证法条检索始终看到本轮最新事实，并使 `current_input` 成为一次性状态。

`human_review` 等待真实律师决定；`human_alert` 终止高风险自动路径；`wait_for_user` 在事实不足时等待新输入。它们提供可核验的控制流边界，而非模型推理能力。评测入口及目前缺失的 live-chain 条件见 [评估说明](evaluation.md)。

## 关键状态

| 领域 | 字段 | 契约 |
| --- | --- | --- |
| 身份 | `user_id`、`user_role`、`session_id`、`consultation_id` | RAG 权限使用 `user_id`；工作流与数据库 ID 不混用 |
| 本轮输入 | `current_input` | 仅在 `fact_intake` 前写入，成功摄取后清空 |
| 事实 | `facts_raw`、`facts_structured`、`pending_questions` | 原始输入先脱敏，再更新结构化事实 |
| 法条 | `applied_laws`、`element_to_law_mapping`、`law_search_status` | 候选保留来源；未验证来源不能参与覆盖计算 |
| 覆盖 | `facts_coverage_rate` | 仅由可信候选的权威 `required_elements` 计算 |
| 重试 | `fact_law_attempts`、`fact_law_failure_streak`、`fact_law_last_failure` | 记录每次尝试，并对所有非事实失败共享连续窗口 |
| 降级 | `workflow_status`、`fact_law_termination_reason` | 连续 3 次非事实失败后标记 degraded 并转人工 |
| LLM 产物 | `artifact_results`、`degraded_reason`、`source`、`validation_errors` | 保存 Fact/Law/Risk/Service 的 schema 校验与降级元数据，不含模型原文 |
| 人审 | `lawyer_review_needed`、`awaiting_lawyer_review`、`lawyer_decision` | 缺少有效决定时保持中断，不默认批准 |

## 事实、法条与覆盖的数据流

```text
current_input
  -> fact_intake
       高风险检测
       脱敏并追加 facts_raw
       刷新 facts_structured
       current_input = None
  -> law_ref
       RAG 召回
       结构化法条 JSON 验证/关键词补召回（数据存在时）
       标记 data_source
       连接 required_elements
  -> fact_digger
       校验 LawDataSource
       仅可信候选进入 CoverageCandidateSchema
       每个候选独立计算覆盖率
       生成追问、摘要或 degraded 人工转交
```

`required_elements` 必须来自与候选法条编号成功连接的受控知识条目。模型生成的匹配/缺失判断可用于候选内部说明，但不能自行创建权威要件或改变数据来源。

## 失败处理

`LawRef` 将结果区分为：

- `success`：存在可信且带权威要件的候选；
- `missing_facts`：尚无足够结构化事实；
- `no_law_match`：依赖可用但没有可信匹配；
- `dependency_failure`：运行中的 RAG 等外部依赖失败。法条验证快照缺失或损坏会在 startup preflight 阶段直接阻止服务启动。

`no_law_match` 与 `dependency_failure` 共享三次连续失败预算。第 3 次失败时写入 degraded 状态并进入 `human_review`。这条路径保留审计记录，不会把依赖故障误当作事实已经充分，也不会绕过人工审核直接进入风险评估。

四类 LLM 产物共用严格 Pydantic 契约。`source` 表示实际产物通道：Fact 可为 `tool_call` 或 `content_json`，Law/Risk/Service 为 `content_json`，只有基于已召回候选的确定性 Law 回退为 `deterministic_fallback`。Fact 校验失败时保留上一轮有效事实；Risk 或 Service 缺字段、错类型或非 JSON 时不生成默认成功产物，并直接进入 `human_review`。风险降级使用真实条件边跳过 ServicePlanner。

LLM Gateway 对每个 attempt 和整个调用分别设置 deadline，最多执行两次。只有明确的 429、短暂 5xx、单次超时或网络瞬态错误可在带有界抖动的退避后重试；其他 4xx 和未知错误不重试。日志只记录 attempt、outcome、错误类型和状态码，不记录 prompt、案件事实、工具参数值或模型原文。

## 调用链与预算

HTTP 消息和 WebSocket 消息都生成或校验 UUID correlation id，并通过 `contextvars` 贯穿 `workflow -> node -> LLM/RAG`。每个事件用 `trace_id`、`span_id`、`parent_span_id` 表示父子关系，不依赖日志文本推断。统一字段包括 `request_id`、`session_id`、`node`、`route_reason`、`model`、`prompt_version`、输入/输出 token、`duration_ms`、`attempt`、`outcome`、`cache_hit` 与 `cost_usd`；不适用于某类事件的字段保持空值。供应商未返回 usage 时 token 和成本都记为 `unknown`，不会填 0。

观测事件只允许保存字段名、长度和规范化值 hash，不保存 prompt、案件事实、工具参数值或模型原文。`TRACE_MAX_EVENTS` 限制进程内事件总量，`SESSION_BUDGET_MAX_SESSIONS` 限制预算 registry 的 session 数量。`SESSION_MAX_CALLS` 和 `SESSION_MAX_TOKENS` 为单会话宽松上限；每次 LLM attempt、HyDE 调用与 RAG 检索先占用 call budget，已知 usage 再计入 token budget。超限抛出类型化 `SESSION_BUDGET_EXCEEDED`，不会改写既有 `repair_required` 或 `degraded` 状态。

`MODEL_PRICING_USD_PER_MILLION` 可按模型显式配置输入、输出单价；只有模型有价格且 usage 完整时才估算 `cost_usd`。当前不自动抓取供应商价格，避免价格漂移或错误的零成本假设。

## 存储一致性

| 存储 | 当前用途 | 不能保证的内容 |
| --- | --- | --- |
| LangGraph checkpointer | 执行状态与 pending node（默认 `MemorySaver` 为进程级；生产需注入 durable saver） | 未配置 durable saver 时的进程重启/多实例共享 |
| Redis | 可选缓存/观测投影 | pending node 推断、完整图恢复、与 SQLite 强一致 |
| SQLite | 用户、咨询、生命周期与消息审计关系 | 完整 workflow state 和 checkpoint |
| Chroma | 文档向量索引 | clean clone 自带语料、法律正确性 |
| 进程内 trace/budget | 最近的有界事件、单会话调用/token 用量 | 跨进程共享、重启恢复、多 worker 全局预算 |

服务重启后，只有 durable LangGraph checkpointer 才能恢复原执行位置；Redis/SQLite 记录不能替代它。律师工作台通过 `workflow_session_id` 关联业务记录，再用对应的 `session_id` 操作 workflow。

## 命令并发与幂等边界

HTTP 与 WebSocket 消息复用 `process_message` application command；approve、reject 与 close 复用 `execute_lifecycle_command`。当前实现只在同一 `session_id` 内串行执行，其他会话可并发推进，不使用覆盖工作流执行期的全局锁。会话锁 registry 统计持有者与等待者，最后一个引用退出后立即删除锁项。

客户端可为消息和生命周期操作提供最长 128 字符的 `idempotency_key`。可安全重放的结果按 `(session_id, command_type, idempotency_key)` 缓存在当前进程；同 key 同载荷返回原结果，同 key 不同载荷返回 409。缓存只保存载荷摘要与结果，默认保留一小时，并按最近使用顺序限制为 2048 条。生命周期命令的跨存储失败不缓存，因此 `repair_required` 可继续用同 key 重试审计修复。消息命令若已推进 workflow、但后续 checkpoint 投影或 SQLite 写入失败，则缓存原错误结果，避免同 key 再次调用 LLM 或追加 history；该路径不会自动补写缺失的 SQLite 消息，需要另行对账。

这些锁和幂等结果与默认 `MemorySaver` 一样都是单进程边界：进程重启会丢失，多 worker 或多实例之间也不共享。生产多实例部署需要 durable checkpointer 配合数据库幂等表、版本 CAS 或共享锁；当前实现不宣称跨进程 exactly-once。

## 数据发布边界

Git 与 Docker 构建对 `backend/data/` 使用精确 allowlist：只放行 `backend/data/law_knowledge/criminal_law_chapters.json` 六条最小验证快照，SQLite、Chroma、模型、旧来源文件和上传资料仍被排除。应用在数据库与 Redis 初始化前校验快照来源/版本元数据、字段完整性、唯一条号和覆盖清单；校验失败时拒绝启动，不把缺失可信数据伪装成可服务状态。

旧数据文件仍可能存在于既有 Git/远端历史。当前删除不会清理历史；如未来因合规要求需要改写历史，必须先做独立 provenance 审计并取得明确授权。

这是一项已知限制，不是已完成的数据治理或知识库交付。
