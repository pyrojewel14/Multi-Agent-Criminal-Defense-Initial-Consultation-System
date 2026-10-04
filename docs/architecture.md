# 系统架构

这篇说明组件如何连接、状态存在哪里、一次命令失败后如何恢复。
读完可以定位接口、工作流、检索和存储的职责。
节点细节读 [工作流](workflow.md)，启动操作读 [安装说明](setup.md)，能力范围见 [边界说明](limitations.md)。

## 组件关系

主流程由 LangGraph 控制节点顺序、条件路由和人工中断；LawRef 内部另有受工具、步骤和超时限制的模型决策循环。

```text
React / TypeScript 用户界面
  → FastAPI 路由 → JWT 身份校验 / RBAC 角色权限
  → ConsultationService：授权、恢复、串行执行、幂等与业务审计
  → LangGraph 工作流：9 个执行节点
      → LLM Gateway：模型调用、deadline 与有限重试
      → 法条检索：full 公共索引 / snapshot 通用 RagService
      → LangGraph checkpoint SQLite：图执行状态
      → 业务 SQLite：用户、咨询与消息审计
      → Redis：缓存与观测投影
```

JWT 是签名访问令牌，RBAC 是按角色控制权限，LangGraph 是保存状态并执行图流程的框架。人工和等待节点也是图节点；仓库名称不能直接解释为自主 Agent 数量。

## 一次咨询经过哪些层

1. 路由校验 token、角色与会话归属，调用统一 service 命令。
2. `fact_intake` 消费本轮输入，检测高风险、脱敏并刷新结构化事实。
3. `law_ref` 检索并读取法条；程序核验候选来源与最终答案。
4. `fact_digger` 计算覆盖，决定追问、继续分析或交给人工。
5. 风险与服务节点生成草案，律师明确审核后才可完成普通流程。

9 个节点、五种法条结果和三次连续失败窗口只在 [工作流](workflow.md) 维护。语料与覆盖资格见 [法条与检索](rag.md)，混合召回与重排由 [全量检索说明](knowledge/full_law_retrieval.md) 维护，原文、派生记忆和上下文预算由 [Memory 说明](memory/README.md) 维护。

## 存储与恢复

checkpoint 保存图状态与待执行节点（pending node），是恢复执行位置的依据。

| 存储 | 保存什么 | 使用范围 |
| --- | --- | --- |
| LangGraph checkpoint SQLite | 完整 workflow state、待执行节点 | FastAPI lifespan 注入 `AsyncSqliteSaver`，当前单实例可重启恢复 |
| 业务 SQLite | users、consultations、生命周期、HTTP/WS 外部原文与回复审计 | 不用这些行推断图执行位置 |
| Redis | 缓存、观测投影 | 启动硬依赖；不是恢复来源 |
| Chroma | 公共全量法条与用户上传的两套文档索引 | 分别配置；不随仓库预填充 |
| 进程内 trace / budget | 有界调用事件、会话调用/token 预算 | 不跨进程，重启清空 |

直接构造 `ConsultationOrchestrator()` 的单元测试默认使用进程内 `MemorySaver`。它与服务的 SQLite saver 不是同一种持久化方式；恢复配置、注入约束见 [工作流](workflow.md#中断恢复与持久化边界)。

`session_id` 是 LangGraph 的流程标识，`consultation_id` 是业务 SQLite 主键，两者通过 `consultations.workflow_session_id` 关联。接口使用哪一种 ID，见 [API](api.md#核心接口)。

checkpoint 会保存咨询事实、对话和报告等敏感状态。保留原 checkpoint 文件才能恢复；删除业务库、checkpoint 或索引分别影响不同的数据域。

## 并发与幂等

HTTP 和 WebSocket 消息复用 `process_message`；approve/reject/close 复用 `execute_lifecycle_command`。幂等指同一请求重试时返回原结果，避免重复推进。

| 规则 | 当前行为 |
| --- | --- |
| 串行范围 | 只锁同一 `session_id`；不同会话可并发 |
| 锁回收 | 最后一个持有者或等待者退出后删除锁项 |
| 幂等键 | `idempotency_key` 最长 128 字符；外部消息以会话、命令类型与键生成稳定 ID，并校验载荷指纹 |
| 重放 | 同 key 同载荷返回原结果；载荷不同返回 409 |
| 进程内 registry | 内部兼容消息与生命周期结果最多缓存 2048 条，默认保留一小时 |
| 外部消息回执 | HTTP/WS 以业务消息表和 checkpoint `message_audit` 修复或重放，不依赖该结果缓存 |
| 跨进程 | 重启、多 worker、多实例不共享串行锁与进程内 registry；持久外部回执的恢复仍依赖同一业务库与 checkpoint |

这不是跨进程 exactly-once（一次且仅一次执行）保证。

### 生命周期命令：先推进图，再提交审计

- 首次 approve/reject 只允许在真实 `human_review` 断点调用；其他位置返回 409。
- checkpoint 推进后若 SQLite 提交失败，标记 `repair_required`，保留真实执行位置。
- `consistency_error` 只记录 action、失败阶段和错误码；不会伪造回滚。
- 修复前阻断普通 resume；相同 action 可重试业务审计，即使图已到 END。
- 此类失败不缓存为终局结果，修复成功后清除故障标记。

### 消息命令：原文先提交，执行后修复审计

HTTP/WS 的 `process_message` 在同会话锁内重读 checkpoint，再交给 `process_external`。业务 SQLite 先提交完整输入；checkpoint 回执从 `prepared` 进入 `running`，图完成后记录 `applied` 与回复元数据，最后提交回复并清回执。

- 同键同载荷且回复已存在时返回已记录结果；不同载荷返回冲突。
- 已有 `applied` 回执而回复提交失败时，相同键只补写审计，不再次 resume。
- 崩溃遗留 `running`，或原文存在但没有可判定回执时，保守停止并要求核验 checkpoint；不能猜测是否执行过。
- 可选摘要/投影失败不撤销成功的消息命令；重试只刷新投影。

原文日志是审计来源，checkpoint 是唯一执行权威，两套 SQLite 没有共同事务。详见 [Memory 的恢复与失败边界](memory/README.md#恢复与失败边界)。

## 模型产物与失败处理

Fact、Law、Risk、Service 产物使用 Pydantic schema，即字段和类型校验契约。校验失败时：

- Fact 保留上一轮有效事实；保留不等于完成本轮提取。
- Law 不补造 `applied_laws`，按结果进入有限失败窗口。
- Risk/Service 转人工，风险降级通过条件边跳过 ServicePlanner。

LLM Gateway 对单次 attempt 和整个调用分别设 deadline，最多两次 attempt。只重试明确的 429、短暂 5xx、超时或网络瞬态错误；其他 4xx 和未知错误不重试。日志保存错误类型、状态码和结果，不保存模型原文或案件文本。

## 观测与预算

HTTP/WebSocket 请求生成或校验 UUID correlation id（请求关联标识）。`contextvars` 沿调用链传递上下文，事件用 `trace_id`、`span_id`、`parent_span_id` 表示父子关系。

| 配置/字段 | 用途 |
| --- | --- |
| `TRACE_MAX_EVENTS` | 限制进程内事件总量，超过后淘汰最早事件 |
| `SESSION_BUDGET_MAX_SESSIONS` | 限制预算 registry 中的会话数；满后拒绝新会话，不重置旧预算 |
| `SESSION_MAX_CALLS` / `SESSION_MAX_TOKENS` | 单会话调用/token 上限；调用前占用 call budget，返回已知 usage 后计 token |
| `SESSION_BUDGET_EXCEEDED` | 类型化预算错误；不覆盖已有 `repair_required` / `degraded` 状态 |
| `MODEL_PRICING_USD_PER_MILLION` | 显式静态模型单价，仅用于成本估算 |

事件含请求/会话/节点、模型与提示词版本、attempt、耗时、结果、缓存命中、token 和成本等字段；不适用的字段留空。usage 缺失时 token 为 `unknown`；价格或 usage 不完整时成本为 `unknown`，不能填零。观测 metadata 只保留允许的字段名、长度与规范化值 hash，不保存工具参数值或案件事实。

LawRef 的 `law_research` 摘要随 workflow state 保存；trace/budget registry 仍在进程内。局部工具预算与轨迹字段见 [工作流](workflow.md#法条来源与覆盖契约)。这些能力不是生产 APM（应用性能监控）、计费账本或全局配额。

## 代码入口与边界

| 入口 | 负责的层 |
| --- | --- |
| `backend/main.py` | FastAPI lifespan、路由和异常处理 |
| `backend/app/consultation/workflow.py` | 图节点、条件边和中断恢复 |
| `backend/app/consultation/agents/` | 各节点实现与 LawRef 局部工具循环 |
| `backend/app/knowledge/` | 语料加载、来源核验、两种检索路径 |
| `frontend/src/` | 客户端与律师界面 |
| `evaluation/` | 离线与真实组件评测入口 |

测试命令见 [testing.md](testing.md)。当前边界是单实例工程原型：没有多实例协调、完整备份恢复演练或法律专家质量验收。历史源文件可能仍在 Git 对象或远端引用中；当前文档修订不改写历史。更多限制见 [limitations.md](limitations.md)。
