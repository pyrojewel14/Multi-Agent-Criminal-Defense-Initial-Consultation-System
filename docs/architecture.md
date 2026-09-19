# 系统架构与状态边界

本文概述当前仓库的组件关系。详细节点行为见 [workflow.md](workflow.md)，部署限制见 [setup.md](setup.md)，法条来源边界见 [rag.md](rag.md)。

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
        +--> MemorySaver checkpoint（进程内）
        +--> Redis state cache
        +--> SQLite users / consultations / messages
        +--> LLM / embedding / Chroma / reranker（外部或本地依赖）
```

前端通过 FastAPI 访问会话与审核接口。服务层负责认证资源边界、LangGraph 恢复以及部分 Redis/SQLite 同步。Agent 共享 `ConsultationState`，条件边决定下一节点。

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
- `dependency_failure`：RAG 依赖失败或法条验证库不可用。

`no_law_match` 与 `dependency_failure` 共享三次连续失败预算。第 3 次失败时写入 degraded 状态并进入 `human_review`。这条路径保留审计记录，不会把依赖故障误当作事实已经充分，也不会绕过人工审核直接进入风险评估。

## 存储一致性

| 存储 | 当前用途 | 不能保证的内容 |
| --- | --- | --- |
| `MemorySaver` | LangGraph checkpoint | 进程重启或多实例共享 |
| Redis | 会话状态缓存 | 完整图恢复、与 SQLite 强一致 |
| SQLite | 用户、咨询、部分消息与分配关系 | 完整 workflow state 和 checkpoint |
| Chroma | 文档向量索引 | clean clone 自带语料、法律正确性 |

服务重启后仅有 Redis/SQLite 记录，不代表原 LangGraph 能继续执行。律师工作台需要用 `consultation_id` 关联业务记录，再用对应的 `session_id` 操作仍然活跃的 workflow。

## 数据发布边界

`backend/data/` 被 Git 与 Docker 构建共同排除，用于防止法条来源文件、SQLite、Chroma、模型和上传资料进入当前提交树或镜像。代码期待的运行时验证库路径是 `backend/data/law_knowledge/criminal_law_chapters.json`，clean clone 缺少它时会走依赖失败与有限降级路径。

旧数据文件仍可能存在于既有 Git/远端历史。当前删除不会清理历史；如未来因合规要求需要改写历史，必须先做独立 provenance 审计并取得明确授权。

这是一项已知限制，不是已完成的数据治理或知识库交付。
