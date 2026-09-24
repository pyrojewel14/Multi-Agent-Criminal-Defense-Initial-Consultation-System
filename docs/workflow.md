# LangGraph 多 Agent 工作流

本文以 `backend/app/orchestrator/workflow.py`、`backend/app/agents/fact_digger.py`、`backend/app/agents/law_ref.py` 和相应定向测试为准。它描述控制流与状态契约，不证明外部模型、RAG 质量、法律结论或生产级恢复能力。

## 角色与执行节点

工作流有 8 个角色、9 个 LangGraph 执行节点。FactDigger 被拆成两个阶段，所以角色数少于节点数。

| 执行节点 | 角色 | 主要职责 |
| --- | --- | --- |
| `receptionist` | Receptionist | 告知、知情同意与身份入口 |
| `fact_intake` | FactDigger | 单次消费本轮输入，高风险检测、脱敏、结构化事实刷新 |
| `law_ref` | LawRef | RAG/JSON 召回、来源标记和权威要件连接 |
| `fact_digger` | FactDigger | 基于刷新事实与法条候选计算覆盖度，生成追问或摘要 |
| `wait_for_user` | WaitForUser | 覆盖不足时中断，等待下一条用户输入 |
| `risk_assessor` | RiskAssessor | 生成风险分析草案 |
| `service_planner` | ServicePlanner | 生成服务方案和报告草案 |
| `human_review` | HumanReview | 律师审核或 degraded 人工接管 |
| `human_alert` | HumanAlert | 高风险输入短路并提示人工处理 |

## 当前拓扑

```text
START -> receptionist
  ├─ 未同意 -> END
  └─ 已同意 -> fact_intake -> law_ref -> fact_digger
                   │                         ├─ 覆盖充分 -> risk_assessor
                   │                         │              -> service_planner
                   │                         │              -> human_review
                   │                         ├─ 覆盖不足 -> wait_for_user
                   │                         │              -> fact_intake（新输入）
                   │                         ├─ 连续非事实失败耗尽 -> human_review
                   └─ 高风险 -> human_alert -> END

human_review
  ├─ approved -> END
  ├─ revise_facts -> fact_intake
  ├─ revise_risk -> risk_assessor
  └─ 无有效决定 -> human_review（继续保持中断）
```

四个 `interrupt_after` 节点是 `receptionist`、`wait_for_user`、`human_review` 和 `human_alert`。

## 两阶段 FactDigger 与一次性输入

旧的“FactDigger 先分析、再去 LawRef、恢复后直接 LawRef”的顺序会让法条检索看不到用户刚补充的事实，也可能在异常重放时重复追加同一条输入。当前顺序固定为：

1. `fact_intake` 读取 `current_input`，执行高风险检测和 `sanitize_input()`。
2. 本轮提取成功后写回 `facts_raw` / `facts_structured`，并把 `current_input` 清为 `None`。
3. `law_ref` 只读取已经刷新的 `facts_structured`。
4. `fact_digger` 使用新法条候选计算覆盖度。
5. 覆盖不足时经过 `wait_for_user` 中断；下一轮仍从 `fact_intake` 开始。

`resume_workflow()` 只有在快照下一节点包含 `fact_intake` 时才接受 `current_input` 更新。流程一旦越过摄取节点，后续恢复会丢弃该一次性字段，避免把同一输入重放到失败节点。

律师选择 `revise_facts` 时也回到 `fact_intake`。该入口会清除一次性的律师决定；若此前状态是 `degraded`，还会重置当前失败窗口，但保留 `fact_law_attempts` 审计历史。

## 法条来源与覆盖契约

法条候选的 `data_source` 必须属于 `LawDataSource`：

- `rag_verified`
- `json_keyword`
- `rag_unverified`
- `llm_extracted`

只有 `rag_verified` 或 `json_keyword` 候选，并且携带非空、由受控知识源提供的 `required_elements`，才能参与覆盖度计算。LLM 返回的 `elements_matched` 只能在这些权威要件内取子集，不能自行扩大覆盖分母或把未连接候选升级为可信来源。

多个候选法条可能互斥，FactDigger 分别计算每个可信候选的覆盖率，并选择支持率最高者；不会把所有罪名的要件相加后要求事实同时满足。

`facts_coverage_rate >= 0.8` 只表示当前结构化事实覆盖了所选候选的多数权威要件，不表示事实真实、法条适用正确或案件结论可靠。

## 有限失败窗口

`no_law_match` 与 `dependency_failure` 都属于“非事实失败”。它们共享同一个连续失败计数 `fact_law_failure_streak`：

- 每次失败都会追加一条 `fact_law_attempts` 审计记录；
- 两种失败交替出现仍累计在同一个窗口内；
- 可信法条候选成功或回到正常事实补充路径时，连续失败计数归零；
- 连续第 3 次非事实失败时，状态变为 `workflow_status="degraded"`，记录 `fact_law_termination_reason`，并进入 `human_review`；
- degraded 状态不会自动进入 RiskAssessor，也不会继续无限调用外部依赖。

`human_review` 在 degraded 路径中明确说明自动检索已停止，并设置 `lawyer_review_needed` / `awaiting_lawyer_review`。人工选择 `revise_facts` 后可以开启新的失败窗口，已有 attempts 不删除。

## 中断、恢复与持久化边界

`start_workflow()` 和 `resume_workflow()` 都以 `session_id` 作为 LangGraph `thread_id`。执行权威始终是 LangGraph checkpointer；当前默认实现是进程内 `MemorySaver`，因为本仓依赖未提供 durable saver。服务层不再从 Redis 或进程缓存猜测 pending node；SQLite 只保存业务审计状态。

因此：

- 同一进程内的中断恢复有定向测试覆盖；
- 注入非 `MemorySaver` 的 durable checkpointer 并显式设置 `persistent=True` 后才允许跨进程恢复；缺少 saver 或把 `MemorySaver` 标为 persistent 会在构造时被拒绝；
- `session_id` 是工作流标识，`consultation_id` 是 SQLite 记录标识，两者不能混用；
- API 鉴权和律师分配校验不能由 LangGraph 中断机制替代。

approve/reject/close 通过同一个 application command 写路径推进 checkpoint，再更新 SQLite 审计行。首次 approve/reject 只允许在真实 `human_review` 待执行断点调用，否则返回 409 且不推进图、不写数据库；repair retry 即使已到 END 仍可用相同 action 修复审计。SQLite 提交失败后不会伪造 checkpoint 回滚：工作流在当前真实执行位置标记为 `workflow_status="repair_required"`，并记录仅含 `action`、`failed_stage`、`error_code` 的非敏感 `consistency_error`。普通 resume 会被阻断；使用相同 action 重试时只修复 SQLite 审计投影，成功后清除故障标记。Redis 不是恢复源，也不是生命周期命令的写路径。

## 可复现验证

从 `backend/` 运行：

```bash
.venv/bin/python -m pytest -q \
  tests/agents/test_fact_digger.py \
  tests/agents/test_law_ref.py \
  tests/orchestrator/test_workflow.py \
  tests/orchestrator/test_workflow_degraded.py \
  tests/orchestrator/test_workflow_example.py \
  tests/orchestrator/test_workflow_minimal.py \
  tests/integration/test_data_flow.py
```

这些测试主要使用确定性替身或 monkeypatch，证明节点顺序、状态更新、来源校验、覆盖计算和失败恢复契约；它们不证明外部 LLM、Redis、Chroma 或法律资料当前可用。
