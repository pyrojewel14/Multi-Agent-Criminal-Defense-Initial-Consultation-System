# 咨询工作流：节点、路由与恢复

这篇说明一条咨询经过哪些节点、何时追问或转人工、如何从中断恢复。
读完可以核对状态和条件边，定位流程偏差。
组件与命令一致性读 [架构](architecture.md)，运行示例读 [Demo](demo.md)，能力范围见 [边界说明](limitations.md)。

## 角色与执行节点

工作流有 8 个角色、9 个 LangGraph 执行节点。FactDigger 被拆成两个阶段，所以角色数少于节点数。

| 执行节点 | 角色 | 主要职责 |
| --- | --- | --- |
| `receptionist` | Receptionist | 告知、知情同意与身份入口 |
| `fact_intake` | FactDigger | 单次消费本轮输入，高风险检测、脱敏、结构化事实刷新 |
| `law_ref` | LawRef | 有界工具决策、RAG/JSON 召回、来源核验和权威要件连接 |
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
                   │                         ├─ 正文标注待复核 / 连续非事实失败耗尽 -> human_review
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
2. 仅从本轮输入提取候选，schema 校验后增量合并到 `memory.case`，再投影兼容 `facts_structured`；`facts_raw` 仅保留有界近期陈述，消费成功后清空 `current_input`。
3. `law_ref` 只读取已经刷新的 `facts_structured`。
4. `fact_digger` 使用新法条候选计算覆盖度。
5. 覆盖不足时经过 `wait_for_user` 中断；下一轮仍从 `fact_intake` 开始。

`resume_workflow()` 只有在快照下一节点包含 `fact_intake` 时才接受 `current_input` 更新。流程一旦越过摄取节点，后续恢复会丢弃该一次性字段，避免把同一输入重放到失败节点。

字段来源、空值保留、冲突/更正和调用预算统一见 [Memory 说明](memory/README.md#事实摄取与字段合并)。

律师选择 `revise_facts` 时也回到 `fact_intake`。该入口会清除一次性的律师决定；若此前状态是 `degraded`，还会重置当前失败窗口，但保留 `fact_law_attempts` 审计历史。

## 法条来源与覆盖契约

LawRef 内部是局部 Tool Agent：模型选工具，程序校验参数并返回 observation（工具执行结果），模型再决定下一步。

| 工具 | 用途 |
| --- | --- |
| `search_laws(query)` | 按当前 profile 检索、JSON 核验、关键词补召回 |
| `get_article(article_id)` | 读取已搜索且通过核验的条文；未放行正文也可阅读 |
| `search_elements(article_id)` | 读取同一条文的项目要件；未放行条文返回空要件 |

默认采用 `legacy` 工具消息循环。`LAW_AGENT_FINAL_PROTOCOL=native_candidate` 显式启用原生最终输出候选；它有独立的失败证据，不能当作默认修复。协议对照见 [响应诊断](knowledge/lawref_response_protocol.md)。

| 预算 | 默认值 | 配置 |
| --- | --- | --- |
| 模型决策步骤 | 4，允许配置 1–8 | `LAW_AGENT_MAX_STEPS` |
| 单次工具执行 | 90 秒 | `LAW_AGENT_TOOL_TIMEOUT_SECONDS` |
| 整个局部循环 | 240 秒 | `LAW_AGENT_TIMEOUT_SECONDS` |

- 相同工具名和规范化参数重复出现时，以 `duplicate_call` 停止。
- 无效工具、无效参数、空结果和无效最终答案分别进入轨迹。
- 工具超时、依赖失败或预算耗尽会停止本次循环。
- RAG 失败但关键词仍有候选时，工具记录 `partial_dependency_failure`，模型可继续核验。
- 只有合法最终答案连接到已核验、已读取的候选后，才适配成法条结果；不会自动把检索候选当作成功产物。

`law_research` 保存步骤、工具顺序、参数指纹、状态、结果计数、耗时、终止原因和调用统计。它不保存模型原文、工具参数值或案件事实。token 使用当前 session budget 的差值；usage 缺失时为 `unknown`。观测与预算范围见 [架构](architecture.md#观测与预算)。

候选来源及 `text_only` 的枚举边界统一见 [法条与检索](rag.md#候选来源与覆盖资格)。只有可信来源并带非空受控 `required_elements` 的候选才可计算覆盖。

FactDigger 对每个可信候选分别算覆盖率，选择支持率最高者；不会把互斥罪名的要件相加。`facts_coverage_rate >= 0.8` 是继续自动流程的阈值，只表示项目要件的覆盖程度，不表示事实真实或法律适用正确。

## 法条结果与失败窗口

| `law_search_status` | 含义 | 后续处理 |
| --- | --- | --- |
| `success` | 有可信、带要件且通过最终校验的结果 | FactDigger 计算覆盖 |
| `text_only` | `law_text_candidates` 中有标注未放行的正文参考；也可能同时有可覆盖候选 | 覆盖置零、清空追问，以 `annotation_review_required` 直接进入 `human_review` |
| `missing_facts` | 结构化事实不足 | 补充事实路径 |
| `no_law_match` | 检索空结果满足无匹配判定 | 计入连续失败窗口 |
| `dependency_failure` | 依赖、超时、预算或最终答案失败等未产生合法结果 | 计入连续失败窗口 |

`text_only` 不等待三次失败。它要解决的是标注缺口，反复询问案件事实无法解决；模型或检索故障也不应解释为“用户事实必然不足”。

### 连续失败如何计数

`no_law_match` 与 `dependency_failure` 都属于“非事实失败”。它们共享同一个连续失败计数 `fact_law_failure_streak`：

- 每次失败都会追加一条 `fact_law_attempts` 审计记录；
- 两种失败交替出现仍累计在同一个窗口内；
- 可信法条候选成功或回到正常事实补充路径时，连续失败计数归零；
- 连续第 3 次非事实失败时，状态变为 `workflow_status="degraded"`，记录 `fact_law_termination_reason`，并进入 `human_review`；
- degraded 状态不会自动进入 RiskAssessor，也不会继续无限调用外部依赖。

`human_review` 在 degraded 路径中明确说明自动检索已停止，并设置 `lawyer_review_needed` / `awaiting_lawyer_review`。人工选择 `revise_facts` 后可以开启新的失败窗口，已有 attempts 不删除。

## 中断、恢复与持久化边界

`start_workflow()` 和 `resume_workflow()` 都以 `session_id` 作为 LangGraph `thread_id`。执行位置以 checkpointer 为准。

- FastAPI lifespan 注入独立 SQLite 文件的 `AsyncSqliteSaver`，当前单实例可按原 `session_id` 重启恢复。
- 直接构造 orchestrator 默认用进程内 `MemorySaver`，供纯单元测试使用。
- Redis、兼容缓存和业务 SQLite 都不能用来猜测 pending node。

恢复配置的约束：

- 同一 SQLite checkpoint 文件上的 orchestrator 重建、中断后续跑与 session 隔离有确定性测试覆盖；
- lifespan 在图首次编译前注入 saver，并在关闭时释放连接；saver 初始化失败会阻止启动，不回退到 `MemorySaver`；
- 注入非 `MemorySaver` 的 durable checkpointer 并显式设置 `persistent=True` 后才声明跨进程恢复；缺少 saver 或把 `MemorySaver` 标为 persistent 会在构造时被拒绝；
- `session_id` 是工作流标识，`consultation_id` 是 SQLite 记录标识，两者不能混用；
- API 鉴权和律师分配校验不能由 LangGraph 中断机制替代；
- 已知 `session_id` 恢复与进程内活跃列表发现旧会话不同；业务历史返回可空的 `workflow_session_id`。外部消息回执和保守崩溃处理见 [Memory](memory/README.md#恢复与失败边界)，[阶段 3 正常停启交付](memory/phase3-verification.md)待对应主线程独立验收；多 worker 未由上述配置能力验证。

生命周期命令的执行断点校验、`repair_required` 与相同 action 审计修复，统一见 [架构的命令一致性说明](architecture.md#生命周期命令先推进图再提交审计)。API 的角色和分配校验见 [接口说明](api.md)。

## 可复现验证

从 `backend/` 运行：

```bash
.venv/bin/python -m pytest -q \
  tests/consultation/agents/test_fact_digger.py \
  tests/consultation/agents/test_law_ref.py \
  tests/consultation/agents/test_legal_research.py \
  tests/consultation/test_workflow.py \
  tests/consultation/test_workflow_degraded.py \
  tests/consultation/test_workflow_example.py \
  tests/consultation/test_workflow_minimal.py \
  tests/integration/test_data_flow.py
```

预期：pytest 列出通过/失败数，全部通过时退出码为 0。分组验证节点顺序、状态更新、来源校验、覆盖和失败恢复。

## 术语与边界

LangGraph 的 `interrupt_after` 表示执行指定节点后暂停；pending node 是恢复时待执行的节点；checkpointer 是保存和读取这些状态的组件。Pydantic 校验字段和类型，不验证法律推理。

上述定向测试主要使用模型替身，验证控制流与状态契约。实际模型、索引和部署需分别执行检查，不能从定向测试外推；见 [边界说明](limitations.md)。
