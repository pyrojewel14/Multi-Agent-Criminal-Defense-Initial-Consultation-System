# Agent Memory 与上下文管理

本页说明原文审计、单案字段记忆、滚动摘要和调用前预算，核对日期为 2026-10-04。源码存在不能代替真实模型质量或部署验收。流程与恢复见 [工作流](../workflow.md)，接口权限见 [API](../api.md)，检索配置见 [全量检索](../knowledge/full_law_retrieval.md)。

## 四类数据的职责

| 数据 | 保存位置 | 权威与用途 |
| --- | --- | --- |
| raw transcript | 业务 SQLite `consultation_messages` | 完整外部输入、实际回复与内部审计事件；稳定 ID、单案 sequence、command_id、record_kind。不是图执行位置来源 |
| `memory.case` | LangGraph checkpoint | 单案字段、来源 ID、未核实状态和冲突/更正版本；投影成兼容 `facts_structured` |
| `memory.recent` | LangGraph checkpoint | 有界近期外部对话，以完整命令组保留；不是完整原文档案 |
| `memory.summary` | LangGraph checkpoint | 滚动摘要、through_sequence、version、status；摘要文本和覆盖游标一同提交 |

checkpoint 是唯一执行权威，包含真实 state 与 pending node。Redis、原文日志和进程内活动缓存不能替代它。`facts_raw` 是有界的脱敏近期陈述，`conversation_history` 是兼容投影；不能据此声称每个 checkpoint 累计完整 raw transcript。

原文审计不以知情同意为前提；派生近期上下文和摘要刷新要求 `consent_given is True`。每个案件沿现有 owner、consultation ID 与 workflow ID 隔离，没有同用户跨案件共享记忆或 semantic retrieval memory。

## 外部命令与原文

HTTP/WS 共用 `service.process_message`。传输入口未传数据库会话时，服务创建 `AsyncSessionLocal`；持有同会话进程内锁后重读 checkpoint，再进入 `coordinator.process_external`。原文先提交业务 SQLite，然后记录 checkpoint 消息回执并执行图，最后提交实际回复。稳定消息 ID 与载荷指纹允许同键重放或拒绝冲突。

消息按 `external` 与 `internal` 区分；内部错误事件不进入摘要，实际返回给用户的错误响应可作为外部记录。同意与生命周期相关外部载荷也有相应审计入口；旧消息缺失的原文、轮次或确认状态不会被凭空补出。

## 事实摄取与字段合并

`fact_intake_node` 对本轮输入先检测高风险、脱敏，再仅从本轮候选输入提取并校验 FactArtifact；ContextBuilder 的 fact_intake 任务不加入旧摘要、案件字段或近期记忆。没有新陈述且已有 case memory 时直接返回，不以旧输入重新提取。兼容旧状态时可使用最后一条旧陈述，旧字段以 `legacy_unverified` 导入。

`merge_case_memory` 的字段语义：

- 空值不删除已有字段；相同值合并来源 ID。
- 标量新值与旧值冲突时保留 alternatives，并标记 `conflicted`，不会自动证明哪一个为真。
- 列表去重；具名实体可按 id/name 等身份字段关联，无法识别身份的条目仍可能分开保留。
- 输入包含“更正”“纠正”“之前说错”时启用更正合并，旧版本及来源保留，新版本标为 `user_corrected_unverified`。这是当前字符串门禁，不能保证理解所有自然语言更正或撤回。
- `facts_structured` 是 case memory 的兼容投影；覆盖率、模型 confidence 或字段存在不等于律师确认。

提取失败保留上一轮有效字段；核心 typed LLM 异常仍按既有错误/重试契约处理。本轮成功消费才清除一次性输入，避免异常重试重复追加。

## 配置与预算

默认值来自 `MemorySettings.from_env()`，单位为保守估算值，并非实际 tokenizer 计数。

| 环境变量 | 默认 | 用途 |
| --- | ---: | --- |
| `MEMORY_RECENT_MESSAGES` | 8 | 近期消息目标条数，按完整命令组选择，至少保留最新完整组；极长组可从派生 recent 排除 |
| `MEMORY_CONTEXT_TOKEN_BUDGET` | 24000 | 每次模型输入估算上限 |
| `MEMORY_MODEL_WINDOW` | 32768 | 部署声明的模型窗口 |
| `MEMORY_OUTPUT_RESERVE` | 2048 | 输出预留；输入上限取预算与窗口减预留的较小值 |
| `MEMORY_SUMMARY_INPUT_BUDGET` | 6000 | 摘要指令、旧摘要、新增段与协议开销的独立输入预算 |
| `MEMORY_SUMMARY_OUTPUT_BUDGET` | 1000 | 摘要输出估算上限 |
| `MEMORY_SUMMARY_TIMEOUT_SECONDS` | 20 | 可选摘要总超时 |

`estimate_tokens` 按 UTF-8 字节数加 16 的协议余量估算；中文字节数会明显高于常见 tokenizer token 数，不能将该数与供应商 actual usage 混用。配置值必须为正，模型窗口必须大于输出预留；部署者仍需按实际模型核对窗口。

Gateway 的文本、结构化与工具调用经 ContextBuilder 装配输入，估算包括 system、当前任务、工具/schema、背景记忆和工具历史。本轮必要输入及最新完整工具调用组必须适配；超预算抛出明确错误，不静默截断用户尾部或拆散 tool call/observation。可选背景与旧历史按完整组舍弃；tool call ID 关联不完整也被拒绝。上下文预算与进程内 session call/token budget 是不同约束。

## 增量滚动摘要

`memory_rows` 分页读取 cursor 后的待摘要段和有界 recent，不每轮读取完整原文。`advance_summary` 只选择 `through_sequence` 之后、近期窗口之前的新增外部消息，按完整命令组装入预算；旧摘要和指令也占预算。

摘要候选合法、非空且未超过输出预算后，更新文本与 cursor/version；checkpoint 保存成功才成为派生权威。摘要失败不推进旧 cursor，保留有界 recent；无法放入首个完整组时标记 `blocked_large_message`。失败不会退回发送无限原文，也不撤销已成功的消息命令。

当前摘要使用一次 attempt、有限总超时，Ollama 单次 `reasoning=False`；输出生成限额按摘要输出预算换算，返回文本仍按保守估算复核。该开关不改变共享模型或其他供应商。摘要是不可信背景，可能遗漏或漂移；调用前会标明未核实并脱敏。

## 恢复与失败边界

- FastAPI lifespan 注入 SQLite `AsyncSqliteSaver`；独立构造 orchestrator 默认是进程内 MemorySaver。保存同一 checkpoint、业务库并持有原 workflow ID，才具备恢复基础。
- `message_audit` 的 `prepared` 可继续执行；已持久确认的 `applied` 回执只补写回复审计，不再次 resume。已记录回复可返回原结果。
- 崩溃留下 `running` 且结果不明，或原文存在但没有明确回执时，保守停止并要求人工核验 checkpoint；两套 SQLite 没有原子事务，不能声称所有崩溃窗口自动修复。
- 同 session 锁、内部/生命周期结果 registry、trace 和会话预算仍在进程内。外部持久回执不等于跨进程锁或 exactly-once。
- 已知 ID 恢复与发现旧会话不同：活跃 `/sessions` 列表读取进程内缓存，不自动枚举旧 checkpoint。业务历史返回可空 `workflow_session_id`，映射存在时可提供恢复 ID，旧记录可能没有。
- lifecycle 的 `repair_required` 保留独立契约，相同 action 修复审计；待完成消息会阻断冲突生命周期操作。参见 [命令一致性](../architecture.md#生命周期命令先推进图再提交审计)。

当前定向测试与跨进程探针不能外推真实模型多轮质量、全部崩溃窗口、多 worker 或生产稳定性。新增 [阶段3交付记录](phase3-verification.md) 待对应主线程独立验收：隔离宿主真实 fact/summary、法律和 preflight 替身、固定合成样例及单 worker 正常停启。长样例输入减少、短样例增加，回忆延迟没有改善；不能据此宣称总体成本或延迟下降。本轮仅核对文档与源码，未运行行为测试或模型。验证入口见 [testing](../testing.md#memory-与-full-检索契约) 与 [evaluation](../evaluation.md)。

## 源码入口与历史分析

| 入口 | 关键符号与职责 |
| --- | --- |
| [service.py](../../backend/app/consultation/service.py) | process_message / process_consent / execute_lifecycle_command，外部入口与串行 |
| [coordinator.py](../../backend/app/consultation/memory/coordinator.py) | process_external / refresh_memory，消息回执与摘要投影 |
| [transcript.py](../../backend/app/consultation/memory/transcript.py) | append_record / memory_rows，原文追加与分页 |
| [case.py](../../backend/app/consultation/memory/case.py) | merge_case_memory / project_facts，字段来源和冲突 |
| [context.py](../../backend/app/consultation/memory/context.py) | MemorySettings / ContextBuilder / recent_rows，完整组与预算 |
| [summary.py](../../backend/app/consultation/memory/summary.py) | advance_summary，增量摘要和游标 |
| [fact_digger.py](../../backend/app/consultation/agents/fact_digger.py) | fact_intake_node，本轮候选摄取 |
| [gateway.py](../../backend/app/infrastructure/llm/gateway.py) | _bounded_messages / generate / generate_with_tools，message_history 工具历史与实际调用装配 |

[2026-10-02 实施前分析](../history/memory/2026-10-02-analysis.md) 保留当时完整技术分析、失败风险和测试记录；其中“尚未实现”与历史输入增长探针不代表本页核对后的实现状态。
