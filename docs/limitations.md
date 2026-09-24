# 失败场景与边界说明

本项目是刑事咨询工程原型。代码路径存在或定向测试通过，不等于开放输入、真实案件、法律质量、安全性或生产稳定性得到保证。

## 事实与覆盖

- FactDigger 的覆盖率只针对所选可信法条候选的权威 `required_elements`，不代表事实真实、完整或无矛盾。
- 多人、多次、多地点、多罪名以及相互冲突叙述仍可能被压入同一结构化字段；系统没有完整事件图或证据来源链。
- 否定、待鉴定和未知表达采用保守规则，但规则无法覆盖所有自然语言变体。
- `current_input` 是一次性字段；流程越过 `fact_intake` 后不会重放，调用方应根据当前中断节点发送更新。

## 法条与 RAG

- clean clone 只包含第 232、234、263、264、266、293 条的 tracked 最小验证快照，不包含完整刑法库、Chroma 索引或模型权重。
- 快照中的条文编号与正文有政府发布文本和版本元数据；`title`、`base_sentence`、`elements`、`charge_tags`、`common_keywords` 是非官方项目索引、摘要或标注，不是法律专家确认的完整构成要件或法律意见。
- `rag_unverified` 与 `llm_extracted` 候选不参与覆盖计算，但仍需人工检查其内容和来源。
- HyDE、关键词匹配、法条编号提取、向量召回和 rerank 都可能漏召回、误召回或排序错误。
- 仓库没有完整、持续更新的公开法条数据交付，也没有法律专家标注的检索评测集。

## 失败恢复

- `no_law_match` 与 `dependency_failure` 共享连续失败窗口；第 3 次失败进入 `degraded` 与 `HumanReview`。
- degraded 只是停止自动重试并请求人工接管，不表示故障原因已修复。
- 人工 `revise_facts` 会开启新的失败窗口，但保留既有 attempt 审计；若依赖仍不可用，仍可能再次 degraded。
- 高风险节点只更新工作流状态和提示，不等同于短信、邮件、报警或外部律师工单已发送。

## LLM 输出

- FactDigger、LawRef、RiskAssessor 和 ServicePlanner 的输出受模型、提示词、上下文、超时和服务状态影响。
- 结构化解析失败不会成为成功产物：Fact 可保留上一轮已验证事实，Law 可使用明确标记的确定性候选回退，Risk/Service 则转人工；任何回退都不能作为可靠法律判断。
- 报告草案必须经过有权限的律师复核；系统不能承诺罪名、量刑、程序或案件结果。

## 观测与预算

- 调用树、token、延迟、成本估算和 session budget 都是单进程内能力；多 worker 不共享，服务重启后清空，不能当作生产 APM、计费账本或全局配额。预算 registry 达到 `SESSION_BUDGET_MAX_SESSIONS` 后会拒绝新的 session，不会淘汰旧条目并重置其预算。
- 事件存储受 `TRACE_MAX_EVENTS` 限制，达到上限会淘汰最早事件；它不是持久审计源。
- token 依赖供应商 usage；缺失时明确记录 `unknown`。此时 token budget 无法增加，但 call budget 仍会阻止无限模型/RAG 调用。
- `cost_usd` 只使用显式配置的静态模型单价估算；未配置价格、模型名不匹配或 usage 缺失时为 `unknown`，不代表真实账单。

## 状态与持久化

- 默认 LangGraph 使用进程内 `MemorySaver`；当前依赖未提供 durable saver，因此默认部署不宣称服务重启后可恢复原图执行位置。
- 同 session 串行锁和幂等结果也只存在于当前进程。锁项会在最后一个持有者/等待者退出后回收；可安全重放结果最多缓存 2048 条、默认保留一小时。进程重启、多 worker 或多实例之间不共享这些记录，不能据此宣称跨进程 exactly-once。
- HTTP 消息在 workflow 已推进后若 checkpoint 投影或 SQLite 消息提交失败，同 key 会重放原错误而不会再次 resume；这避免重复 LLM/history，但不会自动补写缺失的 SQLite 消息，需要人工或后续对账任务处理。
- Redis 不是执行状态源；SQLite 是生命周期/消息审计源。approve/reject/close 先推进 checkpoint，再提交 SQLite；审计提交失败时不会伪造回滚，而是在真实执行位置标记 `repair_required` 并阻断普通 resume，等待相同操作重试修复。
- `session_id` 与 `consultation_id` 属于不同存储域，不能混用。
- access token 没有服务端即时撤销列表；登出、禁用或改角色后，已签发 token 的生命周期边界需要额外治理。

## 部署与数据

- Compose 只包含 backend 与 Redis，并携带六条最小验证快照；不包含前端、Ollama、完整法条语料、Chroma 内容或模型权重。
- `.env`、数据库、索引、模型、日志、上传资料和私有文档均不应进入 Git 或 Docker context。
- Docker 配置解析通过不等于镜像 build、容器健康、模型调用或真实 RAG 链已验证。
- 项目没有 Alembic migration、默认生产级持久化 checkpointer、备份恢复演练或多实例一致性方案。

## 评估解释

- `evaluation/` 的活动 runner 是小规模 deterministic baseline，不调用真实 LLM、Chroma 或 reranker。
- 历史指标必须附日期、命令和样例范围，不得当作当前结果或生产指标。
- 定向 pytest 只证明所列契约；历史上完整收集曾被 `Killed: 9` 终止，因此不能声称全量测试通过。
