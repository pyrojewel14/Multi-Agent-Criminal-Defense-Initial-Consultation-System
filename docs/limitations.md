# 失败场景与边界说明

这篇帮助你判断哪些结果可以作为工程证据，哪些仍需人工或额外验证。
阅读后可以区分文本、标注、模型判断和部署证据的范围。
想先操作，读 [安装说明](setup.md)；想看结果依据，读 [评估](evaluation.md)。

本项目是刑事咨询工程原型。咨询报告必须由有权限的律师复核。

## 事实与覆盖

- FactDigger 的覆盖率只针对所选可信法条候选的权威 `required_elements`，不代表事实真实、完整或无矛盾。
- 当前 case memory 已保存字段级来源 ID、冲突 alternatives 与更正旧版本；多人、多次、多地点、多罪名仍可能被压入兼容结构化字段。它没有完整事件图，也没有从原始证据到法律结论的完整证据链。
- 否定、待鉴定和未知表达采用保守规则，但规则无法覆盖所有自然语言变体。
- `current_input` 是一次性字段；流程越过 `fact_intake` 后不会重放，调用方应根据当前中断节点发送更新。

## 法条与 RAG

- 新克隆包含默认 full 语料（505 条记录、504 条有效正文），以及 snapshot 六条历史回归资产；没有预建 Chroma 索引或模型权重。标注与来源统一见 [法条与检索](rag.md#语料与标注)。
- 条文编号与正文有政府发布文本和版本元数据；`title`、`base_sentence`、`elements`、`charge_tags`、`common_keywords` 是非官方项目索引、摘要或标注，不是法律专家确认的完整构成要件或法律意见。
- `rag_unverified`、`llm_extracted` 和 `text_only` 不参与覆盖计算。38 条演示标注可参与计算，也没有法律专家背书；程序放行不等于法律认可。
- 关键词、条号与向量召回都可能出错；默认 full 使用混合召回与 Qwen 重排，可选 HyDE 默认关闭。重排超时、繁忙或失败回退融合顺序，HyDE 失败只丢弃额外通道；完整查询超出评分预算时明确降级，长条文尾部可能未参与重排评分。配置与截断范围见 [全量检索说明](knowledge/full_law_retrieval.md)。
- 带版本的公开刑法正文不提供持续法律更新服务，也没有法律专家标注的检索评测集。

## 失败恢复

- 检索和最终生成采用有限失败窗口；`text_only` 则直接请求人工标注复核，不继续追问。计数与路由统一见 [工作流](workflow.md#法条结果与失败窗口)。
- degraded 只是停止自动重试并请求人工接管，不表示故障原因已修复。
- 人工 `revise_facts` 会开启新的失败窗口，但保留既有 attempt 审计；若依赖仍不可用，仍可能再次 degraded。
- 高风险节点只更新工作流状态和提示，不等同于短信、邮件、报警或外部律师工单已发送。

## Memory 与上下文

- 业务 SQLite 保留 raw transcript；checkpoint 中保存案件字段、近期消息、滚动摘要及游标。`facts_raw` 是有界兼容输入，不能当作完整历史原文。
- 每次模型调用使用 ContextBuilder。预算按 UTF-8 字节加协议开销保守估算，实际 token 仍依赖供应商 usage；没有部署模型 tokenizer 的精确测量。
- 必要的本轮输入与最新工具调用组完整保留，超过预算明确拒绝；背景记忆按完整组淘汰。摘要需要已同意，失败保留旧摘要/游标且不回退无限原文。
- 来源、空值、冲突和显式更正是当前字段合并契约；它们不验证用户陈述真实性，也不能替代律师确认。[阶段 3](memory/phase3-verification.md)已于 2026-10-04 由主线程独立验收固定合成样例的真实提取、摘要和服务停启功能；摘要有损、模型误提取及脱敏误伤仍是内容质量限制，技术主说明见 [Memory](memory/README.md)。

## LLM 输出

- FactDigger、LawRef、RiskAssessor 和 ServicePlanner 的输出受模型、提示词、上下文、超时和服务状态影响。
- 结构化解析失败不会成为成功产物：Fact 可保留上一轮已验证事实；Law 不生成 `applied_laws`，按失败状态进入有限重试与人工审核；Risk/Service 则转人工。保留的事实不等于可靠法律判断。
- 报告草案必须经过有权限的律师复核；系统不能承诺罪名、量刑、程序或案件结果。

## 观测与预算

- 调用树、token、延迟、成本估算和 session budget 都是单进程内能力；多 worker 不共享，服务重启后清空，不能当作生产 APM、计费账本或全局配额。预算 registry 达到 `SESSION_BUDGET_MAX_SESSIONS` 后会拒绝新的 session，不会淘汰旧条目并重置其预算。
- 事件存储受 `TRACE_MAX_EVENTS` 限制，达到上限会淘汰最早事件；它不是持久审计源。
- token 依赖供应商 usage；缺失时明确记录 `unknown`。此时 token budget 无法增加，但 call budget 仍会阻止无限模型/RAG 调用。
- `cost_usd` 只使用显式配置的静态模型单价估算；未配置价格、模型名不匹配或 usage 缺失时为 `unknown`，不代表真实账单。

## 状态与持久化

- FastAPI lifespan 使用独立的 LangGraph checkpoint SQLite 文件，单实例后端进程重启后可恢复原图执行位置；直接构造 orchestrator 的纯单元测试默认仍使用进程内 `MemorySaver`。checkpoint 包含完整咨询工作流状态，属于敏感本地数据，Git 已忽略运行数据库。
- 同 session 串行锁与内部兼容/生命周期结果 registry 是进程内能力；HTTP/WS 原文、回复和 `message_audit` 回执另行持久保存，不能概括为重启后全部丢失。它们不构成跨进程锁或 exactly-once 保证。
- 外部输入先提交业务 SQLite，图执行后记录 checkpoint `applied` 回执。该回执已确定执行成功时，相同键只补审计；崩溃留下 `running` 或缺少明确回执时保守停下。[阶段 3 正常停启验证](memory/phase3-verification.md)已于 2026-10-04 由主线程独立验收，范围限合成小样例与单 worker 正常停启；完整崩溃窗口与多 worker 仍需验证，见 [Memory 说明](memory/README.md#恢复与失败边界)。

- Redis 不是执行状态源；SQLite 是生命周期/消息审计源。approve/reject/close 先推进 checkpoint，再提交 SQLite；审计提交失败时不会伪造回滚，而是在真实执行位置标记 `repair_required` 并阻断普通 resume，等待相同操作重试修复。
- `session_id` 与 `consultation_id` 属于不同存储域，不能混用。
- access token 没有服务端即时撤销列表；登出、禁用或改角色后，已签发 token 的生命周期边界需要额外治理。

## 部署与数据

- Compose 只包含 backend 与 Redis，镜像携带公开法条 JSON；前端、Ollama、索引与权重需自行准备。当前 Compose 没有全量索引挂载，补充方式见 [安装说明](setup.md#docker-compose)。
- 真实 `.env`、数据库、索引、模型、日志、上传资料和私有文档不应进入 Git 或 Docker 构建上下文。公开法条 JSON 通过精确白名单进入仓库和镜像。
- Docker 配置解析通过不等于镜像 build、容器健康、模型调用或真实 RAG 链已验证。
- 项目没有 Alembic migration、checkpoint 备份恢复演练或多实例一致性方案；SQLite checkpoint 适用于当前单实例边界，扩展性有限。跨会话 semantic retrieval memory 未实现。

## 评估解释

- `evaluation/run_eval.py` 是固定 30 条 input-only deterministic baseline，不调用真实 LLM、Chroma 或 reranker。`run_full_eval.py live-rag` 与 `run_full_retrieval_ablation.py` 另走真实 full 检索/读取或候选消融，按配置调用 embedding、Chroma、重排和可选 HyDE；`live-lawref` 才增加最终模型判断。各入口数据契约与可证明范围见 [评估说明](evaluation.md#先选评估入口)。
- 历史指标必须附日期、命令和样例范围，不得当作当前结果或生产指标。
- 定向 pytest 只证明所列契约；历史完整收集曾被 `Killed: 9` 终止，该次记录不能作为全量通过证据。当前全量通过声明必须有目标代码上的完整运行、退出码与失败边界，不能由历史中断或其他日期的通过数推断。

## 当前模型证据

2026-10-01 的默认本地小模型两轮六例 LawRef 历史复测均为 0/6。云模型在修订提示词后的固定六例对照两轮均为 6/6，但只证明该输入与配置下的局部契约；没有完整咨询闭环或法律专家评审证据。失败历史与精确配置见 [响应协议诊断](knowledge/lawref_response_protocol.md)。
