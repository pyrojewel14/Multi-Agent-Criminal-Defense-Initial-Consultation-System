# 失败场景与边界说明

本项目是刑事咨询工程原型。代码路径存在或定向测试通过，不等于开放输入、真实案件、法律质量、安全性或生产稳定性得到保证。

## 事实与覆盖

- FactDigger 的覆盖率只针对所选可信法条候选的权威 `required_elements`，不代表事实真实、完整或无矛盾。
- 多人、多次、多地点、多罪名以及相互冲突叙述仍可能被压入同一结构化字段；系统没有完整事件图或证据来源链。
- 否定、待鉴定和未知表达采用保守规则，但规则无法覆盖所有自然语言变体。
- `current_input` 是一次性字段；流程越过 `fact_intake` 后不会重放，调用方应根据当前中断节点发送更新。

## 法条与 RAG

- clean clone 不包含 `backend/data/` 下的法条来源文件、代码实际读取的 `law_knowledge/criminal_law_chapters.json`、Chroma 索引或模型权重。
- `rag_unverified` 与 `llm_extracted` 候选不参与覆盖计算，但仍需人工检查其内容和来源。
- HyDE、关键词匹配、法条编号提取、向量召回和 rerank 都可能漏召回、误召回或排序错误。
- 仓库没有版本化的公开法条数据交付，也没有法律专家标注的检索评测集。

## 失败恢复

- `no_law_match` 与 `dependency_failure` 共享连续失败窗口；第 3 次失败进入 `degraded` 与 `HumanReview`。
- degraded 只是停止自动重试并请求人工接管，不表示故障原因已修复。
- 人工 `revise_facts` 会开启新的失败窗口，但保留既有 attempt 审计；若依赖仍不可用，仍可能再次 degraded。
- 高风险节点只更新工作流状态和提示，不等同于短信、邮件、报警或外部律师工单已发送。

## LLM 输出

- FactDigger、LawRef、RiskAssessor 和 ServicePlanner 的输出受模型、提示词、上下文、超时和服务状态影响。
- 结构化解析失败可能保留上一轮事实或使用“待评估”类默认结构；这类回退不能作为可靠法律判断。
- 报告草案必须经过有权限的律师复核；系统不能承诺罪名、量刑、程序或案件结果。

## 状态与持久化

- LangGraph 使用进程内 `MemorySaver`；Redis、SQLite 与 checkpoint 不是强一致事务。
- 服务重启或多实例切换后，Redis/SQLite 记录不能保证恢复原图执行位置。
- `session_id` 与 `consultation_id` 属于不同存储域，不能混用。
- access token 没有服务端即时撤销列表；登出、禁用或改角色后，已签发 token 的生命周期边界需要额外治理。

## 部署与数据

- Compose 只包含 backend 与 Redis，不包含前端、Ollama、法条数据、Chroma 内容或模型权重。
- `.env`、数据库、索引、模型、日志、上传资料和私有文档均不应进入 Git 或 Docker context。
- Docker 配置解析通过不等于镜像 build、容器健康、模型调用或真实 RAG 链已验证。
- 项目没有 Alembic migration、生产级持久化 checkpointer、备份恢复演练或多实例一致性方案。

## 评估解释

- `evaluation/` 的活动 runner 是小规模 deterministic baseline，不调用真实 LLM、Chroma 或 reranker。
- 历史指标必须附日期、命令和样例范围，不得当作当前结果或生产指标。
- 定向 pytest 只证明所列契约；历史上完整收集曾被 `Killed: 9` 终止，因此不能声称全量测试通过。
