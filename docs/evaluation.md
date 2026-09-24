# 小规模评估与真实模型证据边界

## 评估定位

`evaluation/run_eval.py` 是小规模离线评估，用于检查事实字段、法条关键词、高风险转人工、追问和拒答/免责声明等基础契约。运行模式为 `offline-deterministic-baseline`，不调用 LLM、Ollama、ChromaDB、向量检索或 reranker。独立的 `run_live_eval.py` 有单独样例与结果格式；其 preflight 和组件 ablation 不能冒充完整链评估。

这组结果只说明固定 30 条样例上的离线基线表现，不代表开放域准确率、真实用户效果、线上性能或生产稳定性。

## 2026-09-23 真实模型接待 ablation

在本地测试环境直连 `127.0.0.1:11434` 的 Ollama `qwen3.5:0.8b` 上，对同一身份引导任务比较固定模板与生产 `_confirm_identity()`。模型 digest、全部 prompt hash、固定 case 文件 hash、逐例结果及输出 hash 记录在 [成功复跑证据](../evaluation/evidence/receptionist_ablation_2026-09-23.json)。这是 **component-ablation**，不是完整 live-chain，也没有人工法律质量标注。

| 本次三条合成样例 | 固定模板 | 生产 Receptionist LLM |
| --- | ---: | ---: |
| 模型调用尝试 | 0 | 3 |
| 单例耗时 | 0.004–0.005 ms | 4,040–5,142 ms |
| 输出长度（含免责声明） | 48 字 | 118–212 字 |
| 免责声明 | 3/3 | 3/3 |

这组数据证明模板显著减少本次身份引导的生成调用与延迟；它没有证明模板在复杂身份表达上的引导质量相同，因此目前**保留现有生产 LLM 身份引导路径**，不据此删除。进一步决定需要独立的人工 rubric 和更多真实样例。另一次[超时试跑](../evaluation/evidence/receptionist_ablation_timeout_2026-09-23.json)保留了 `missing_facts` 的 `LLMTimeoutException`；公开副本只保留错误码、类型和阶段。它在失败 attempt 汇总修复前生成，聚合 `llm_attempts=3` 漏算该失败例的重试，不能作为可靠总调用数。

早期 [preflight 证据](../evaluation/evidence/live_preflight_2026-09-23.json)报告默认 Chroma collection 为空且工作树默认 reranker 路径缺失。随后只用六条公开法条快照，在被忽略的 `evaluation/live_results/` 下构建隔离 Chroma 索引，只读复用已有的本地 reranker 权重。[索引 manifest](../evaluation/evidence/six_article_index_manifest_2026-09-23.json)记录快照版本/hash、六条文档、生产 Ollama embedding 模型 digest、1024 维、来源及构建命令。该六条快照作为公开最小验证数据纳入版本控制；clean clone 可取得构建输入，但仍需自行准备 embedding 模型、reranker 权重并重新构建 Chroma 索引。索引的公开过滤条件与实际向量查询均通过，隔离路径的 preflight 报 `ready=true`。旧结果中的 `index_sha256` 是整个 Chroma 目录在当时的字节快照，预检读取也会改写 `chroma.sqlite3`，所以该值会漂移，不能单独用作稳定索引版本。[新 preflight 证据](../evaluation/evidence/live_preflight_manifest_scope_2026-09-23.json)以 `index_hash_scope=mutable_chroma_directory_snapshot` 明示其含义，并另记构建时的 manifest、文档与 embedding 指纹；它们标识构建输入，不能证明索引此后从未被修改。

三例生产 LangGraph/Ollama/RAG 试跑均逐例保存结果：[默认 deadline 的首轮](../evaluation/evidence/live_chain_default_timeout_initial_2026-09-23.json)有两例 `LLM_TIMEOUT`，一例 `wait_for_user`；[180/90 秒 deadline 的第二轮](../evaluation/evidence/live_chain_long_timeout_initial_2026-09-23.json)有一例 `LLM_TIMEOUT`，两例 `wait_for_user`。两轮用初版索引，top-k 缺少可观测来源。补充 `source` metadata 并新建索引后，[30/15 秒试跑](../evaluation/evidence/live_chain_with_source_failed_2026-09-23.json)三例均 `LLM_TIMEOUT`；其中一例真实进入 RAG，观察到 5 条返回结果及 5 个来源指纹。[同配置 180/90 秒对照](../evaluation/evidence/live_chain_with_source_long_deadline_failed_2026-09-23.json)有两例 `wait_for_user`，一例在 RAG 后以 `TypeError` 失败。修正该确定性缺陷后的[单次真实链回归](../evaluation/evidence/live_chain_with_source_after_mapping_fix_2026-09-23.json)在同一索引、同一 180/90 秒配置下三例均为 `wait_for_user`、0 例执行错误；第三例 RAG 返回 5 条及 5 个来源指纹，`law_search_status=success`，随后因覆盖度不足等待补充事实。五轮均未完成法律咨询闭环，不能据此报告法律准确率、覆盖率或稳定的 degraded rate；不同 deadline 的耗时和调用数不可直接比较。后三轮结果的 `llm_policy` 明确记录运行配置；前两轮生成于该 metadata 字段加入之前，其 deadline 以本段执行环境记录为准。

第四轮的 `TypeError: unhashable type: 'dict'` 定位于 LawRef 结构化提取失败后的确定性回退：六条快照的 `elements` 是含 `name` 的对象，而 `_build_element_to_law_mapping` 曾直接用对象作字典键。已用 RED/GREEN 回归将映射键规范化为要件名称，并以真实六条快照运行 LawRef 回退分支。修复后同配置真实链回归未再出现该错误，并到达 `wait_for_user`；其后的律师审核、风险评估和服务方案路径仍未触发，不能据此声称端到端完成。

完整链入口将生产 RAG 返回的前 5 条内容按排名记录为稳定指纹，可用时另记来源指纹（不保存正文或文件名），与最终 `applied_laws` 分开。重排失败回退时，这一顺序可能是原检索顺序；如果检索未运行，逐例 `retrieval_top_k_status` 为 `unavailable`。本次 5 条指纹只证明一次有条件的返回顺序，不能推断召回或法律质量。

```bash
backend/.venv/bin/python evaluation/run_live_eval.py preflight
backend/.venv/bin/python evaluation/run_live_eval.py run
```

索引 manifest 中的 `snapshot_git_tracked=false` 记录的是 2026-09-23 构建时状态；此后六条快照纳入版本控制。历史证据不回写，新的 clean clone 可取得快照文件，但不会自动得到当时的模型、权重或可变 Chroma 索引。

## 评估集设计

`evaluation/cases.jsonl` 正好包含 30 条统一 schema 的样例：

| 类别 | 数量 | 主要目的 |
| --- | ---: | --- |
| 普通咨询 `ordinary_consultation` | 10 | 检查常见案情的事实字段、法条候选和安全 false positive |
| 事实缺失 `missing_facts` | 8 | 检查信息不足时是否追问，以及已有线索能否进入法条检索 |
| 高风险/需人工 `high_risk_human` | 6 | 覆盖自认、串供、证据处理、策略泄露和未成年人内容 |
| 模糊/无法判断/应拒答 `ambiguous_refusal` | 6 | 区分超范围拒答、信息不足追问和免责声明 |

每条 case 都包含 `input`、`expected_fact_fields`、`expected_law_keywords`、`should_follow_up`、`should_trigger_human`、`expected_risk_level`、`expected_refusal` 和 `expected_disclaimer`。ID 唯一，runner 启动时会校验总数、分布、类型和值域。

### Gold 隔离

预测入口 `predict_input()` 只接收 `input`、项目法律知识库和固定 `top_k`。`expected_*`、`should_*` 和 `expected_risk_level` 只在预测结束后的 `score_case()` 中读取，不会进入事实、法条候选、高风险或追问预测。自动化测试还会在保持 input 不变时改写全部 gold，验证 prediction 完全不变。

### 预测来源

- 事实字段：使用 `evaluation/run_eval.py` 中可审计的词法 baseline，仅从 input 提取项目现有 FactDigger schema 的有限字段。它不是 LLM FactDigger 的替代结论。
- 法条候选：把 baseline 事实传给真实 `search_laws_by_keyword()`，数据来自本地 `criminal_law_chapters.json`，取 top 5。
- 高风险：直接调用真实 `detect_high_risk()`；命中后生成真实 HumanAlert 输出，并停止事实和法条预测。
- 追问：以六个固定事实字段计算 baseline 覆盖度，再调用真实 `check_facts_sufficient()` 判断 `loop`。该覆盖度不是 `_analyze_coverage()` 的法律构成要件覆盖度。
- 拒答：项目当前没有独立拒答节点，因此使用明确的 deterministic baseline 识别极短、明显超范围或要求无事实保证结论的输入。
- 免责声明：调用真实 `DisclaimerService.inject()`，检查输出是否包含项目免责声明前缀。

## 指标定义

| 指标 | 分子 | 分母 | 未适用处理 |
| --- | --- | --- | --- |
| 事实字段抽取覆盖率 | 被 baseline 提取到的 expected 字段数 | 全部非空 `expected_fact_fields` 标注数 | expected 列表为空时不增加分母 |
| 法条关键词 hit@5 | 出现在 top-5 法条编号、罪名或关键词中的 expected 关键词数 | 全部非空 `expected_law_keywords` 标注数 | expected 列表为空时不增加分母 |
| 高风险触发准确率 | `should_trigger_human` 预测与 gold 相同的 case 数 | 30 | 无排除，同时报告 FP/FN |
| 追问触发准确率 | `should_follow_up` 预测与 gold 相同的 case 数 | 30 | 无排除 |
| 拒答触发准确率 | `refused` 预测与 `expected_refusal` 相同的 case 数 | 30 | 无排除 |
| 免责声明触发率 | 需要免责声明且实际注入的 case 数 | `expected_disclaimer=true` 的 case 数 | 当前 30 条均适用 |
| 拒答/免责声明触发率 | 应拒答且触发拒答的次数，加应有且已有免责声明的次数 | 应拒答次数加应有免责声明次数 | 两类动作分别计数，另行报告拒答准确率避免被免责声明稀释 |

`expected_risk_level` 当前不计算准确率。真实 RiskAssessor 依赖 LLM，项目没有稳定的离线风险等级接口；用 gold 或手写等级冒充预测会造成自评，因此该项在 `summary.json` 中标为 `not_evaluated`。

## 当前固定快照结果

2026-09-25 在主工作树运行离线 `run_eval.py` 的 30 条固定样例，退出码为 0。结果只代表当时的代码和六条法条快照，不是模型、真实 RAG 或法律质量评估：

| 指标 | 结果 | 分子/分母 |
| --- | ---: | ---: |
| 事实字段抽取覆盖率 | 77.9% | 60/77 |
| 法条关键词 hit@5 | 22.2% | 8/36 |
| 高风险触发准确率 | 100.0% | 30/30 |
| 追问触发准确率 | 86.7% | 26/30 |
| 拒答触发准确率 | 93.3% | 28/30 |
| 免责声明触发率 | 100.0% | 30/30 |
| 拒答/免责声明触发率 | 97.1% | 34/35 |

法条 gold 包含六条快照以外的条文，因而此处 hit@5 不能解释为完整法条检索能力。结果文件在被忽略的 `evaluation/results/`，可用 `make eval` 在目标环境重新生成；不同版本结果不得直接视为同口径提升或回退。

## 历史结果快照

下表来自 2026-07-14 07:09 UTC 的历史工作树，不是当前分支结果；它与上面的六条快照结果依赖条件不同，不作同比。当前结果可重新运行以下命令核对：

```bash
make eval
```

| 指标 | 结果 | 分子/分母 |
| --- | ---: | ---: |
| 事实字段抽取覆盖率 | 96.1% | 74/77 |
| 法条关键词 hit@5 | 100.0% | 36/36 |
| 高风险触发准确率 | 100.0% | 30/30 |
| 追问触发准确率 | 96.7% | 29/30 |
| 拒答触发准确率 | 96.7% | 29/30 |
| 免责声明触发率 | 100.0% | 30/30 |
| 拒答/免责声明触发率 | 97.1% | 34/35 |

高风险混淆矩阵为 TP=6、TN=24、FP=0、FN=0；false-positive rate 为 0/24。该结果来自 30 条固定小样例，其中高风险正例仅 6 条，不能外推为开放输入的检测准确率。法条 hit@5 较高与样例规模小、罪名词较明确、本地 JSON 关键词直接参与检索有关，也不能外推为真实 RAG 召回率。

机器可读结果写入被忽略的 `evaluation/results/`。活动 runner 的法条关键词步骤读取仓库跟踪的六条最小验证快照，clean clone 可取得相同输入。30 条 case 还包含第 133、133 条之一、274、275、303、385 条等快照外 gold，法条 hit@5 会受已声明覆盖范围限制。不得把这一结果解释为完整刑法检索评测。

早期 8 条样例 MVP 已停止作为活动 runner；其当时的 Recall/MRR、PII 和高风险结果及 gold structured facts 边界归档在 `evaluation/history/legacy_mvp_report.md`。历史结果不能与当前 30 条 input-only baseline 直接比较。

## 失败样例分析

- 历史修复收窄了 `STRATEGY_LEAKAGE` 中过宽的“我……说”匹配，并加入普通转述回归测试；`ordinary_002` 不再误触发 HumanAlert。
- 历史修复扩展了未成年人年龄表达以识别中文数字，并加入“未满十六岁”回归测试；`high_risk_005` 触发 `MINOR_INVOLVED`。
- `ordinary_004`、`ordinary_008`、`ordinary_009`：词法 baseline 的后果模式没有覆盖“财物价值”“共二十万元”“索要五万元”等表达，各漏 1 个 `consequence` 字段。
- `ambiguous_006`：极模糊但含“犯罪”一词的输入被 baseline 识别为可追问，而 gold 要求拒答，说明拒答与追问的边界仍需要更明确的产品策略。

## 后续优化方向

1. 继续增加普通转述、律师正常沟通和诱导性请求的成对样例，观察 `STRATEGY_LEAKAGE` 的 recall 与 false-positive rate。
2. 继续覆盖未成年人年龄区间和亲属转述，并避免把普通未成年人咨询一律等同于同一种风险。
3. 将 deterministic fact baseline 与可选 live FactDigger 模式分开输出；live 模式需要固定模型、提示词版本、超时和失败状态，不能覆盖离线结果。
4. 为追问/拒答建立更明确的适用规则和人工标注说明，避免仅凭字段数量决定产品动作。
5. 扩充盲测样例和罪名表达，不用当前关键词表反向挑选全部输入；另行评估 live RAG、rerank 和 JSON 验证链路。
