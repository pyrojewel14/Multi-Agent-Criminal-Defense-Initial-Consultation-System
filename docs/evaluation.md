# 评估：基线、真实检索与模型证据

这篇帮助你选评测入口，并解释结果能支持什么结论。
读完可以区分离线基线、真实组件测试和完整咨询闭环。
只想运行测试，读 [testing.md](testing.md)；使用限制见 [limitations.md](limitations.md)。

## 先选评估入口

| 入口 | 语料与依赖 | 检查什么 |
| --- | --- | --- |
| `run_eval.py` / `make eval` | 固定六条 snapshot，不调用模型 | 30 条 input-only 离线基线 |
| `run_full_eval.py offline` | full，真实关键词和读取工具 | 全量数据/工具契约 |
| `run_full_eval.py live-rag` | full，真实 embedding、Chroma 与配置启用的重排 | 固定条号或语义查询的检索/读取 |
| `run_full_retrieval_ablation.py` | full，真实 Registry/索引/重排，可选 HyDE | 四组检索候选对照，详见 [全量检索说明](knowledge/full_law_retrieval.md#复跑消融) |
| `run_memory_eval.py` | 固定多轮样例，按 runner 模式使用替身或真实模型 | Memory/上下文评测，不能与单输入 baseline 混合 |
| `run_full_eval.py live-lawref` | full，真实聊天模型和检索 | LawRef 局部最终答案；不是整条咨询 |
| `run_live_eval.py` | 历史 snapshot 与配套索引 | preflight、组件对照或生产图链路 |
| `run_cloud_lawref.py` | 指定云模型、full 索引、公开固定案例 | 显式配置下的 LawRef 云对照 |

先按 [安装说明](setup.md) 准备依赖。不同入口有独立样例、评分与结果格式，不能混合计分。

Memory 的状态与预算契约见 [当前说明](memory/README.md)。[阶段 3 固定样例交付](memory/phase3-verification.md)待对应主线程独立验收，范围说明见该页；本轮文档核对未运行任何评测。

## 最近的真实模型记录

2026-10-01 的默认本地小模型与新提示词复测两轮均 0/6；云模型固定六例对照两轮均 6/6。前者没有可验收的最终判断，后者没有完整咨询闭环或法律专家审查。准确配置、失败阶段与公开证据见 [响应协议诊断](knowledge/lawref_response_protocol.md)。这些是历史结果；当前模型效果需在目标环境按相同配置重新执行。

全量正文的索引与离线工具证据见 [全量资产说明](knowledge/full_criminal_law.md#评测与证据)。

## 评估定位

`evaluation/run_eval.py` 是小规模离线评估，用于检查事实字段、法条关键词、高风险转人工、追问和拒答/免责声明等基础契约。运行模式为 `offline-deterministic-baseline`，不调用 LLM、Ollama、ChromaDB、向量检索或 reranker。独立的 `run_live_eval.py` 有单独样例与结果格式；其 preflight 和组件 ablation 不能冒充完整链评估。

这组结果只说明固定 30 条样例上的离线基线表现，不代表开放域准确率、真实用户效果、线上性能或生产稳定性。

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

## 2026-09-25 固定快照结果

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

下表来自 2026-07-14 07:09 UTC 的历史工作树，不是当前分支结果；它与上面的六条快照结果依赖条件不同，不作同比。新的目标环境结果可用以下命令生成：

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

早期 8 条样例 MVP 已停止作为活动 runner；其指标与 gold structured facts 边界归档在 [历史报告](../evaluation/history/legacy_mvp_report.md)。历史结果不能与当前 30 条 input-only baseline 直接比较。

## 历史失败样例分析

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

## 2026-09-23 真实模型接待 ablation

在本地测试环境直连 `127.0.0.1:11434` 的 Ollama `qwen3.5:0.8b` 上，对同一身份引导任务比较固定模板与生产 `_confirm_identity()`。

模型 digest、全部 prompt hash、固定 case 文件 hash、逐例结果及输出 hash 记录在 [成功复跑证据](../evaluation/evidence/receptionist_ablation_2026-09-23.json)。

这是 **component-ablation**，不是完整 live-chain，也没有人工法律质量标注。

| 本次三条合成样例 | 固定模板 | 生产 Receptionist LLM |
| --- | ---: | ---: |
| 模型调用尝试 | 0 | 3 |
| 单例耗时 | 0.004–0.005 ms | 4,040–5,142 ms |
| 输出长度（含免责声明） | 48 字 | 118–212 字 |
| 免责声明 | 3/3 | 3/3 |

这组数据证明模板显著减少本次身份引导的生成调用与延迟；它没有证明模板在复杂身份表达上的引导质量相同，因此目前**保留现有生产 LLM 身份引导路径**，不据此删除。

进一步决定需要独立的人工 rubric 和更多真实样例。

另一次[超时试跑](../evaluation/evidence/receptionist_ablation_timeout_2026-09-23.json)保留了 `missing_facts` 的 `LLMTimeoutException`；公开副本只保留错误码、类型和阶段。

它在失败 attempt 汇总修复前生成，聚合 `llm_attempts=3` 漏算该失败例的重试，不能作为可靠总调用数。

### 当时的索引条件

- [早期 preflight](../evaluation/evidence/live_preflight_2026-09-23.json)：默认 Chroma collection 为空，默认 reranker 路径缺失。
- 随后在忽略的 `evaluation/live_results/` 构建六条隔离索引，只读复用已有 reranker。[manifest](../evaluation/evidence/six_article_index_manifest_2026-09-23.json)记录快照、模型 digest、1024 维与来源。
- 六条 JSON 后来纳入 Git；新克隆可取得构建输入，仍需自行准备模型、权重和索引。这是 snapshot 历史路径，不描述当前 full 默认。
- Chroma 读取也可能改写数据库；旧 `index_sha256` 是可变目录当时的字节快照，不能作稳定索引版本。[新 preflight](../evaluation/evidence/live_preflight_manifest_scope_2026-09-23.json)明确 hash 范围，另存 manifest、文档和 embedding 指纹。

公开索引 manifest 的 `build_command` 已脱敏，因此公开副本的整文件字节不同于当时运行产物。

[preflight 记录](../evaluation/evidence/live_preflight_manifest_scope_2026-09-23.json)中的 `index_build_manifest_sha256` 保留脱敏前原始 manifest 的哈希，不能用公开副本重算来验证。

manifest 内的快照、文档和 embedding 内容哈希未改；脱敏也未改案例输入、模型响应或评分。

### 五轮真实链路试跑

每轮都是同一组 3 条合成输入，按时间保留结果：

| 轮次与证据 | 条件 | 结果 |
| --- | --- | --- |
| [首轮](../evaluation/evidence/live_chain_default_timeout_initial_2026-09-23.json) | 默认 deadline，初版索引 | 2 例超时，1 例等待补充 |
| [第二轮](../evaluation/evidence/live_chain_long_timeout_initial_2026-09-23.json) | 180/90 秒，初版索引 | 1 例超时，2 例等待补充 |
| [第三轮](../evaluation/evidence/live_chain_with_source_failed_2026-09-23.json) | 30/15 秒，新索引补 source | 3 例超时；其中 1 例实际检索返回 5 条来源指纹 |
| [第四轮](../evaluation/evidence/live_chain_with_source_long_deadline_failed_2026-09-23.json) | 同新索引，180/90 秒 | 2 例等待补充，1 例检索后 TypeError |
| [修复后回归](../evaluation/evidence/live_chain_with_source_after_mapping_fix_2026-09-23.json) | 同索引、同 180/90 秒 | 3 例等待补充，0 执行错误；第 3 例检索成功后覆盖不足 |

五轮都没有完成咨询闭环。不同 deadline 的耗时/调用数不可直接比较；后三轮有 `llm_policy` metadata，前两轮以此处历史配置记录为准。这里的“检索成功”不能解释为法律质量得到确认。

第四轮的 `TypeError: unhashable type: 'dict'` 定位于当时的 LawRef 结构化提取失败后的确定性回退：六条快照的 `elements` 是含 `name` 的对象，而 `_build_element_to_law_mapping` 曾直接用对象作字典键。

已用 RED/GREEN 回归将映射键规范化为要件名称，并以真实六条快照运行当时的 LawRef 回退分支。

修复后同配置真实链回归未再出现该错误，并到达 `wait_for_user`；其后的律师审核、风险评估和服务方案路径仍未触发，不能据此声称端到端完成。

该回退分支已在后续版本移除，本段仅记录当时的验证结果。

完整链入口将生产 RAG 返回的前 5 条内容按排名记录为稳定指纹，可用时另记来源指纹（不保存正文或文件名），与最终 `applied_laws` 分开。重排失败回退时，这一顺序可能是原检索顺序；如果检索未运行，逐例 `retrieval_top_k_status` 为 `unavailable`。本次 5 条指纹只证明一次有条件的返回顺序，不能推断召回或法律质量。

```bash
backend/.venv/bin/python evaluation/run_live_eval.py preflight
backend/.venv/bin/python evaluation/run_live_eval.py run
```

索引 manifest 中的 `snapshot_git_tracked=false` 记录的是 2026-09-23 构建时状态；此后六条快照纳入版本控制。历史证据不回写，新的 clean clone 可取得快照文件，但不会自动得到当时的模型、权重或可变 Chroma 索引。

## 术语与结果读取

| 术语 | 含义 |
| --- | --- |
| deterministic baseline | 固定规则、可重复运行的基线 |
| input-only / gold | 预测只读输入 / 事后评分用的期望标注 |
| hit@5 | 项目期望关键词出现在前五候选中的比例；详见指标表 |
| MRR | 正确候选排名倒数的平均值，属于历史指标 |
| ablation | 固定任务下替换或移除组件的对照试验 |
| preflight | 正式运行前检查依赖是否齐备 |

`make eval` 正常完成时生成 `evaluation/results/summary.json` 与逐例结果；进程退出 0 表示 runner 执行完成，不表示所有质量指标满分。live 入口需检查实际执行阶段、逐例错误与汇总，不能只看模式标签。当前样例规模和标注范围不足以支撑开放域法律准确率或生产稳定性结论。
