# 小规模离线评估

## 评估定位

`evaluation/` 是小规模离线评估集，用于检查事实字段、法条关键词、高风险转人工、追问和拒答/免责声明等基础契约。运行模式为 `offline-deterministic-baseline`，不调用 LLM、Ollama、ChromaDB、向量检索或 reranker。

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

## 历史结果快照

下表来自 2026-07-14 07:09 UTC 的历史工作树，不是当前分支结果。只有在目标提交重新执行 `make eval` 并保留完整输出后，才能更新为当前结果：

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

机器可读结果写入被忽略的 `evaluation/results/`。活动 runner 的法条关键词步骤读取代码指定的运行时结构化法条库；clean clone 不包含该数据，所以不能直接复现法条 hit@5。这项数据依赖必须单独准备、审计和核验。

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
