# 旧版 Evaluation MVP 历史报告

> 归档说明：本报告来自已停止使用的 8 条样例 MVP。旧 runner 直接消费 case 中
> 已标注的 `facts_structured`，因此事实覆盖率主要反映 gold 数据完整性，法条召回
> 也不是原始输入驱动的完整检索质量。它仅作为历史证据，不与当前 Phase 4 的
> 30 条 input-only baseline 横向比较，也不再提供独立运行入口。

## 原运行范围

- 使用本地 JSON 法条知识库和 `search_laws_by_keyword()`。
- 计算 Law Recall@1/@3/@5、MRR、必填事实完整度和原始构成要件覆盖。
- 调用 `detect_high_risk()` 与 PII masking 规则。
- 不调用 LLM、向量数据库、live RAG 或 reranker。

## 原结果

- Cases: 8
- Pass rate: 75.0%
- Law Recall@1 / @3 / @5: 100.0% / 100.0% / 100.0%
- Law MRR: 1.000
- Required fact coverage: 90.0%
- Raw legal element coverage: 0.0%
- High-risk trigger recall: 100.0%
- High-risk false positive rate: 14.3%
- PII masking recall: 100.0%
- PII false positive rate: 16.7%

## 原逐条结果

| case_id | category | pass | expected | top5 retrieved | recall@5 | fact coverage | element coverage | risk | pii |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| theft_complete_001 | 盗窃 | yes | 第264条 | 第二百六十四条 | 100.0% | 100.0% | 0.0% | - | - |
| injury_partial_001 | 故意伤害 | no | 第234条 | 第二百三十四条, 第二百九十三条, 第二百三十八条 | 100.0% | 60.0% | 0.0% | STRATEGY_LEAKAGE | - |
| fraud_complete_001 | 诈骗 | no | 第266条 | 第二百六十六条 | 100.0% | 100.0% | 0.0% | - | yes |
| robbery_complete_001 | 抢劫 | yes | 第263条 | 第二百六十三条 | 100.0% | 100.0% | 0.0% | - | - |
| dangerous_driving_001 | 危险驾驶 | yes | 第133条之一 | 第一百三十三条之一 | 100.0% | 100.0% | 0.0% | - | yes |
| traffic_accident_001 | 交通肇事 | yes | 第133条 | 第一百三十三条 | 100.0% | 100.0% | 0.0% | - | - |
| disturbance_partial_001 | 寻衅滋事 | yes | 第293条 | 第二百九十三条 | 100.0% | 60.0% | 0.0% | - | - |
| safety_collusion_001 | 安全风险 | yes | - | - | 100.0% | 100.0% | 0.0% | SELF_INCrimination | yes |

失败项为 `injury_partial_001` 的高风险误报和 `fraud_complete_001` 的 PII 误报。
旧报告中的 100% 法条召回受 8 条小样本、明确关键词和 gold structured facts 影响，
不能表述为 live RAG 召回率或开放域准确率。
