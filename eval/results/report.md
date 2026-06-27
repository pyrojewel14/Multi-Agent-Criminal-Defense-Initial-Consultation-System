# Evaluation MVP Report

## Summary

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

## Case Results

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

## Failing Cases

- injury_partial_001: high-risk false positive (STRATEGY_LEAKAGE)
- fraud_complete_001: PII mask false positive

## Notes

- This MVP is an offline deterministic baseline.
- Keyword recall is not a substitute for RAG quality; add vector/RAG runs as the next evaluator mode.
- Raw legal element coverage is diagnostic only because current law elements are natural-language phrases, not FactDigger field keys.
- ServicePlanner report quality still needs human or LLM-judge rubric scoring.
