# 离线评估

`evaluation/` 保存 30 条固定数据集、deterministic runner 和被忽略的本地生成结果。

从仓库根目录运行：

```bash
make eval
```

从 `evaluation/` 目录运行：

```bash
../backend/.venv/bin/python run_eval.py
```

目录职责：

- `cases.jsonl`：30 条统一 schema 的正式小型评估集。
- `run_eval.py`：只接收原始 `input` 的可审计离线 baseline。
- `results/`：可重新生成的 JSON、CSV 和 Markdown 结果，默认不提交版本控制。
- `history/`：不再作为当前指标使用的历史评估报告。

详细指标定义、历史结果快照和能力边界见 `docs/evaluation.md`。活动 runner 的法条关键词步骤依赖 clean clone 中缺失的运行时结构化法条库；运行前必须单独准备并核验。历史 MVP 使用已提供的
`facts_structured` 作为检索输入，不能与当前 input-only 结果直接比较。
