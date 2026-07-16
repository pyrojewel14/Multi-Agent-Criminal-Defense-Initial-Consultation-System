# 离线评估

`evaluation/` 是项目唯一活动评估入口，保存 Phase 4 的 30 条固定数据集、
deterministic runner 和本机生成结果。

从仓库根目录运行：

```bash
conda run -n Agent_dev python evaluation/run_eval.py
```

从 `evaluation/` 目录运行：

```bash
conda run -n Agent_dev python run_eval.py
```

目录职责：

- `cases.jsonl`：30 条统一 schema 的正式小型评估集。
- `run_eval.py`：只接收原始 `input` 的可审计离线 baseline。
- `results/`：可重新生成的 JSON、CSV 和 Markdown 结果，默认不提交版本控制。
- `history/`：不再作为当前指标使用的历史评估报告。

详细指标定义、当前结果和能力边界见 `docs/evaluation.md`。历史 MVP 使用已提供的
`facts_structured` 作为检索输入，不能与当前 input-only 结果直接比较。
