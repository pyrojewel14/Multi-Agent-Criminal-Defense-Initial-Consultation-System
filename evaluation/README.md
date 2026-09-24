# 评估入口

`evaluation/` 将 30 条固定样例的确定性离线 baseline 与真实模型/工作流试跑分开。两者使用不同样例、结果格式与输出目录，不可合并为同一法律质量指标。结果解读、历史失败及证据链接见 [评估说明](../docs/evaluation.md)。

从仓库根目录运行离线 baseline：

```bash
make eval
```

`run_live_eval.py` 有三个独立子命令：`preflight` 检查模型、六条法条快照、非空 Chroma collection 与 reranker 条件；`ablation` 比较 Receptionist 身份引导的固定模板和生产 LLM 路径；`run` 只在 preflight 通过后调用生产 LangGraph 链路。安装后可从仓库根目录运行：

```bash
backend/.venv/bin/python evaluation/run_live_eval.py preflight
backend/.venv/bin/python evaluation/run_live_eval.py ablation
backend/.venv/bin/python evaluation/run_live_eval.py run
```

真实链默认输出到被 Git 忽略的 `evaluation/live_results/`。若需要保存可公开的证据，可显式传入 `--output evaluation/evidence/<name>.json`；输出只保留受控状态、错误码和内容/来源指纹，不保存案件正文、原始模型输出或异常消息。索引构建器 `build_live_index.py` 拒绝覆盖已有目录；构建 manifest 标识快照、文档和 embedding 输入。Chroma 目录本身可在读取时变化，其字节哈希不是稳定的索引版本号。运行准备与结果边界见 [评估说明](../docs/evaluation.md)。

| 路径 | 用途 |
| --- | --- |
| `cases.jsonl` / `run_eval.py` | 30 条输入驱动的离线 baseline |
| `live_cases.jsonl` / `run_live_eval.py` | 三条合成 live 样例与独立真实链入口 |
| `build_live_index.py` | 为公开六条快照构建隔离评测索引 |
| `results/` / `live_results/` | 被忽略的可再生本地结果 |
| `evidence/` | 保留的受控公开证据，包含失败试跑 |
| `history/` | 不作为当前指标使用的历史报告 |

公开快照仅覆盖六条《刑法》条文，正式 baseline case 仍含快照外的 gold；法条 hit@5 只反映这一已声明范围。当前真实链证据没有完成咨询闭环，不能声称法律准确率、完整 RAG 召回率或生产稳定性。
