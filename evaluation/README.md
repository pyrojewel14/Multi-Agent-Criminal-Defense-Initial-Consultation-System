# 评估入口

本页维护评测执行入口；技术背景与结果解读分别见 [全量检索说明](../docs/knowledge/full_law_retrieval.md) 和 [评估说明](../docs/evaluation.md)。先按实验范围选择入口，不合并不同样例和结果格式的指标。

| 实验范围 | 输入与入口 | 能检查的内容 |
| --- | --- | --- |
| 六条快照离线 baseline | `cases.jsonl` / `run_eval.py` | 30 条输入驱动的确定性流程回归 |
| 快照真实链试跑 | `live_cases.jsonl` / `run_live_eval.py` | preflight、身份引导消融与三条合成样例的 LangGraph 执行 |
| 全量正文分层评测 | `full_cases.jsonl` / `run_full_eval.py` | 离线工具、实际 RAG 或实际 LawRef；分别记录 |
| 全量检索消融 | `full_retrieval_cases.jsonl` / `run_full_retrieval_ablation.py` | 向量、混合、重排与 HyDE 的候选相关性、耗时及降级 |

## 六条快照 baseline 与真实链

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

真实链默认输出到被 Git 忽略的 `evaluation/live_results/`。若需要保存可公开的证据，可显式传入 `--output evaluation/evidence/<name>.json`；该入口的输出只保留受控状态、错误码和内容/来源指纹，不保存案件正文、原始模型输出或异常消息。索引构建器 `build_live_index.py` 拒绝覆盖已有目录；构建 manifest 标识快照、文档和 embedding 输入。Chroma 目录本身可在读取时变化，其字节哈希不是稳定的索引版本号。运行准备见 [评估说明](../docs/evaluation.md)。

## 全量正文与检索消融

`run_full_eval.py` 的三种模式分别调用离线关键词与读取工具、实际 RAG、实际 LawRef 节点；`--query-mode semantic` 使用样例中的自然语言查询，默认 `article` 使用条号查询。准备好 [全量资产与索引](../docs/knowledge/full_criminal_law.md) 后，从仓库根目录选择模式运行：

```bash
backend/.venv/bin/python evaluation/run_full_eval.py offline --output evaluation/results/full-offline.json
backend/.venv/bin/python evaluation/run_full_eval.py live-rag --query-mode semantic --output evaluation/live_results/full-rag.json
backend/.venv/bin/python evaluation/run_full_eval.py live-lawref --query-mode semantic --output evaluation/live_results/full-lawref.json
```

可用 `--case <id>` 选择案例、`--cases <path>` 指定同契约输入。全量检索消融使用独立的检索样例集，默认运行四组；配置与指标定义见 [复跑消融](../docs/knowledge/full_law_retrieval.md#复跑消融)：

```bash
backend/.venv/bin/python evaluation/run_full_retrieval_ablation.py --output evaluation/live_results/full-retrieval-ablation.json
```

该脚本读取已有配置，不构建索引或下载模型；输出文件必须尚不存在。可用 `--modes vector hybrid hybrid_rerank` 选择组别，`--limit <n>` 缩小探针范围，`--case-timeout <seconds>` 设置逐例时限。保留的 [2026-10-02 四组结果](evidence/full_retrieval_ablation_final_2026-10-02.json) 与 [独立核验](evidence/full_retrieval_verification_2026-10-02.json) 是当日证据，不能替代新配置的复跑。

## 结果与历史材料

| 路径 | 用途 |
| --- | --- |
| `build_live_index.py` | 为公开六条快照构建隔离评测索引 |
| `results/` / `live_results/` | 被忽略的可再生本地结果 |
| `evidence/` | 保留的受控公开证据，包含失败试跑 |
| `history/` | 不作为当前指标使用的历史报告 |

六条公开快照仅服务固定回归和隔离真实链实验，baseline case 仍含快照外的 gold；它不是项目全量资产范围。全量正文范围见 [资产说明](../docs/knowledge/full_criminal_law.md)。检索消融评价候选相关性，LawRef 评测评价节点执行；当前真实链证据没有完成咨询闭环，不能声称法律准确率、完整 RAG 召回率或生产稳定性。
