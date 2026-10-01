# RAG 查询样例

`queries.json` 包含五条合成法律检索输入。运行器把 live RAG 与确定性 JSON 关键词候选分开记录，不会把规则回退包装成向量检索结果。

从 `backend/` 运行离线模式：

```bash
.venv/bin/python -m examples.rag_samples
```

运行 HyDE、Chroma、条件式 BM25、reranker 与 JSON 验证子链：

```bash
.venv/bin/python -m examples.rag_samples --live-rag --compact
```

clean clone 包含 `backend/data/law_knowledge/criminal_law_chapters.json` 六条最小验证快照，足以运行离线 JSON 关键词样例；快照外的两条样例按契约返回未命中。仓库不提供历史 live 运行所用的完整 Chroma 语料、索引或模型权重，因此 `--live-rag` 仍需自行准备这些依赖。

`results/2026-07-14-ollama.json` 等现有 JSON 是历史环境快照，只能用于理解输出结构和当时的成功/失败条件。当时的完整 Chroma 语料、索引和模型状态不随当前提交树发布，因此不能把该记录称为当前可复现结果、质量指标或可重复构建证明。新的结果应记录目标提交、依赖版本、数据来源和完整命令。
