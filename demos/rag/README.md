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

clean clone 不包含运行时代码读取的 `law_knowledge/criminal_law_chapters.json`、Chroma 内容或模型权重，因此上述命令不会自动获得完整依赖。

`results/2026-07-14-ollama.json` 等现有 JSON 是历史环境快照，只能用于理解输出结构和当时的成功/失败条件。其原始输入数据不再随当前提交树发布，因此不能把该记录称为当前可复现结果、质量指标或可重复构建证明。新的结果应记录目标提交、依赖版本、数据来源和完整命令。
