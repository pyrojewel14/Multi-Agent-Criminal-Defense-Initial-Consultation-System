# Phase 3 RAG 查询样例

`queries.json` 包含五条本地法律检索样例。结果把真实 live RAG 与确定性 JSON
关键词回退分别记录，不会把回退候选包装成向量检索结果。

从 `backend/` 运行离线可复现模式：

```bash
conda run -n Agent_dev python -m examples.rag_samples
```

运行真实 HyDE、Chroma + BM25/Ensemble、去重、reranker 与 JSON 验证子链：

```bash
conda run -n Agent_dev python -m examples.rag_samples --live-rag --compact
```

正式记录包括：

- `results/2026-07-13.json`：历史空 collection + Ollama 502 探测，真实向量结果为空。
- `results/2026-07-14-ollama.json`：经现有 `KnowledgeService` 入库 608 个公开切片后的 live 记录。
- `live_rag_top_k`：真实 live 子链经 reranker 后的 top-k。
- `json_fallback_top_k`：独立 JSON 关键词回退结果，不与 live 结果混写。
- `json_verification_passed`：预期 live 候选经过 `_verify_and_enrich_with_json()` 精确编号验证。

第二次运行实测 HyDE `qwen3.5:0.8b`、embedding `qwen3-embedding:0.6b`
和本地 Qwen reranker。此前 502 来自 Python HTTP 客户端读取 ClashParty 系统代理，
loopback 地址设置 `trust_env=False` 后恢复，并非缺少模型。五条命中不是准确率或召回率。
