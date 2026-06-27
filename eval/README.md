# Evaluation MVP

This MVP runs an offline baseline for the consultation system without calling an LLM or external vector database.

It evaluates:

- Keyword law recall against `backend/data/law_knowledge/criminal_law_chapters.json`.
- Fact coverage with `app.agents.fact_digger._analyze_coverage`.
- Required fact-field completeness from the gold case data.
- PII masking and high-risk trigger behavior from `app.security.sensitive_filter`.

Run from the repository root:

```bash
python3 eval/run_eval.py
```

Outputs:

- `eval/results/results.csv`
- `eval/results/report.md`

This is intentionally a deterministic baseline. RAG, rerank, and full workflow evaluation can be added behind the same case schema after API keys/vector-store fixtures are available.
