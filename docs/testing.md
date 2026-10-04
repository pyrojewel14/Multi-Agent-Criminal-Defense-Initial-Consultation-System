# 测试与验证

这篇按风险列出验证命令，帮助你选到与改动有关的测试。
读完可以分别检查工作流、法条数据、模型调用契约和前端构建。
要检查实际模型效果，读 [评估](evaluation.md)；测试证据范围见 [边界说明](limitations.md)。

## 如何选分组

| 改了什么 | 先跑什么 |
| --- | --- |
| 节点、恢复、路由 | 工作流契约组 |
| 全量语料、标注、索引 | 全量数据与索引组 |
| 模型超时、结构化输出 | LLM 契约组 |
| 原文审计、增量事实、摘要、上下文预算 | Memory 契约组 |
| 配置、API、前端 | `make test`、前端类型检查/构建 |
| 文档 | 链接、源码路径、命令目录与 `git diff --check` |

## 安装

从仓库根目录执行：

```bash
make install
```

也可以在已有项目虚拟环境中直接运行下列 `.venv/bin/python` 命令。

## 工作流契约

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/consultation/agents/test_fact_digger.py \
  tests/consultation/agents/test_law_ref.py \
  tests/integration/test_main_startup.py \
  tests/consultation/test_workflow.py \
  tests/consultation/test_workflow_degraded.py \
  tests/consultation/test_workflow_example.py \
  tests/consultation/test_workflow_minimal.py \
  tests/integration/test_data_flow.py \
  tests/demo/test_demo_complete.py
```

该分组覆盖：

- 9 节点拓扑与 `fact_intake -> law_ref -> fact_digger` 顺序；
- `current_input` 单次消费和异常恢复不重放；
- 法条来源枚举与权威 `required_elements`；
- 当前所选语料的启动预检先于数据库与 Redis；六条 snapshot 的来源/版本/字段/唯一性回归另保留；
- 多候选法条独立覆盖计算；
- `no_law_match` / `dependency_failure` 共享三次连续失败窗口；
- 交替失败、degraded 转人工以及人工 `revise_facts` 后开启新窗口；
- 确定性示例与数据流兼容性。

## 全量数据与索引

从仓库根目录运行：

```bash
(cd backend && .venv/bin/python -m pytest -q \
  tests/knowledge/test_full_law_corpus.py \
  tests/knowledge/test_demo_law_annotations.py)
backend/.venv/bin/python -m pytest -q \
  evaluation/test_full_index.py evaluation/test_full_eval.py
```

前一组验证 full/snapshot、标注放行和真实编译图路由；后一组用合成向量验证索引完整性、manifest、模型标识与评测契约。预期 pytest 报告全部通过、退出码为 0。它们不需要真实模型，不证明真实语义检索质量。

## Memory 与 full 检索契约

从 `backend/` 选择对应组运行，命令仅列执行入口，本轮文档整理未执行：

```bash
.venv/bin/python -m pytest -q tests/consultation/test_memory.py \
  tests/consultation/test_memory_persistence.py tests/consultation/test_memory_live_contract.py
.venv/bin/python -m pytest -q tests/knowledge/test_full_law_retrieval.py \
  tests/knowledge/test_full_law_reranker_device.py
```

Memory 组核对原文/回执、增量字段、游标与预算；名称中的 `live_contract` 仍需核对具体替身和外部依赖范围。full 组核对召回、重排协议与设备失败边界。[阶段 3 定向验证交付](memory/phase3-verification.md)待对应主线程独立验收，真实模型质量与耗时另按 [评测](evaluation.md) 执行；源码职责见 [Memory](memory/README.md) 和 [全量检索](knowledge/full_law_retrieval.md)。

## LLM deadline 与结构化产物契约

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/infrastructure/llm/test_llm_gateway.py \
  tests/consultation/schemas/test_llm_artifacts.py \
  tests/consultation/agents/test_llm_artifact_integration.py \
  tests/consultation/agents/test_fact_digger.py \
  tests/consultation/agents/test_law_ref.py \
  tests/consultation/agents/test_risk_assessor.py \
  tests/consultation/agents/test_service_planner.py \
  tests/consultation/test_workflow_minimal.py
```

该分组使用 fake model 验证应用级总 deadline、单次 deadline、最多两次 attempt、瞬态错误分类、可注入退避抖动、四类 Pydantic schema、实际产物通道 `tool_call` / `content_json` 的来源元数据、结构化 degraded 结果，以及 Risk 失败后跳过 ServicePlanner 的真实条件边。它不调用真实模型，也不证明供应商在线或模型输出质量。

## 编译检查

```bash
cd backend
.venv/bin/python -m compileall -q app examples tests
```

## API 与部署定向组

根目录的 `make test` 运行一个较小的后端配置/JWT/数据库/OpenAPI 分组和全部前端 Vitest：

```bash
make test
```

需要单独检查前端类型与构建时：

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

这些命令不会自动验证真实 Ollama/云模型、已填充 Chroma、法条来源许可或浏览器端到端操作。

## Markdown 与发布检查

发布前至少检查：

- Markdown 本地相对链接均指向存在且准备跟踪的文件；
- 私有材料、临时草稿、真实 `.env`、数据库、索引、模型与日志仍被忽略；
- 文档中没有完整运行 UUID、账号、令牌、固定密码或案件式可识别信息；
- 图片的扩展名与真实格式一致，且只包含明确合成数据；
- `git diff --check` 通过。

## 历史结果边界

仓库历史文档曾记录不同日期、不同依赖快照下的测试通过数、前端包大小和本地 RAG 样例。这些数字不是当前分支结果，已从活动说明中移除。任何新的“当前通过数”都必须在目标提交上重新执行对应命令后记录，并注明命令、日期和未覆盖的外部依赖。

历史上完整 pytest 收集曾被系统以 `Killed: 9` 终止，因此本项目保留按风险拆分的定向分组。定向分组通过不能表述为“全量测试通过”。2026-09-30 在 `backend/` 的隔离环境运行了 `python -m pytest -q --tb=short`（省略本地环境包装命令），结果为 `1349 passed, 345 warnings`；该结果仍不证明外部模型、RAG 索引或部署链路可用。

2026-09-25 的 RAG 测试分组曾得到 `970 passed, 2 failed, 104 warnings`。

两项失败来自旧 Phase 3 样例要求公开六条快照之外的法条，以及第 133 条/第 133 条之一索引断言依赖缺失的条文。

这些历史结果仍保留作来源记录；当前 `tests/knowledge/rag/test_phase3_samples.py` 对快照外样例明确验证未命中，并用独立测试 fixture 验证条号索引，未向公开快照加入未经审计的法条。

上述两项已包含在 2026-09-30 通过的全量后端测试中。

## 怎样报告结果

写明日期、命令、退出码、通过/失败数及是否使用替身。编译检查预期静默退出 0；类型检查、构建和 pytest 分别报告自己的结果。当前输出才能支持当前通过声明，历史数字仅供追溯。
