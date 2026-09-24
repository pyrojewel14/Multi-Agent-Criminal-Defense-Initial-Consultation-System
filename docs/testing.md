# 测试与验证

本文给出公开、环境中立的验证命令。测试结果只覆盖所列文件与确定性依赖替身，不代表外部 LLM、Redis、Chroma、Docker build 或法律质量已经验证。

## 安装

从仓库根目录执行：

```bash
make install
```

也可以在已有项目虚拟环境中直接运行下列 `.venv/bin/python` 命令。

## P0 工作流契约

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/agents/test_fact_digger.py \
  tests/agents/test_law_ref.py \
  tests/test_main_startup.py \
  tests/orchestrator/test_workflow.py \
  tests/orchestrator/test_workflow_degraded.py \
  tests/orchestrator/test_workflow_example.py \
  tests/orchestrator/test_workflow_minimal.py \
  tests/integration/test_data_flow.py \
  tests/demo/test_demo_complete.py
```

该分组覆盖：

- 9 节点拓扑与 `fact_intake -> law_ref -> fact_digger` 顺序；
- `current_input` 单次消费和异常恢复不重放；
- 法条来源枚举与权威 `required_elements`；
- tracked 六条法条快照的来源/版本/字段/唯一性 preflight，以及其先于数据库与 Redis 启动；
- 多候选法条独立覆盖计算；
- `no_law_match` / `dependency_failure` 共享三次连续失败窗口；
- 交替失败、degraded 转人工以及人工 `revise_facts` 后开启新窗口；
- 确定性示例与数据流兼容性。

## LLM deadline 与结构化产物契约

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/utils/test_llm_gateway.py \
  tests/schemas/test_llm_artifacts.py \
  tests/agents/test_llm_artifact_integration.py \
  tests/agents/test_fact_digger.py \
  tests/agents/test_law_ref.py \
  tests/agents/test_risk_assessor.py \
  tests/agents/test_service_planner.py \
  tests/orchestrator/test_workflow_minimal.py
```

该分组使用 fake model 验证应用级总 deadline、单次 deadline、最多两次 attempt、瞬态错误分类、可注入退避抖动、四类 Pydantic schema、`tool_call` / `content_json` / `deterministic_fallback` 来源元数据、结构化 degraded 结果，以及 Risk 失败后跳过 ServicePlanner 的真实条件边。它不调用真实模型，也不证明供应商在线或模型输出质量。

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
- `.private/`、`.scratch/`、真实 `.env`、数据库、索引、模型与日志仍被忽略；
- 文档中没有完整运行 UUID、账号、令牌、固定密码或案件式可识别信息；
- 图片的扩展名与真实格式一致，且只包含明确合成数据；
- `git diff --check` 通过。

## 历史结果边界

仓库历史文档曾记录不同日期、不同依赖快照下的测试通过数、前端包大小和本地 RAG 样例。这些数字不是当前分支结果，已从活动说明中移除。任何新的“当前通过数”都必须在目标提交上重新执行对应命令后记录，并注明命令、日期和未覆盖的外部依赖。

历史上完整 pytest 收集曾被系统以 `Killed: 9` 终止，因此本项目优先使用按风险拆分的定向分组。定向分组通过不能表述为“全量测试通过”。

已知的旧 Phase 3 样例前提与当前最小快照不一致：`tests/rag/test_phase3_samples.py` 的离线全命中断言和第 133 条/第 133 条之一索引断言，依赖当前公开六条快照之外的法条。2026-09-25 在主工作树运行包含整个 `tests/rag` 的扩大后端分组得到 `970 passed, 2 failed, 104 warnings`，失败均为这两项；这不是全量后端通过。旧样例与历史结果仍保留作来源记录，后续应以独立测试 fixture 或经过来源核验的数据扩展解决，不能为了让断言通过而把未经审计的法条加入公开快照。
