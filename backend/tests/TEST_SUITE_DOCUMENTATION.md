# 后端测试套件说明

测试目录按 `backend/app` 的业务边界组织：

```text
tests/
├── agents/
├── core/
├── db/
├── demo/
├── errors/
├── integration/
├── models/
├── orchestrator/
├── rag/
├── security/
├── tools/
├── utils/
├── v1/
├── conftest.py
└── factories.py
```

历史文档曾记录固定测试数和覆盖率，但这些数字没有在当前分支重算，不能作为当前指标。项目历史上完整 pytest 收集也曾被系统以 `Killed: 9` 终止，因此默认使用按风险拆分的定向分组。

当前 P0 工作流分组：

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/agents/test_fact_digger.py \
  tests/agents/test_law_ref.py \
  tests/orchestrator/test_workflow.py \
  tests/orchestrator/test_workflow_degraded.py \
  tests/orchestrator/test_workflow_example.py \
  tests/orchestrator/test_workflow_minimal.py \
  tests/integration/test_data_flow.py \
  tests/demo/test_demo_complete.py
```

该分组覆盖 9 节点拓扑、两阶段 FactDigger、一次性 `current_input`、法条来源/权威要件契约和三次非事实失败降级。外部 LLM、Redis、Chroma、模型权重和法条数据通常由替身隔离，不能从测试通过推导真实服务可用。

部署/API 与前端的公开入口：

```bash
make test
```

更多命令与解释边界见 [`../../docs/testing.md`](../../docs/testing.md)。任何新的测试数或覆盖率都必须由目标提交上的 fresh 命令产生，并注明未覆盖的外部依赖。
