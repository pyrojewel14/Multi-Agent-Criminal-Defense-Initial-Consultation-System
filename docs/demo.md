# 确定性工作流 Demo

本 Demo 用合成输入和明确标记的 fixture 替代外部 LLM/RAG 输出，用于复现 LangGraph 节点、条件边、中断、恢复和状态契约。它不证明模型准确率、法条数据完整性或法律结论正确性。

## 示例数据

`demos/consultation/cases/` 包含三个合成场景：

| case | 目的 | 预期流程 |
| --- | --- | --- |
| `ordinary_assault` | 完整控制流 | 到达律师审核并由 fixture 决定结束 |
| `missing_facts` | 信息不足 | 停在 `wait_for_user` 并给出追问 |
| `high_risk_collusion` | 高风险短路 | 进入 `human_alert` |

这些 JSON 只包含合成事实与期望契约，不包含运行 UUID、账号、令牌或真实案件信息。

## 运行

先按 [setup.md](setup.md) 创建项目环境，然后从 `backend/` 运行：

```bash
.venv/bin/python -m examples.demo_complete --case ordinary_assault
.venv/bin/python -m examples.demo_complete --case missing_facts
.venv/bin/python -m examples.demo_complete --case high_risk_collusion
```

运行器会输出机器可读 JSON，并对 case ID、模式、最终节点、是否结束、是否需要人工介入和输出类型执行断言。普通场景采用当前 9 节点拓扑：

```text
receptionist
  -> fact_intake
  -> law_ref
  -> fact_digger
  -> risk_assessor
  -> service_planner
  -> human_review
```

覆盖不足的下一轮恢复顺序是：

```text
wait_for_user -> fact_intake -> law_ref -> fact_digger
```

## 自动化测试

```bash
cd backend
.venv/bin/python -m pytest -q \
  tests/demo/test_demo_complete.py \
  tests/orchestrator/test_workflow_example.py \
  tests/orchestrator/test_workflow_minimal.py
```

## 真实服务边界

真实 FastAPI/React 链路还依赖 Redis、模型配置、法条验证数据、Chroma 内容和有权限的账号。公共仓不提供预置管理员密码、令牌、数据库、索引或运行会话，也不保留含这些信息的截图。

如需做本地联调，应自行创建临时数据并在验证后清理；不要把账号、完整运行 ID、访问令牌、案件叙述或生成报告提交到文档、issue 或版本库。
