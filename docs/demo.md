# 看一遍咨询流程：确定性 Demo

这篇用三个合成场景演示咨询流程、追问和高风险转人工。
安装依赖后即可运行，不需要启动 Redis、API 或模型服务。
要联调真实服务，读 [安装说明](setup.md)；能力范围见 [边界说明](limitations.md)。

## Demo 验证什么

运行器保留真实 LangGraph 节点、条件边和中断机制，把外部模型/检索输出替换为固定的合成数据（fixture）。同一场景可重复核对控制流。它没有执行真实模型推理或真实向量召回。

## 运行三个场景

先从仓库根目录安装依赖：

```bash
make install
```

此命令同时安装前端依赖，所以需要 Node.js/npm；Demo 执行本身只使用后端环境。已有后端环境时可跳过安装。

从仓库根目录运行，括号内切换到 `backend/`：

```bash
(cd backend && .venv/bin/python -m examples.demo_complete --case ordinary_assault)
(cd backend && .venv/bin/python -m examples.demo_complete --case missing_facts)
(cd backend && .venv/bin/python -m examples.demo_complete --case high_risk_collusion)
```

输出 JSON，并在脚本内校验场景与状态。预期关键字段如下；这些是输出节选，省略了中间轨迹和文字内容。

### 事实完整：到达律师审核

```json
{
  "case_id": "ordinary_assault",
  "mode": "deterministic_workflow_contract",
  "finished": true,
  "requires_human_intervention": false,
  "current_agent": "HumanReview",
  "output_type": "lawyer_reviewed_report"
}
```

路径为 `receptionist → fact_intake → law_ref → fact_digger → risk_assessor → service_planner → human_review`。审核决定也由 fixture 提供，真实运行需有权限的律师明确审核。

### 事实不足：中断等待补充

```json
{
  "case_id": "missing_facts",
  "finished": false,
  "requires_human_intervention": false,
  "current_agent": "FactDigger",
  "next_node": "fact_intake",
  "output_type": "follow_up_questions"
}
```

输出的 `trace` 中，`fact_and_law_contract` 阶段记录 `facts_coverage_rate=0.4` 和追问列表；这些字段不在顶层。

流程执行 `wait_for_user` 后中断。checkpoint（流程状态快照）已记录恢复时下一节点为 `fact_intake`，因此新输入会先刷新事实再重新检索。`current_agent` 是业务输出角色，不能用它推断底层待执行节点。

### 高风险：结束自动流程并提示人工处理

```json
{
  "case_id": "high_risk_collusion",
  "finished": true,
  "requires_human_intervention": true,
  "current_agent": "HumanAlert",
  "output_type": "human_intervention_notice",
  "alert_triggered": true
}
```

高风险检测后进入 `human_alert`，跳过后续法条、风险和服务方案生成。这里只生成工作流提示，没有发送外部通知。

## 怎样判断通过

命令正常退出、脚本断言通过，并且关键字段与上面的场景一致。场景数据在 `demos/consultation/cases/`；若修改了契约，要同时核对 fixture、运行器和测试。

可运行对应定向测试：

```bash
(cd backend && .venv/bin/python -m pytest -q \
  tests/demo/test_demo_complete.py \
  tests/consultation/test_workflow_example.py \
  tests/consultation/test_workflow_minimal.py)
```

预期：pytest 列出通过/失败数，全部通过时退出码为 0。测试解释见 [testing.md](testing.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError` | 检查后端依赖是否安装，模块命令是否在 `backend/` 执行 |
| 出现 `data_source="demo_fixture"` | 这是固定数据标记；实际来源及 `text_only` 说明见 [rag.md](rag.md#候选来源与覆盖资格) |
| Demo 通过，但真实咨询失败 | 分别核对 Redis、模型、全量索引、账号权限与最终响应；见 [setup.md](setup.md) |

## 边界与下一步

Demo 的合成数据不包含账号、令牌或真实案件。它证明固定场景的控制流，不证明模型质量、法律判断或跨进程恢复。了解节点和恢复规则，继续读 [工作流](workflow.md)；查看真实模型证据，读 [评估](evaluation.md)。
