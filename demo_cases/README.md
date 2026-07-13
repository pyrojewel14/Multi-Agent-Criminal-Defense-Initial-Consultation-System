# Demo case 数据契约

`demo_cases/*.json` 是 Phase 2 的标准展示用例，不是真实案件，也不是模型评测集。

每条 case 统一包含：

- `user_input`：用户原始输入。
- `expected_fact_fields`：与 `extract_case_facts` 工具一致的十个预期字段。
- `expected_law_keywords`：候选法条或检索结果中应出现的关键词。
- `should_follow_up`：是否应进入事实补充路径。
- `should_trigger_human`：是否应停止自动分析并转人工。
- `expected_final_output_type`：预期终态输出类型。
- `demo_fixture`：确定性 Demo 使用的固定节点输出，`data_source=demo_fixture` 不得描述成真实 RAG 检索结果。

从 `backend/` 运行完整普通咨询 Demo：

```bash
conda run -n Agent_dev python -m examples.demo_complete --case ordinary_assault
```

验证全部 case 的数据契约和三条控制路径：

```bash
conda run -n Agent_dev pytest tests/demo/test_demo_complete.py -q
```
