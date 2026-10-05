# Agent Memory 第三阶段验证

验证日期：2026-10-04。第三阶段功能范围已由主线程于当日独立验收，验收更正与证据入口随公开提交 `15bbade` 保存。下文分别保留原运行和主线程复跑/重评分的范围；本次文档更正没有重新运行模型或回归。

本次使用固定合成案件验证真实事实提取、滚动摘要、有界上下文、HTTP/WebSocket 原文记录和服务停启恢复。法律检索、覆盖度节点及法律索引预检使用确定性替身；这不是完整咨询闭环或法律准确率评估。

## 可复现入口与证据

从仓库根目录执行，使用项目依赖环境并确保 Redis 可连接。每次选择新的临时工作目录：

```bash
python evaluation/run_memory_eval.py --mode deterministic \
  --work-dir .scratch/memory-example-offline \
  --report .scratch/memory-example-offline-public.json

python evaluation/run_memory_eval.py --mode live \
  --work-dir .scratch/memory-example-live \
  --report .scratch/memory-example-live-public.json

python -m pytest -q -o asyncio_mode=auto evaluation/test_memory_eval.py
```

真实模式读取部署配置的 Ollama 模型，不在命令中固定私人地址。入口仅接受固定合成场景；原始请求、模型响应、隔离数据库与服务日志保存在 `.scratch/`。公开报告不包含原文、摘要正文、地址、凭据或实际用户内容。

- [固定场景](../../evaluation/memory_cases.json)：短场景 1 轮，默认 recent=8；长场景 6 轮，明确使用评测 recent=2，包含补充、重复、冲突、更正和否认。
- [真实模型结果](../../evaluation/evidence/memory_live_2026-10-04.json)：Qwen3.5 9B、Q4_K_M，模型 digest、代码及输入指纹、实际 usage、耗时、游标和断言。
- [确定性结果](../../evaluation/evidence/memory_deterministic_2026-10-04.json)：同一服务停启流程，供应商响应为替身，actual usage 为 null。
- [评测程序](../../evaluation/run_memory_eval.py)、[评测宿主](../../evaluation/memory_eval_app.py)：只创建和停止自己的隔离 Uvicorn 进程，不增加生产端点，不清空 Redis。

模型 digest 为 `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。最终真实运行包含 17 次服务内模型调用、4 次同题回忆对照，共 21 次；合计实际输入 10633 tokens、输出 2217 tokens。这不是原单输入离线 baseline 的指标。

## 真实请求经过的路径

真实注册/登录取得 access token → 创建咨询 → 欢迎语落入原文档案 → 确认同意 → HTTP 或 WebSocket 消息 → application service 会话锁 → 用户原文提交业务 SQLite → checkpoint 消息回执 → 恢复真实 LangGraph → FactIntake 只提取本轮 → Ollama 原生 JSON schema → FactArtifact 严格校验 → 确定性案件字段合并 → 法律/覆盖替身 → 等待节点 → 实际回复提交原文档案 → 可选增量摘要 → checkpoint 保存 summary/case/recent → 返回回复。

随后使用 SIGTERM 正常退出该测试服务，重新打开相同两个隔离 SQLite 文件，通过真实 HTTP 读取同一咨询，再以另一 transport 重放成功末轮的相同幂等键。原文与图状态没有重复，继续新消息新增 2 条原文。日志确认两次真实 app lifespan 均完成清理；当前 Uvicorn 的 OS 退出码为 -15，报告保留该带符号值，不将退出码单独当成恢复证明。

短场景原文为 5 条，summary version=0/cursor=0/status=empty。长场景原文为 15 条，summary version=7/cursor=13/status=ready、recent=2；继续新消息后原文为 17 条，摘要推进至 version=8/cursor=15。摘要成功批次覆盖序号连续且不重复。case、summary、recent、raw 和 pending=fact_intake 在停启前后相同，真实权限/未同意检查分别返回 403，未认证返回 401。业务咨询列表可返回持久 workflow session ID；`/sessions` 的进程缓存列表仍不自动发现全部旧会话。

## 同题上下文对照

两种输入使用相同 system、当前问题、模型、temperature=0、thinking=false、输出上限；full-history 和 active-context 都执行相同脱敏。完整历史超预算时评测拒绝比较，不把被裁剪的历史当成完整历史。

| 场景 / 输入 | UTF-8 字节 | 保守输入估算 | 实际输入 tokens | 回忆耗时秒 |
| --- | ---: | ---: | ---: | ---: |
| 短 / full-history | 1726 | 2463 | 498 | 4.24 |
| 短 / active-context | 2607 | 3508 | 840 | 5.96 |
| 长 / full-history | 4526 | 6223 | 1157 | 10.43 |
| 长 / active-context | 2917 | 3644 | 788 | 11.72 |

长样例实际输入减少约 31.9%，短样例增加。当前小样例的回忆耗时没有改善，滚动摘要还有额外调用成本，不能据此宣称总体成本或延迟下降。估算采用保守 UTF-8 字节上界和协议预留，不是模型 tokenizer。完整审计来源和时间仍保存在 checkpoint；模型只接收字段值及争议/更正关系，减少重复审计 metadata。

## 修复与验证

真实试跑先暴露工具嵌套列表不合法、摘要 thinking 超时、空值示例被照抄、纯脱敏占位符产生伪冲突，以及上下文重复携带大量审计 metadata。原失败证据仍保留在私有临时工作目录。

- Ollama 本轮事实提取使用 FactArtifact 原生 JSON schema，保留严格业务校验；其他供应商保留工具调用。
- 可选摘要单次关闭 thinking，不修改共享模型；20 秒默认 deadline、输入/输出预算及失败降级保持。
- 事实 prompt 去掉空值示例，明确未核实陈述、否认及脱敏标记的含义。原“事实 prompt 必須有 JSON 示例”的测试参数取消，其他产物示例校验保留，事实 schema 由真实供应商边界测试校验。
- 案件合并的新增保护仅忽略标量字段中的纯脱敏占位符，不猜测原值或制造冲突；包含其他有效描述的内容仍保留。列表对象仍可能包含占位符，此保护不等于模型正确提取了重复输入。
- ContextBuilder 省略传给模型的重复来源/时间 metadata，持久化的原始案件版本及来源不变。

756 项受影响后端回归、2 项补充回归、39 项 evaluation 回归通过；Ruff 和 `git diff --check` 通过。两类固定场景的最终断言通过。回忆评分是粗粒度必要信息检查，不能代替逐项人工审查、事实正确率或法律质量评估。

主线程随后独立复跑21次真实调用，原评分唯一失败为长样例 active-context 的地点更正关键词：答案明确使用“修正”，旧评分仅识别“更正”。[原失败报告](../../evaluation/evidence/memory_parent_live_original_2026-10-04.json)保留原运行结果和运行时指纹；评分器增加明确“修正”同义词，城市、旧版本、时间、证据和否认条件不变。[独立重评分报告](../../evaluation/evidence/memory_parent_live_rescore_2026-10-04.json)仅重新读取原答案，唯一改变为该更正检查从 false 转为 true；没有重新运行模型或服务场景，不能把它表述成旧评分直接通过。返修后的非服务停启测试为34 passed、1 deselected，原回归计数未重新执行。

## 原始十项验收映射

| 用户要求 | 主要确定性测试 / 真实证据 |
| --- | --- |
| 1. 短对话不压缩 | `test_short_conversation_does_not_call_summary`；真实 short version=0 |
| 2. 长对话触发摘要 | `test_summary_incremental_success_failure_and_whole_message_budget`；真实 long version=7 |
| 3. active context 缩短 | `test_long_raw_archive_becomes_shorter_active_context`；同题长样例 1157→788 |
| 4. raw 不丢失 | `test_http_ws_share_raw_ids_and_no_duplicate_graph_on_replay`；真实内容/顺序/计数及恢复断言 |
| 5. 案件事实跨轮保存 | `test_intake_only_extracts_current_input_and_merges_null_lists`；真实 old_facts_retained |
| 6. 重复不无限新增 | `test_case_memory_preserves_empty_deduplicates_and_keeps_conflicts`；纯占位符保护；真实 repeat_no_new_version |
| 7. 新信息更新旧记忆 | 同一合并测试及具名实体版本测试；真实 correction_with_history / denials_retained |
| 8. 冲突不静默覆盖 | `test_named_party_updates_keep_conflicting_versions_without_duplicate_party`；真实 conflict_retained |
| 9. 服务/graph 重建恢复 | 两进程恢复测试、`test_real_uvicorn_restart_http_ws_is_offline_repeatable`；真实停启与继续消息 |
| 10. memory 失败安全降级 | `test_invalid_candidate_and_failed_summary_keep_raw_old_case_and_pending`；投影保存失败测试 |

第 10 项区分可选摘要/无效候选与必要 LLM 调用错误。可选失败保留旧记忆、原文及旧游标；必要主流程的 typed LLM 错误继续报告，不能以成功回复掩盖。

## 能力边界

主线程新鲜样例人工审阅发现，既有脱敏将合成输入“行程记录”误处理为“行[NAME-MASKED]”，该内容又进入提取及回忆。这是内容质量及既有脱敏限制，本阶段没有改动安全框架。新增占位符保护仅覆盖纯占位符标量字段，列表对象仍可能保留占位符；schema及粗粒度检查通过不表示记忆内容完全准确。功能验证与后续内容质量改进需分别评估。

仅验证合成小样例、单 worker、正常退出后重启。没有验证任意时刻 kill 的恢复、跨两库原子提交、crash exactly-once、多 worker、通用修复 API 或生产部署。摘要有损，模型仍可能遗漏、误提取或生成不准确措辞；schema 合法不等于事实正确。去重仍是精确值/具名实体规则，未实现语义消歧。正常状态变化通过冲突或明确更正保留历史，没有自动时间语义推理。后续扩大质量样本、存储与并发验证需另行安排。第三阶段当时不扩展 diagnostics，也不交付完整架构或面试材料；当前架构、预算与就绪说明统一见 [Memory 主说明](README.md)。
