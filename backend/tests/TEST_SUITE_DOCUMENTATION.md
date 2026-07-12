# 后端测试套件文档

## 目录结构

测试目录按 `backend/app` 的业务结构组织，跨模块场景集中放在 `integration/`：

```text
tests/
├── agents/              # app/agents
├── core/                # app/core
├── db/                  # app/db
├── errors/              # app/errors
├── integration/         # Redis、SQLite、工作流等跨模块场景
├── models/              # app/models 及 ORM 类型契约
├── orchestrator/        # app/orchestrator 及 LangGraph 类型契约
├── rag/                 # app/rag（含 md5_manager、reranker 子目录）
├── security/            # app/security
├── tools/               # app/tools
├── utils/               # app/utils
├── v1/                  # app/v1（router、service 子目录）
├── conftest.py          # 全局共享 fixture
└── factories.py         # 全局测试数据工厂
```

2026-07-12 重组前后均可收集到 **1114** 个测试。

## 测试覆盖率总览

**历史覆盖率记录：57%。本次仅重组目录，未重新生成覆盖率报告；当前收集基线为 1114 个测试。**

### 按模块覆盖率

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| **agents/** | | |
| human_alert.py | 100% | 完全覆盖 |
| risk_assessor.py | 96% | 仅缺 prompt 加载 fallback |
| receptionist.py | 91% | 仅缺部分 LLM 交互分支 |
| fact_digger.py | 86% | 缺少部分 prompt fallback 和异常分支 |
| service_planner.py | 86% | 缺少部分异常处理和边缘分支 |
| law_ref.py | 77% | 缺少 search_laws_by_rag 完整流程和部分异常分支 |
| **security/** | | |
| sensitive_filter.py | 100% | 完全覆盖 |
| disclaimer.py | 100% | 完全覆盖 |
| jwt.py | 98% | 仅缺 1 行边缘代码 |
| config.py | 93% | 仅缺 reload 路径 |
| rbac.py | 69% | 缺少中间件和部分依赖注入路径 |
| **errors/** | | |
| codes.py | 100% | 完全覆盖 |
| exceptions.py | 100% | 完全覆盖 |
| register.py | 100% | 完全覆盖 |
| handlers.py | 94% | 仅缺 1 行 |
| **core/** | | |
| success_response.py | 100% | 完全覆盖 |
| rate_limit.py | 52% | 缺少 FastAPI 依赖注入集成路径 |
| **state/** | | |
| consultation_state.py | 100% | 完全覆盖 |
| **models/** | | |
| user.py | 100% | 完全覆盖 |
| **schemas/** | | |
| models.py | 100% | 完全覆盖 |
| auth_schemas.py | 100% | 完全覆盖 |
| consultation_schemas.py | 100% | 完全覆盖 |
| **orchestrator/** | | |
| workflow.py | 78% | 缺少完整工作流执行和律师反馈路径 |
| **rag/** | | |
| sse_models.py | 100% | 完全覆盖 |
| reranker/factory.py | 100% | 完全覆盖 |
| retrievers/empty_retriever.py | 100% | 完全覆盖 |
| task_queue.py | 90% | 仅缺 4 行边缘代码 |
| retrievers/hybrid_retriever.py | 73% | 缺少 BM25 retriever 创建路径 |
| rag_service.py | 31% | 缺少完整 RAG 管道（需 ChromaDB） |
| reorder_service.py | 33% | 缺少模型加载和重排路径 |
| legal_text_splitter.py | 35% | 缺少完整文档拆分流程 |
| vector_store.py | 30% | 缺少 ChromaDB CRUD 操作 |
| reranker/causal_lm.py | 19% | 缺少模型推理路径 |
| reranker/cross_encoder.py | 31% | 缺少模型推理路径 |
| md5_manager/md5_store.py | 14% | 缺少数据库操作路径 |
| document_handler/processor.py | 15% | 缺少文件处理管道 |
| text_spliter.py | 24% | 缺少文本拆分流程 |
| **utils/** | | |
| logger.py | 85% | 仅缺部分初始化路径 |
| prompt_loader.py | 78% | 缺少部分 fallback 路径 |
| path_tool.py | 72% | 缺少部分路径解析 |
| config_loader.py | 67% | 缺少部分配置加载路径 |
| factory.py | 66% | 缺少模型创建路径 |
| llm_gateway.py | 20% | 缺少实际 LLM 调用路径 |
| file_handler.py | 16% | 缺少文件加载路径 |
| **v1/router/** | | |
| consultation.py | 39% | 缺少 WebSocket 和完整会话流程 |
| auth_router.py | 52% | 缺少 admin 管理端点 |
| knowledge_router.py | 42% | 缺少知识库管理端点 |
| consultation_history.py | 21% | 缺少历史记录查询端点 |
| knowledge_service.py | 23% | 缺少知识库服务逻辑 |
| lawyer.py | 15% | 缺少律师端点 |

### 未覆盖的核心业务逻辑

以下模块覆盖率低于 50%，包含核心业务逻辑但尚未充分测试：

1. **rag_service.py (31%)** — HyDE 生成、文档检索、摘要生成的完整管道
2. **vector_store.py (30%)** — ChromaDB 文档增删改查操作
3. **legal_text_splitter.py (35%)** — 法律文本按法条拆分、长法条分段
4. **md5_manager/md5_store.py (14%)** — 文件去重、MD5 校验
5. **document_handler/processor.py (15%)** — 文件上传处理管道
6. **llm_gateway.py (20%)** — LLM 调用网关（含重试、超时处理）
7. **file_handler.py (16%)** — PDF/Word/PPT 等文件加载
8. **consultation.py (39%)** — WebSocket 实时通信、完整会话生命周期
9. **lawyer.py (15%)** — 律师审核、干预端点
10. **knowledge_service.py (23%)** — 知识库文档管理服务

这些模块需要真实外部依赖（ChromaDB、LLM API、文件系统、WebSocket），建议后续通过集成测试或契约测试补充覆盖。

---

## 测试文件与业务模块对应关系

### 纯函数单元测试

| 测试文件 | 业务模块 | 测试数 | 覆盖内容 |
|----------|----------|--------|----------|
| security/test_sensitive_filter.py | security/sensitive_filter.py | 48 | PII 脱敏（身份证/手机/姓名/地址/车牌）、高风险检测、输入清洗 |
| agents/test_law_ref_pure.py | agents/law_ref.py | 67 | 中文数字转换、法条编号归一化、法条索引构建、RAG 验证补充、去重合并、构成要件映射 |
| agents/test_receptionist_pure.py | agents/receptionist.py | 18 | 用户类型关键词匹配、同意确认关键词检测 |
| agents/test_risk_assessor_pure.py | agents/risk_assessor.py | 21 | fallback 评估解析、关键风险提取、报告格式化、情节列表格式化 |
| agents/test_service_planner_pure.py | agents/service_planner.py | 12 | LLM 响应解析、服务方案结构提取、请求消息构建 |
| security/test_jwt.py | security/jwt.py | 11 | 密码哈希验证、token 创建/解码/刷新、过期/无效 token 处理 |
| security/test_disclaimer.py | security/disclaimer.py | 6 | 免责声明注入、幂等性 |
| errors/test_errors.py | errors/ | 12 | 错误码枚举、异常层次结构、异常处理器 |
| core/test_rate_limit.py | core/rate_limit.py + success_response.py | 9 | 滑动窗口限流、响应格式化 |
| rag/test_utils.py | rag/ 子模块 | 44 | 法条拆分器、混合检索权重、空检索器、SSE 事件、任务队列、重排工厂、文档去重 |
| utils/test_utils.py | utils/ | 15 | 日志 PII 过滤、提示词加载、路径工具 |

### Agent 节点集成测试

| 测试文件 | 业务模块 | 测试数 | 覆盖内容 |
|----------|----------|--------|----------|
| agents/test_fact_digger.py | agents/fact_digger.py | 9 | 首次交互、结构化事实提取、覆盖度分析、追问/摘要/高风险检测 |
| agents/test_receptionist.py | agents/receptionist.py | 10 | 欢迎语、同意确认、用户类型提取、完整接待流程 |
| agents/test_law_ref.py | agents/law_ref.py | 16 | 关键词搜索、RAG 检索、结构化法条提取、无匹配场景 |
| agents/test_risk_assessor.py | agents/risk_assessor.py | 4 | LLM JSON 响应、fallback 解析、状态更新 |
| agents/test_service_planner.py | agents/service_planner.py | 8 | 服务方案生成、报告输出、状态流转 |
| agents/test_human_alert.py | agents/human_alert.py | 7 | 高风险安抚、流程终止、免责声明注入 |

### 工作流测试

| 测试文件 | 业务模块 | 测试数 | 覆盖内容 |
|----------|----------|--------|----------|
| orchestrator/test_workflow.py | orchestrator/workflow.py | 46 | 条件边路由（consent/coverage/lawyer）、覆盖度计算、会话管理、节点执行 |

### API 端点测试

| 测试文件 | 业务模块 | 测试数 | 覆盖内容 |
|----------|----------|--------|----------|
| v1/router/consultation/test_api.py | v1/router/consultation.py | 18 | 创建会话、确认同意、获取状态、列表查询、关闭会话、认证/权限 |
| v1/router/test_auth.py | v1/router/auth.py | 10 | 注册、登录、刷新 token、获取当前用户、重复注册/错误密码 |
| security/test_rbac.py | security/rbac.py | 16 | 角色检查器、当前用户获取、律师角色验证、权限控制工厂 |

### 并发一致性测试

| 测试文件 | 业务模块 | 测试数 | 覆盖内容 |
|----------|----------|--------|----------|
| integration/test_concurrent_consistency.py | 多模块集成 | 10 | 并发会话创建、限流器并发、会话关闭一致性 |

---

## 测试基础设施

| 文件 | 用途 |
|------|------|
| tests/conftest.py | 6 个共享 fixture：mock_llm_gateway、sample_state、mock_db_session、mock_redis、auth_headers、test_app |
| tests/factories.py | 5 个测试数据工厂：make_consultation_state、make_law_data、make_risk_assessment、make_user_dict、make_applied_law |
