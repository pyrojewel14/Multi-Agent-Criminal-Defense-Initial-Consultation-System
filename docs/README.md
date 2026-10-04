# 文档导航

本页按阅读目的导读，技术主题由各篇统一维护；项目概览与快速开始见 [项目首页](../README.md)。

## 运行与联调

[安装与启动](setup.md) 说明依赖、模型、索引与部署 → [确定性 Demo](demo.md) 展示三种流程 → [API](api.md) 说明鉴权、会话与律师审核。读到 API 可完成接口联调；需要运行具体示例时进入 [demos 导航](../demos/README.md)。

## 开发与排错

[系统架构](architecture.md) 说明组件与存储职责 → [工作流](workflow.md) 说明节点、路由、中断和恢复 → [RAG](rag.md) 说明正文、标注与覆盖边界。随后按问题选择：

- 资产版本、标注与索引更新：[全量刑法说明](knowledge/full_criminal_law.md)。
- 召回、融合排序、重排、HyDE 与设备配置：[全量检索说明](knowledge/full_law_retrieval.md)。
- LawRef 模型产物、协议与失败记录：[响应协议诊断](knowledge/lawref_response_protocol.md)。

- 原文审计、增量字段、近期上下文与滚动摘要：[Memory 当前说明](memory/README.md)。

读到对应主题篇可定位配置、调用路径与诊断入口。[2026-10-02 Memory 分析](history/memory/2026-10-02-analysis.md) 是实施前快照，仅用于追溯当时判断；其他历史定位见 [历史索引](history/README.md)。

## 评估与能力判断

[能力与限制](limitations.md) 先界定能证明什么 → [测试](testing.md) 按风险选择检查 → [评估说明](evaluation.md) 解读证据 → [评测执行入口](../evaluation/README.md) 选择快照 baseline、全量工具、检索消融或真实链试跑。读到执行入口可按目标选择实验，结果仍需回到评估说明核对范围。

[第一批标注复核报告](knowledge/law_review/2026-09-30.batch1.md) 保留逐条建议及待确认事项，属于历史审查材料。

## 读文档时如何判断证据

服务健康检查、确定性 Demo、真实检索、真实 LawRef 和完整咨询闭环分别验证不同层次。查看结果时同时核对日期、配置、样例范围和实际执行阶段。

术语在各篇使用处解释；源码路径均相对于仓库根目录。历史证据保留当时的结果，当前行为以代码和目标环境的复跑结果为准。
