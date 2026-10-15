# 法条数据与检索

这篇解释法条从哪里来、默认检索走哪条路、哪些候选可用于覆盖计算。
读完可以区分“找到正文”“标注可用”和“模型给出有效最终答案”。
配置操作见 [安装说明](setup.md)，使用限制见 [边界说明](limitations.md)。

## 语料与标注

`LAW_KNOWLEDGE_PROFILE` 选择语料，默认 `full`。当前四个 JSON 都由 Git 跟踪，也在 Docker 构建白名单中：

| 资产（均在 `backend/data/law_knowledge/`） | 用途 |
| --- | --- |
| `criminal_law_full.json` | 默认语料：505 条记录，504 条有效正文；第 199 条保留删去记录，不进入搜索或索引 |
| `criminal_law_full.review.json` | 全量逐条复核队列；保留待复核状态 |
| `criminal_law_demo_annotations.json` | 32 条基础校对后的演示标注，默认加载到对应正文条目 |
| `criminal_law_chapters.json` | `snapshot` 回归入口：第 232、234、263、264、266、293 条，保持历史资产 |

默认全量加载中，原六条回归标注加 32 条 demo 标注，共 **38 条具备演示覆盖资格**。设置 `LAW_FULL_DEMO_ANNOTATIONS=off` 后，全量正文仍在，但只有原六条保留覆盖资格。版本、标注字段及更新流程见 [全量资产说明](knowledge/full_criminal_law.md)。

这里的“要件”是项目维护的覆盖标签。正文有官方来源记录，罪名标题、法定刑摘要、关键词与要件则是非官方项目标注，没有法律专家背书。38 条放行表示程序允许用于演示计算，不表示法律适用已得到确认。

正文来源和逐条 hash 保存在 `text_provenance`；快照版本整合至刑法修正案（十二）。不要把资产版本时点解释为每条的最初施行日期或当前个案的适用法。来源与时间字段详见 [审查与适用时间契约](knowledge/full_criminal_law.md#审查与适用时间契约)。

## 一条检索流程，分别绑定数据源

`LegalToolRegistry` 把模型的搜索查询交给 `LawSearchService`。服务收集原查询向量、BM25、可选 HyDE 和受控关键词通道，核验完整候选池后，交 `RetrievalPipeline.finalize()` 统一融合、可选重排并取最终 top5。Registry 只维护工具协议和已搜索/已读取约束，检索算法不会在工具层再执行一次。

`knowledge/wiring.py` 从 `ModelManager` 获取模型并组装服务，领域缓存由 `KnowledgeResources` 管理；法律模型限制和每请求完整性预检由 `law/binding.py` 执行。模型缓存与关闭职责见[架构说明](architecture.md#模型配置与资源所有权)。

`LAW_KNOWLEDGE_PROFILE` 选择 `full` 或六条 `snapshot` 资产。两种资产共用同一服务与 Pipeline；区别在资产、数据源和核验要求：

| 绑定 | 行为 |
| --- | --- |
| 默认 `full` | 只打开明确配置的独立公共 collection；每请求核验 manifest、正文、metadata、向量与实际 Embedding 模型 digest |
| `snapshot` 未绑定向量源 | 仅使用受控 JSON 关键词，向量状态为 `unconfigured` |
| `snapshot` 显式注入匹配 Pipeline | 用相应 collection 和 Embedding 运行同一流程；测试和历史 live 评测须明确绑定 |
| 上传文档 | `DocumentService` 绑定上传 collection，按用户和公共范围检索；不会自动成为法条来源 |

缺少认证 `user_id` 时法条服务跳过 RAG，仍可运行受控关键词。不能用 `session_id` 代替身份。full 索引缺失或不一致是依赖失败；不会改读上传库。构建操作见 [setup](setup.md#4-构建并连接全量法条索引)，排序、预算和设备配置见 [全量检索说明](knowledge/full_law_retrieval.md)。

BM25 文档和查询共用中文单字及相邻双字切分，英文和数字按词处理。每次检索先从 Chroma 读取当前授权快照，再使用按数据源、用户/公共范围、正文/metadata 指纹和分词版本分隔的有界缓存；每个组件最多保留 8 份。读取失败不能返回旧缓存。缓存省去重复分词与统计构建，不省略读库或 full 完整性核验。

向量召回、BM25 和 HyDE 只返回通道候选；RRF 使用通道排名，不直接混加向量距离、BM25 与模型分数。`vector` 模式关闭 BM25 和 HyDE，法条领域关键词仍进入共同候选池；它不会恢复历史来源优先排序。各原始来源及核验结果保存在 `origins`，选择可信代表不会洗白不一致的 RAG 来源。

Qwen 与 Cross-Encoder 分别实现同一评分协议，Pipeline 根据注入的组件决定是否重排。Qwen 使用官方 yes/no token 模板，保留完整查询并只截断文档；评分器返回输入顺序的分数，最终排序与截断由 Pipeline 完成。加载、设备、busy、超时及关闭由共享执行器管理，运行时仅加载本地文件。文件职责见 [Knowledge 模块组织](architecture.md#knowledge-模块如何组织)。

## 候选来源与覆盖资格

`data_source` 描述候选来源，`law_search_status` 描述整个 LawRef 节点结果，两者不是同一个字段。

| `data_source` | 含义 | 覆盖资格 |
| --- | --- | --- |
| `rag_verified` | 检索候选已连接到受控 JSON 条目；full 还校验版本、hash 和完整正文 | 标注通过门槛、要件非空才可计算 |
| `json_keyword` | 关键词补召回命中有覆盖资格的受控条目 | 要件非空才可计算 |
| `rag_unverified` | 检索结果未通过连接或一致性核验 | 不参与 |
| `llm_extracted` | 模型提出条号但未连接到受控条目 | 不参与 |
| `text_only` | 可读取正文，但项目标注不具备覆盖资格 | `coverage_eligible=false`，不参与 |

当前 `LawDataSource` 枚举只包含前四个值。`text_only` 是检索层单独处理的正文参考标记，不能作为 `LawSourceSchema` 的合法来源直接送入覆盖候选。Demo 另用 fixture 专属的 `demo_fixture`。

`CoverageCandidateSchema` 要求可信来源和非空 `required_elements`。模型只能从已经连接、读取的项目要件中选择匹配项，不能自行生成覆盖分母。匹配与否仍依赖模型判断；名称校验不能验证案件事实真实，也不能证明判断正确。

覆盖的计算方法和五种 `law_search_status` 统一见 [工作流](workflow.md#法条结果与失败窗口)。

## 依赖失败时会怎样

| 失败位置 | 当前处理 |
| --- | --- |
| full 索引未配置、manifest/语料不匹配、实际模型 digest 不符 | RAG 返回依赖失败标记；仍运行 JSON 关键词补召回 |
| RAG 失败、关键词没有候选 | 工具状态为 `dependency_failure` |
| RAG 失败、关键词仍有候选 | 工具状态为 `partial_dependency_failure`；模型可继续读取与核验 |
| full 检索正文与当前 JSON 不一致 | 候选变为 `rag_unverified`，不继承可信要件 |
| LawRef 正文候选中含未放行条文 | `text_only`，交给人工审核标注；不继续追问案件事实 |
| 最终答案无效、超时或耗尽预算 | 不补造成功候选；按节点结果进入有限重试或人工处理 |
| 语料/标注缺失或损坏 | 启动预检失败，服务拒绝启动 |

工具的 `partial_dependency_failure` 不是第六种 `law_search_status`。最终能否得到 `success` 还取决于模型是否提交合法答案及候选资格。

## 启动检查与数据更新

启动预检离线校验所选 JSON 和启用的标注，先于数据库与 Redis 初始化。它不生成或验证向量索引。单独运行命令见 [安装说明](setup.md#启动时检查哪些数据)。

更新语料时需核对来源、适用时间、许可/再分发依据、转换过程和逐条标注；再使用新版本目录构建索引。切换或回滚配置不会回滚已生成的报告、checkpoint 或业务记录。具体操作集中在 [全量资产说明](knowledge/full_criminal_law.md#构建更新与回滚)。

## 术语速查

| 术语 | 含义 |
| --- | --- |
| RAG | 先检索资料，再让模型结合资料生成答案 |
| embedding | 文本的向量表示，用于语义距离计算 |
| HyDE | 先生成假想回答，再用它辅助检索 |
| BM25 | 基于词项匹配的文本排序方法 |
| Ensemble | 融合多路检索排序 |
| reranker | 对召回候选再次排序的模型 |
| manifest / digest | 索引版本清单 / 模型内容指纹 |
| provenance | 一条数据的来源和转换记录 |

## 边界

505 条记录不是 505 条可自动适用的法律结论。公开仓提供文本与项目标注，不提供预建索引、模型权重、法律专家评测集或持续法律更新服务。历史检索样例只证明当时固定输入的运行结果。法律与评估限制见 [limitations.md](limitations.md)。
