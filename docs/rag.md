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

## 两条检索路径

两条路径都从结构化事实构造查询，经过 JSON 核验与关键词补召回，再由 LawRef 读取条文并提交最终答案。

### 默认 `full`：独立公共索引

自然语言默认链路是：原查询向量 top20 + 中文 BM25 top20 → RRF 融合 → 受控 JSON 核验及关键词合并 → Qwen 统一重排 top5 → LawRef 读取与最终核验。可选 HyDE 默认关闭；精确条号直接读取，跳过扩展与重排。

full 使用独立公共 collection，与用户上传 collection 隔离。索引目录、manifest、语料与实际 embedding digest 必须匹配；构建操作见 [setup](setup.md#4-构建并连接全量法条索引)。专用配置、预算、设备与失败回退只在 [全量检索说明](knowledge/full_law_retrieval.md) 维护。

### `snapshot`：通用 RagService

```text
结构化事实
  → RagService(user_id, include_public=True)
      HyDE 改写查询
      Chroma 向量检索
      有语料时使用 BM25 / Ensemble
      去重、reranker 重排
  → 用六条 JSON 快照连接条号和项目标注
  → JSON 关键词补召回
  → LawRef 读取候选并核验最终答案
```

这条路径需要自行准备用户/公共文档库；六条 JSON 本身不是预建的 Chroma 索引。HyDE 失败时回退原查询，BM25 无语料时保留向量检索，reranker 失败时保留召回顺序。

两条路径的 `search_laws_by_rag()` 都先检查 `user_id`。旧 checkpoint 缺少用户身份时跳过 RAG，关键词检索仍可运行；不能用 `session_id` 代替身份。`full` 的公共索引与 `snapshot` 的用户权限过滤应分别理解。

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
