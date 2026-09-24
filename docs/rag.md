# RAG、法条来源与覆盖边界

本文描述当前代码路径与 clean-clone 限制，不把历史本地索引或模型运行记录当作当前仓库能力。

## 调用链

```text
facts_structured
  -> search_laws_by_rag()
       HyDE（失败时回退原 query）
       Chroma 向量检索
       条件式 BM25 / Ensemble
       去重与 reranker（失败时保留原顺序）
  -> 结构化法条 JSON 编号验证
  -> JSON 关键词补召回
  -> LLM 结构化候选或直接候选回退
  -> applied_laws + data_source + required_elements
```

RAG 权限范围使用 `state.user_id`，不能用 `session_id` 代替用户身份。旧 checkpoint 缺少 `user_id` 时跳过私有 RAG，避免跨用户检索。

## 来源枚举

`LawDataSource` 只允许：

| 来源 | 含义 | 可参与覆盖度 |
| --- | --- | --- |
| `rag_verified` | RAG 编号与受控 JSON 条目成功连接 | 是，且必须有非空 `required_elements` |
| `json_keyword` | 受控 JSON 关键词补召回 | 是，且必须有非空 `required_elements` |
| `rag_unverified` | RAG 候选未通过 JSON 编号验证 | 否 |
| `llm_extracted` | LLM 候选未连接到受控条目 | 否 |

`CoverageCandidateSchema` 同时校验来源与非空 `required_elements`。LLM 返回的 `elements_matched` 只能在已连接候选的权威要件内取子集；无法连接时不得保留模型自报的要件作为覆盖分母。

## Tracked 最小验证快照

代码当前读取：

```text
backend/data/law_knowledge/criminal_law_chapters.json
```

该文件由 Git 跟踪并随 clean clone 直接可用；运行时不联网下载或更新。`.gitignore`、`backend/.gitignore` 与 `.dockerignore` 只精确放行这一文件，其他 `backend/data/` 内容仍被排除。

快照元数据：

- dataset：`criminal-law-core@2026-09-21.1`；
- 核验日期：2026-09-21；
- 官方条文版本：合并至《中华人民共和国刑法修正案（十二）》，对应修正内容自 2024-03-01 施行；
- 官方文本：[上海市发展和改革委员会发布的《中华人民共和国刑法》合并文本](https://fgw.sh.gov.cn/cmsres/cb/cba3dfde07c1437c9c55f9ec15f91e11/51e0f21bda99e0b7e33eaa22e4cc9f4e.pdf)；
- 再分发依据：[《中华人民共和国著作权法》第五条](https://www.npc.gov.cn/c2/c30834/202011/t20201119_308796.html)规定法律、法规及其他具有立法性质的文件不适用著作权法；
- 覆盖范围：第 232、234、263、264、266、293 条，共六条。

### 官方文本与项目标注分层

| 字段 | 来源 | 能否解释为官方法律文本 |
| --- | --- | --- |
| `article_number`、`content` | 上述政府发布的合并文本 | 是，但仍需按快照版本核对时效 |
| `title`、`base_sentence` | 项目维护的索引名称与法定刑摘要 | 否 |
| `elements` / `required_elements` | 项目维护的确定性覆盖标签 | 否 |
| `charge_tags`、`common_keywords` | 项目维护的检索标签 | 否 |

项目标注没有法律专家背书，不是完整构成要件体系，也不能作为定罪、量刑或法律意见。快照仅让编号验证、关键词补召回和 deterministic tests 在同一小范围数据上可复现。

### 启动 preflight

FastAPI lifespan 在初始化数据库和 Redis 前显式调用 `preflight_law_knowledge()`。以下任一情况都会抛出明确错误并阻止启动：

- 文件缺失或 JSON 损坏；
- 顶层来源、版本、核验日期、再分发依据或标注边界缺失；
- chapter/article 必填字段或项目标注为空；
- 归一化后出现重复法条编号；
- metadata 声明的覆盖清单与实际条目不一致。

这项 preflight 不访问网络。若快照在已启动进程中被删除或替换，下一次未命中缓存的加载同样会失败，不会静默返回空 chapters。

clean clone 和 Docker 镜像仍不包含预填充 Chroma collection、reranker 权重、SQLite、MD5 store、上传资料或完整刑法库。没有向量索引时，只有六条快照范围内的 JSON 关键词补召回可用。

## 运行依赖与 fallback

| 失败位置 | 当前行为 | 边界 |
| --- | --- | --- |
| HyDE | 使用原 query | 不证明 query 法律上准确 |
| BM25 无语料 | 退回纯向量检索 | 仍依赖 embedding 与 Chroma |
| Chroma 为空 | 返回空召回 | 不会自动创建可信语料 |
| reranker 失败 | 保留原顺序 | 排序质量未验证 |
| JSON 快照缺失/解析或 schema 失败 | 启动 preflight 直接失败 | 不允许以空验证库提供服务 |
| RAG 编号未匹配 JSON | 标为 `rag_unverified` | 只供人工复核，不参与覆盖 |
| LLM 结构化失败 | 从已召回候选构造结果 | 不改变候选原始来源 |

## 数据治理要求

若要扩展当前六条快照，应在独立任务中完成：

1. 明确原始来源、版本日期和适用范围；
2. 核验版权、许可与再分发条件；
3. 记录清洗、字段映射和校验流程；
4. 为 `article_number`、`required_elements` 和来源元数据建立自动校验；
5. 明确更新与撤回机制；
6. 重新验证 Docker、离线评估与 clean-clone 启动流程。

当前快照已经满足其六条范围内的 clean-clone 可复现性，但不应声称仓库或镜像自带完整法条库、Chroma 可直接复现，或 RAG 达到任何准确率/召回率指标。

历史上的 1 个 DOCX 与 3 个 JSON 只在旧提交中引入，文件之间存在重复，缺少来源/转换 manifest，内容版本也未覆盖后续法律修订，因此没有恢复为当前来源。旧内容仍可能存在于既有 Git 对象、分支或远端引用中；本任务不改写历史、不 force-push。如未来出现合规清除要求，必须重新核对项目引用、远端引用和对象来源，并取得单独授权。
