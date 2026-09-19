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

## Clean-clone 限制

代码当前读取：

```text
backend/data/law_knowledge/criminal_law_chapters.json
```

当前提交树不包含 `backend/data/` 下的法条源或派生文件。历史上的 1 个 DOCX 与 3 个 JSON 只在旧提交中引入，不是运行时代码读取的 `law_knowledge/criminal_law_chapters.json`，文件之间存在重复且缺少来源/转换 manifest，内容版本也未覆盖后续法律修订，因此不作为当前可发布数据保留。

`.gitignore` 与 `.dockerignore` 均排除整个 `backend/data/`，所以 clean clone 和 Docker 镜像都不会复制这些来源文件或运行数据。

因此，干净克隆具有以下限制：

- 不含代码实际读取的结构化法条验证库；
- 不含预填充 Chroma collection；
- 不含 reranker 权重；
- 不含 SQLite、MD5 store 或上传资料；
- Docker 镜像不会从构建上下文复制整个 `backend/data/`。

缺少法条验证库时，RAG 候选不能成为 `rag_verified`，JSON 关键词补召回不可用，`law_search_status` 会进入依赖失败语义。连续 3 次非事实失败后工作流进入 `degraded` 并转 `HumanReview`。

## 运行依赖与 fallback

| 失败位置 | 当前行为 | 边界 |
| --- | --- | --- |
| HyDE | 使用原 query | 不证明 query 法律上准确 |
| BM25 无语料 | 退回纯向量检索 | 仍依赖 embedding 与 Chroma |
| Chroma 为空 | 返回空召回 | 不会自动创建可信语料 |
| reranker 失败 | 保留原顺序 | 排序质量未验证 |
| JSON 文件缺失/解析失败 | 无验证与关键词补召回 | 可信覆盖候选不可用 |
| RAG 编号未匹配 JSON | 标为 `rag_unverified` | 只供人工复核，不参与覆盖 |
| LLM 结构化失败 | 从已召回候选构造结果 | 不改变候选原始来源 |

## 数据治理要求

若要新增可复现的运行时法条数据集，应在独立任务中完成：

1. 明确原始来源、版本日期和适用范围；
2. 核验版权、许可与再分发条件；
3. 记录清洗、字段映射和校验流程；
4. 为 `article_number`、`required_elements` 和来源元数据建立自动校验；
5. 明确更新与撤回机制；
6. 重新验证 Docker、离线评估与 clean-clone 启动流程。

在这些步骤完成前，不应声称仓库或镜像自带完整法条库、Chroma 可直接复现，或 RAG 达到任何准确率/召回率指标。

普通文件删除只清理当前提交树，旧内容仍可能存在于既有 Git 对象、分支或远端引用中。本轮不改写历史、不 force-push；如未来出现合规清除要求，必须重新核对项目引用、远端引用和对象来源，并取得单独授权。
