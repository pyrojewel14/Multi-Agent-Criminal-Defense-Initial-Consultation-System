# 全量刑法正文、标注与独立索引

这篇提供全量刑法资产、项目标注和索引的维护说明。
读完可以核对版本、构建新索引、切换配置并找到历史证据。
首次启动读 [setup.md](../setup.md)，检索概览读 [rag.md](../rag.md)，使用限制见 [limitations.md](../limitations.md)。

## 先看当前状态

- 默认 `full`：505 条记录、504 条有效正文；六条回归标注加 32 条 demo 标注，38 条可用于演示覆盖计算。
- 正文来源和项目标注分层；演示放行没有完成法律人工审核。
- 向量索引独立构建，目录与相邻 manifest 成对保留；不进入 Git 或镜像。
- 当前检索算法与专用配置由 [全量检索说明](full_law_retrieval.md) 维护；本页负责资产、标注、版本与索引维护。LawRef 模型结果按 [协议诊断](lawref_response_protocol.md) 中的历史日期和范围读取。

阅读路线：[资产与版本](#资产与版本) → [标注与时间](#审查与适用时间契约) → [构建与回滚](#构建更新与回滚)。历史实验集中在文末。

## 资产与版本

- `backend/data/law_knowledge/criminal_law_chapters.json`：六条固定回归资产，保持原始字节；旧 `evaluation/build_live_index.py` / `run_live_eval.py` 的索引和评测契约保留。
- `backend/data/law_knowledge/criminal_law_full.json`：`criminal-law-full@2026-09-30.2.integration1`，16章、505条记录，504条有效正文；第199条保留删去记录，不进入搜索或索引。
- `backend/data/law_knowledge/criminal_law_full.review.json`：505条逐条复核队列，包含正文hash、待复核字段、列举项和跨条引用线索。6条标注沿用项目回归契约；其余499条标注保持待复核。任何条文的正文法律人工审查均未自动完成。

正文来源为[政府整合版刑法 PDF](https://fgw.sh.gov.cn/cmsres/cb/cba3dfde07c1437c9c55f9ec15f91e11/51e0f21bda99e0b7e33eaa22e4cc9f4e.pdf)，整合至修正案（十二）。

第169条标题单独对照[两高罪名补充规定（八）](https://www.court.gov.cn/fabu/xiangqing/424452.html)，状态为 `source_checked`；这没有使该条罪名标签、要件或法定刑提议一起通过复核。

两层 `.gitignore` 和 `.dockerignore` 已为正文、复核队列及demo标注JSON放行。提交同时包含新资产与代码，clean clone可以加载全量JSON。`backend/Dockerfile` 的 `COPY backend/ ./` 会携带正文和复核清单，二进制索引仍独立生成、挂载，不进入 Git 或镜像。此项仅验证打包路径；本任务没有实际 Docker build/up/health 或生产部署证明。

- `backend/data/law_knowledge/criminal_law_demo_annotations.json`：32条基础校对后的独立demo标注，版本 `demo-reviewed-2026-10-01-v1`，加载层为 `demo-reviewed-v1`、状态 `demo_ready`、`official=false`。

## 审查与适用时间契约

| 字段 | 来源和状态 | 运行时用途 |
| --- | --- | --- |
| `article_number`, `content`, `text_provenance` | 官方文本来源、逐条hash，`source_attributed` / `program_checked`；法律人工复核 `pending` | 有效条文可召回、完整读取 |
| `annotations.title` | 人工标题表，通常 `pending`；169条为 `source_checked` | `display_title`仅作标有状态的检索展示 |
| `annotations.charges` | 未核实的标签提议，`pending` | 不用于自动罪名适用 |
| `annotations.elements` | 自动派生提议，`pending` | 不进入覆盖分母 |
| `annotations.penalty` | 自动派生法定刑或引用提议，`pending` | 不作为可靠刑罚结论 |
| demo运行标注 | Codex基础校对，`demo_ready`；法律人工审查仍为`pending` | 可进入LawRef和覆盖分母，无需律师确认 |
| 六条同正文标注 | 来源绑定到固定快照hash，`project_regression` | 继续原项目要件契约，不代表完整法律专家审核 |

每层 `applicability.consolidation_effective_from=2024-03-01` 表示整合版本时点，`effective_to=null` 表示该资产未给出结束时间，**不能解释为每条最初施行日期或2026年的现行法已核查**。本实现不自动选择个案行为时法、不处理从旧兼从轻的法律判断。

未放行条文的提议原值只保存在 `annotations.value`，其运行时 `elements=[]`、`base_sentence=""`、`charge_tags=[]`；占位不会靠非空验证取得可信状态。

`text_structure`只按正文保留列举项和跨条引用原文，状态 `syntax_only`，没有把并列、替代、加重情形自动解释为完整构成要件。

总则、程序和多罪名条文都保留完整正文；第133条之一的四项、第149条的跨条适用、第277条的多个情形有专门回归。

已放行32条由独立demo资产提供要件与法定刑，原值保存在`original_annotations`。

替代分支合成一个“任一适用分支”条件，避免要求所有替代项同时满足；其事实支持由已读取条文的LawRef判断，不套用旧关键词规则。

## LawRef 与覆盖计算

应用默认 `full`，`snapshot` 显式加载六条固定快照。`LAW_FULL_DEMO_ANNOTATIONS=off` 可恢复全量正文中仅六条参与覆盖的历史模式；demo 资产随 backend 打包，`LAW_DEMO_ANNOTATIONS_PATH` 可指定另一份标注，`LAW_FULL_CORPUS_PATH` 可选择明确版本语料。

未知配置、缺失资产、重复/错误条号、来源或 hash 损坏都失败关闭。进程缓存按配置、绝对路径、mtime 与大小区分；不要原地覆写运行中资产，更新后切到新路径并重启服务。

`run_eval.py` 的离线基线显式加载 snapshot；`run_live_eval.py` 固定 snapshot 与原六条 live 索引；`run_full_eval.py` 固定 full。各入口不会因应用默认变化混用语料，范围见 [评估说明](../evaluation.md)。

LegalToolRegistry 可读取六条之外的完整正文、来源与标注状态；未复核条文返回空要件。检索候选连接语料 hash、版本和完整正文，不一致时标为 `rag_unverified`。模型只能选择已召回且已读取的条文；正文参考保存为 `law_text_candidates`，与 `applied_laws` 分离。覆盖资格与人工路由由 [RAG 概览](../rag.md#候选来源与覆盖资格) 和 [工作流](../workflow.md#法条结果与失败窗口) 统一解释。

原六条与 32 条基础校对的 demo 标注可进入演示 `applied_laws`，demo 标记和版本来源随候选传递。尚未放行的正文参考触发 `annotation_review_required`，直接进入 HumanReview；不通过重复追问案件事实来填补标注缺口。此门槛不等于 demo 已完成全 505 条法律审查。

历史确定性编译图验证过危险驾驶 demo 候选进入覆盖后路由到 RiskAssessor；模型响应在测试中替代，不能作真实 LLM 成功证明。详细日期与验证数字保留在下方历史证据区。

## 构建、更新与回滚

以下命令从仓库根目录执行，使用 `make install` 创建的 `backend/.venv`。

导入新的供应候选，输出到**不存在的新版本路径**，同时生成复核队列：

```bash
backend/.venv/bin/python evaluation/import_full_corpus.py \
  --candidate /path/to/candidate.json \
  --output backend/data/law_knowledge/criminal_law_full.NEW.json \
  --review-queue backend/data/law_knowledge/criminal_law_full.NEW.review.json
```

现有导入契约针对修正案（十二）的452个基础条号与53个衍生条号。法律版本发生变化时，需要同步审查来源、条号契约与复核队列，不能仅更换版本字符串。新路径也需明确纳入两层Git白名单和Docker白名单。供应候选的hash记录在资产metadata；重复导入相同输入会生成相同输出。本交付没有把供应方构建脚本或个人目录依赖加入运行时。

构建独立全量公共Chroma索引；目标目录和相邻manifest均不得存在：

```bash
backend/.venv/bin/python evaluation/build_full_index.py \
  --index-dir backend/data/law_indexes/full-law-2026-09-30.2.integration1-qwen3 \
  --collection criminal_law_full
```

构建入口从 `backend/.env`读取已有模型配置（环境变量优先），直连Ollama并读取实际embedding digest，每条有效正文一个文档，不包含提议标注、不截断正文。manifest记录语料版本和hash、文档ID与来源hash、模型/digest、维度和float32向量hash。构建后读回全部504条核验；失败目录没有可用manifest，应保留失败证据并另选新目录重建。

测试用合成向量标记 `deterministic-test`，运行时拒绝使用。运行时每次核验整份manifest、Chroma文档/来源/向量以及实际Ollama模型digest，再按条号精确读取或执行当前 [混合检索与统一重排](full_law_retrieval.md#真实调用路径)。这是优先保证可审查性的504条实现，尚未做大规模并发性能优化。

在新进程中配置独立公共索引（full 已是默认，仍可显式声明）：

```bash
export LAW_KNOWLEDGE_PROFILE=full
export LAW_FULL_INDEX_DIRECTORY="$PWD/backend/data/law_indexes/full-law-2026-09-30.2.integration1-qwen3"
export LAW_FULL_INDEX_COLLECTION=criminal_law_full
export LAW_FULL_EMBEDDING_DIGEST='<manifest和/api/tags中的实际digest>'
```

该索引与通用 `CHROMA_*` 用户上传索引隔离；full 使用自己的混合召回、Qwen 重排及可选 HyDE，配置集中见 [全量检索说明](full_law_retrieval.md)。没有 `user_id`时仍跳过RAG，JSON正文检索可以继续；索引未配置或不一致时记录RAG依赖失败，不自动改读旧六条索引。

更新应使用新语料文件、新目录和新manifest；先完整核验，再检查运行中服务和会话保存方式后切换。回滚是恢复上一套明确语料路径、索引目录、collection和digest后重启，或改回 `LAW_KNOWLEDGE_PROFILE=snapshot`。回滚配置不会回滚checkpoint、数据库或已生成报告。

2026-10-01 索引与相邻 manifest 已迁到被忽略的 `backend/data/law_indexes/` 数据目录。本地运行配置保持该目录；首次构建可使用 [安装说明](../setup.md#4-构建并连接全量法条索引) 的 `make build-law-index`，不需要重复迁移。

## 评测与证据

以下保存 2026-09-30～2026-10-01 的实验配置与结果，不是本轮新跑结果。当前混合检索对照见 [检索历史证据](full_law_retrieval.md#2026-10-02-本地证据)；不要把下文早期纯向量结果解释为当前默认配置。

<details>
<summary>展开 2026-09-30 评测命令与历史结果</summary>

```bash
backend/.venv/bin/python evaluation/run_full_eval.py offline \
  --output evaluation/results/full-offline.json
backend/.venv/bin/python evaluation/run_full_eval.py live-rag \
  --query-mode semantic --output evaluation/results/full-rag.json
backend/.venv/bin/python evaluation/run_full_eval.py live-lawref \
  --case dangerous-driving-derived --case core-theft --output evaluation/results/full-lawref.json
backend/.venv/bin/python evaluation/verify_full_text.py \
  --corpus backend/data/law_knowledge/criminal_law_full.json \
  --pdf /path/to/official.pdf --output evaluation/results/full-text-check.json
```

2026-09-30的证据在 `evaluation/evidence/full_*_2026-09-30.json`：

- 提供的154页PDF经pypdf提取，移除空白和页码后505/505正文子串命中。源站本轮直接下载返回403，当前远端PDF字节hash没有确认；来源页面通过浏览核对。这只是程序文本证据。
- 离线真实关键词/读取工具8/8；正式LawRef适配、覆盖计算和编译图另有确定性回归，模型决策在测试中替代，不能称为真实LLM成功。
- 实际 `qwen3-embedding:0.6b` / 504条Chroma索引：初始纯向量条号测试2/8的失败保留；增加按条号精确路径后8/8；8个自然语言向量案例8/8。案例集合小且固定，不代表全量召回准确率或法律适用正确率。
- 实际 `qwen3.5:0.8b` LawRef两例0/2，均为 `duplicate_call`；安全终止、零覆盖。后续2026-10-01的消息链修复及残余限制见下文，历史文件保留。
- 记录了本机组件耗时，包括全量一致性核验和预热；不是并发吞吐、生产延迟或端到端咨询质量评测。

六条原live资产及input-only基线不与上述测试混合计分。2026-09-30记录为历史证据，2026-10-01起的评测另记录`annotation_mode=demo`。当前demo不以全量法律人工复核为门槛；真实模型稳定完成、Docker运行和生产部署仍只按实际执行结果报告。

2026-10-01放行验证：后台全量测试1377项通过，两项Redis连接受沙箱限制；获得本机连接权限后，两项原失败测试均通过。相关24项定向测试通过，demo离线工具评测8/8。放行记录见 `evaluation/evidence/full_demo_release_2026-10-01.json`。

2026-10-01主线程独立验收：从暂存区导出公开提交内容，显式配置Ollama并允许本机Redis后，后端全套1380项通过、评测测试49项通过、离线检索8/8、变更Python文件Ruff及`backend/app` Pyright通过。

没有私人`.env`且未指定模型时，旧RAG样例因默认阿里云配置缺密钥失败，保留该环境边界。

全仓Pyright仍有237条诊断（HEAD基线247条），本次未增加诊断；不能称全仓类型检查通过。

该验证没有重跑真实LLM LawRef，也不把会捕获检索异常的旧样例视为真实检索成功。

记录见 `evaluation/evidence/full_main_review_2026-10-01.json`。


</details>

<details>
<summary>展开默认切换、工具修复、标注放行与索引迁移历史</summary>

## 历史：默认切换与工具消息修复（首轮独立验收未通过）

2026-10-01在基线 `22a5ca440a895051cae99dedc3c764862d0bfb7d` 上复现原两例，[新失败证据](../../evaluation/evidence/full_live_lawref_before_2026-10-01.json)仍为0/2。

原实现每轮重建事实加 observations 的 HumanMessage，没有助手调用与关联的工具返回。

仅替换成标准消息链、保持工具开放和预算的对照，[危险驾驶](../../evaluation/evidence/lawref_history_probe_dangerous-driving-derived_2026-10-01.json)与[盗窃](../../evaluation/evidence/lawref_history_probe_core-theft_2026-10-01.json)均成功。

这支持消息反馈方式是原两例重复检索的直接诱因，不代表所有重复调用都有同一原因。

[LangChain ToolMessage 协议](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage)使用 `tool_call_id` 关联助手调用与工具结果。

保留的修复为局部 AIMessage／ToolMessage 历史，网关新增可选 `message_history`，其他调用者仍可使用原参数。历史只保留在当前函数调用内，不写入 checkpoint 或审计摘要。重复指纹拦截、读取后才能选择、来源与要件名称校验、最多4轮（配置上限8）、工具90秒、局部240秒及网关30/15秒有限重试均保留；没有自动补造要件或放宽最终 JSON 验证。

最终[原两例复跑](../../evaluation/evidence/full_live_lawref_final_2026-10-01.json)为2/2：危险驾驶 `search_laws → get_article → search_elements → final_answer`，盗窃 `search_laws → get_article → final_answer`。

选择来自 `rag_verified`，分别携带 demo 与 project_regression 标注。

该评测的通过条件是选中预期条文并取得覆盖分母，条号输入不是完整案件事实；盗窃模型声称匹配要件不能作为事实支持正确性的证明。

自然语言法规式输入的[补充失败记录](../../evaluation/evidence/full_live_lawref_semantic_2026-10-01.json)为0/2，多候选下仍出现无效最终输出与重复读取。

[口语案件事实最终复跑](../../evaluation/evidence/full_live_lawref_natural_final_2026-10-01.json)同样0/2：均实际搜索、读取正文和要件，盗窃最终格式无效而耗尽轮数，醉驾最终生成触发模型超时；均未生成 applied_laws。

输入为[两个公开合成案例](../../evaluation/lawref_natural_cases.jsonl)，没有真实用户材料。

程序仍拒绝未读取条文、未知要件名称、未放行正文的覆盖要件，但不能证明模型选择的事实匹配本身正确。

关闭已读取后的工具、补充字段错误反馈、引入 final_answer schema 工具三组临时探针分别保留[阶段探针](../../evaluation/evidence/lawref_final_stage_probe_2026-10-01.json)、[反馈探针](../../evaluation/evidence/lawref_feedback_probe_2026-10-01.json)、[schema工具探针](../../evaluation/evidence/lawref_final_tool_probe_2026-10-01.json)。

它们未稳定通过两例，没有合入。

当前修复不能宣称自然语言 LawRef 稳定完成或整体 demo 已获主线程验收。

运行模型为本机 `qwen3.5:0.8b`，digest `f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`；embedding为 `qwen3-embedding:0.6b`，digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。

语料为 `2026-09-30.2.integration1`，demo标注为 `demo-reviewed-2026-10-01-v1`。

来源、要件、配置与输入hash记录在新评测文件中。

模型与索引仅运行于本机，没有Docker运行或生产部署证明。

从根目录重跑（输出请使用新文件名）：

```bash
backend/.venv/bin/python evaluation/run_full_eval.py live-lawref \
  --case dangerous-driving-derived --case core-theft --output evaluation/results/lawref-original-new.json
backend/.venv/bin/python evaluation/run_full_eval.py live-lawref \
  --query-mode semantic --cases evaluation/lawref_natural_cases.jsonl \
  --output evaluation/results/lawref-natural-new.json
```

最终后端全套1383项通过（345条既有警告）；评测50项通过；后端app、tests与本轮变更评测文件Ruff通过；backend/app Pyright无诊断。

首次沙箱全套中的两项Redis连接失败在允许本机连接后通过。

扩大Ruff到整个evaluation目录仍有三个基线诊断（`import_full_corpus.py` E402，`test_annotation_review.py`与`test_full_index.py` I001），没有修改无关文件。

全仓Pyright仍有237条诊断，不能称为通过。

原六条JSON与2026-09-30失败文件的SHA256保持不变。

主线程独立复跑口语盗窃仍失败，第二项当时未验收通过。

后续云模型固定六例对照与本地小模型复测见[响应协议诊断](lawref_response_protocol.md)：本地默认路径仍未通过，相关评测已记录并暂缓，后续计划对照更大本地模型；参数规模不是已证实的唯一原因，也不阻断当前开发。

全量默认切换保留；原生最终生成仅以 `LAW_AGENT_FINAL_PROTOCOL=native_candidate` 显式启用，应用默认 `legacy`。

## 历史：标注放行与索引迁移

应用默认使用 `LAW_KNOWLEDGE_PROFILE=full`，加载全量正文和基础校对后的demo标注，支持检索、LawRef和覆盖计算，不以律师确认作为演示门槛。未设置环境变量的新进程也使用全量，`backend/.env.example` 明示该默认。六条快照仅在显式 `snapshot` 配置下用于历史回归。

2026-10-01新增放行第一批32条独立罪名标注，连同原六条共38条可参与demo覆盖计算。43条建议中的第1、13、17、149条属于一般或转引规则，第199条已删去，这五条不生成独立罪名分母。其余条文仍可读取全文。覆盖率表示演示要件的事实匹配程度，不是法律准确率。

2026-09-30已形成[第一批43条代理复核报告](law_review/2026-09-30.batch1.md)，包括原标注、修正建议、官方来源、PDF页码与待确认事项。

[505条复核旁表](law_review/2026-09-30.review.json)保留当时43条已提建议、462条待代理复核及尚未放行的历史状态。

当前放行资产是 `backend/data/law_knowledge/criminal_law_demo_annotations.json`，加载时覆盖对应项目标注，不改写官方正文或六条快照。

程序按条号边界逐条比对所提供PDF，505条精确匹配；此证据没有认证远端PDF字节或完成法律审查。

代理手写意见保存在 `law_review/2026-09-30.batch1.notes.json`，绑定整个语料和各条正文、原标注的hash。可复现报告生成（输出须采用不存在的新路径）：

```bash
backend/.venv/bin/python -m evaluation.review_annotations \
  --corpus backend/data/law_knowledge/criminal_law_full.json \
  --notes docs/knowledge/law_review/2026-09-30.batch1.notes.json \
  --pdf /path/to/criminal_law.pdf \
  --output /path/to/new-review.json \
  --report /path/to/new-report.md
```

第一批优先核对补充规定（七）（八）涉及的条文、总则年龄规则、跨条适用、六条回归摘要以及法释〔2026〕6号第8条涉及的四条。复杂解释与完整法律认证不作为当前demo验收条件。基础校对后的标注已按用户要求用于演示。

当时的验证配置使用临时隔离索引目录（历史路径记录在证据中），从历史全量索引复制后核验，原索引及用户上传库未重建或覆写。

[默认入口证据](../../evaluation/evidence/full_default_runtime_2026-10-01.json)由 `main` 导入、实际配置加载及启动阶段的法条预检函数生成，记录505条、504条有效正文、38条可覆盖及公共索引查询。

执行时未检测到 uvicorn 或8000端口服务，没有启动或重启服务；这不证明运行中的 HTTP 接口已经切换。

正常 `main.lifespan` 使用 SQLite checkpointer，默认直接构造 Orchestrator 则使用进程内 MemorySaver；重启前必须核实所用入口与 checkpoint 路径。

2026-10-01清理临时目录时，运行索引及相邻manifest迁至 `backend/data/law_indexes/full-law-2026-09-30.2.integration1-qwen3`，本地配置同步更新。历史证据中的路径保留执行当时的位置；原始诊断材料归档在被Git忽略的私有目录，另有逐文件路径与hash清单。长期运行索引应放在数据目录；临时任务目录只保存草稿与中间输出。

</details>
