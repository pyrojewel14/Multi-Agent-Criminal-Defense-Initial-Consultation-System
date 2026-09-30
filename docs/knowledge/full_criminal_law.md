# 全量刑法正文、标注与独立索引

本实现提供显式启用的全量正文路径，默认仍为六条固定回归快照。`LAW_KNOWLEDGE_PROFILE=full` 现在默认加载基础校对后的demo标注，支持检索、LawRef和覆盖计算，不以律师确认作为放行门槛。

2026-10-01新增放行第一批32条独立罪名标注，连同原六条共38条可参与demo覆盖计算。43条建议中的第1、13、17、149条属于一般或转引规则，第199条已删去，这五条不生成独立罪名分母。其余条文仍可读取全文。覆盖率表示演示要件的事实匹配程度，不是法律准确率。

2026-09-30已形成[第一批43条代理复核报告](law_review/2026-09-30.batch1.md)，包括原标注、修正建议、官方来源、PDF页码与待确认事项。[505条复核旁表](law_review/2026-09-30.review.json)保留当时43条已提建议、462条待代理复核及尚未放行的历史状态。当前放行资产是 `backend/data/law_knowledge/criminal_law_demo_annotations.json`，加载时覆盖对应项目标注，不改写官方正文或六条快照。程序按条号边界逐条比对所提供PDF，505条精确匹配；此证据没有认证远端PDF字节或完成法律审查。

代理手写意见保存在 `law_review/2026-09-30.batch1.notes.json`，绑定整个语料和各条正文、原标注的hash。可复现报告生成（输出须采用不存在的新路径）：

```bash
PYTHONNOUSERSITE=1 conda run -n Agent_dev python -m evaluation.review_annotations \
  --corpus backend/data/law_knowledge/criminal_law_full.json \
  --notes docs/knowledge/law_review/2026-09-30.batch1.notes.json \
  --pdf /path/to/criminal_law.pdf \
  --output /path/to/new-review.json \
  --report /path/to/new-report.md
```

第一批优先核对补充规定（七）（八）涉及的条文、总则年龄规则、跨条适用、六条回归摘要以及法释〔2026〕6号第8条涉及的四条。复杂解释与完整法律认证不作为当前demo验收条件。基础校对后的标注已按用户要求用于演示。

## 资产与版本

- `backend/data/law_knowledge/criminal_law_chapters.json`：六条固定回归资产，保持原始字节；旧 `evaluation/build_live_index.py` / `run_live_eval.py` 的索引和评测契约保留。
- `backend/data/law_knowledge/criminal_law_full.json`：`criminal-law-full@2026-09-30.2.integration1`，16章、505条记录，504条有效正文；第199条保留删去记录，不进入搜索或索引。
- `backend/data/law_knowledge/criminal_law_full.review.json`：505条逐条复核队列，包含正文hash、待复核字段、列举项和跨条引用线索。6条标注沿用项目回归契约；其余499条标注保持待复核。任何条文的正文法律人工审查均未自动完成。

正文来源为[政府整合版刑法 PDF](https://fgw.sh.gov.cn/cmsres/cb/cba3dfde07c1437c9c55f9ec15f91e11/51e0f21bda99e0b7e33eaa22e4cc9f4e.pdf)，整合至修正案（十二）。第169条标题单独对照[两高罪名补充规定（八）](https://www.court.gov.cn/fabu/xiangqing/424452.html)，状态为 `source_checked`；这没有使该条罪名标签、要件或法定刑提议一起通过复核。

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

未放行条文的提议原值只保存在 `annotations.value`，其运行时 `elements=[]`、`base_sentence=""`、`charge_tags=[]`；占位不会靠非空验证取得可信状态。`text_structure`只按正文保留列举项和跨条引用原文，状态 `syntax_only`，没有把并列、替代、加重情形自动解释为完整构成要件。总则、程序和多罪名条文都保留完整正文；第133条之一的四项、第149条的跨条适用、第277条的多个情形有专门回归。已放行32条由独立demo资产提供要件与法定刑，原值保存在`original_annotations`。替代分支合成一个“任一适用分支”条件，避免要求所有替代项同时满足；其事实支持由已读取条文的LawRef判断，不套用旧关键词规则。

## LawRef 与覆盖计算

`LAW_KNOWLEDGE_PROFILE=snapshot` 为默认；`full`加载新正文和demo标注，`LAW_FULL_DEMO_ANNOTATIONS=off`可恢复仅六条参与覆盖的历史模式。demo资产默认随backend打包，`LAW_DEMO_ANNOTATIONS_PATH`可指定另一路径。`LAW_FULL_CORPUS_PATH`可选择明确版本文件。未知配置、缺失资产、重复/错误条号、来源或hash损坏都失败关闭。进程缓存按配置、绝对路径、mtime与大小区分；不要原地覆写运行中资产，更新后切换到新路径并重启服务。

关键词按条号、正文和带状态的标题召回。LegalToolRegistry的 `get_article` 可实际读取六条之外的全文、来源、标注状态和列举/引用线索；`search_elements`对未复核条文返回空要件。RAG命中同时连接语料hash、版本和完整正文，不一致时标为 `rag_unverified`，不继承可信要件，关键词JSON路径仍可降级运行。

模型只能选择已召回且已读取的条文。正文参考保存为 `law_text_candidates`，与 `applied_laws`分离。原六条与32条基础校对的demo标注可进入 `applied_laws`；demo标记和版本来源随检索传递。若选择尚未放行的正文参考，FactDigger将覆盖度置零、记录 `annotation_review_required`、直接进入HumanReview并等待律师决定，不向用户反复追问以填补标注缺口；人工重试会清理旧正文候选。确定性真实编译图已验证：危险驾驶demo候选进入覆盖计算后路由到RiskAssessor，跳过`annotation_review_required`拦截。模型响应在此测试中替代。其他尚未放行的正文参考仍保留原处理；不要求当前demo完成全505条法律审查。

## 构建、更新与回滚

所有项目Python命令均使用Agent_dev。以下从仓库根目录执行。

导入新的供应候选，输出到**不存在的新版本路径**，同时生成复核队列：

```bash
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/import_full_corpus.py \
  --candidate /path/to/candidate.json \
  --output backend/data/law_knowledge/criminal_law_full.NEW.json \
  --review-queue backend/data/law_knowledge/criminal_law_full.NEW.review.json
```

现有导入契约针对修正案（十二）的452个基础条号与53个衍生条号。法律版本发生变化时，需要同步审查来源、条号契约与复核队列，不能仅更换版本字符串。新路径也需明确纳入两层Git白名单和Docker白名单。供应候选的hash记录在资产metadata；重复导入相同输入会生成相同输出。本交付没有把供应方构建脚本或个人目录依赖加入运行时。

构建独立全量公共Chroma索引；目标目录和相邻manifest均不得存在：

```bash
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/build_full_index.py \
  --index-dir .scratch/full-law-2026-09-30.2.integration1-qwen3 \
  --collection criminal_law_full
```

构建入口从 `backend/.env`读取已有模型配置（环境变量优先），直连Ollama并读取实际embedding digest，每条有效正文一个文档，不包含提议标注、不截断正文。manifest记录语料版本和hash、文档ID与来源hash、模型/digest、维度和float32向量hash。构建后读回全部504条核验；失败目录没有可用manifest，应保留失败证据并另选新目录重建。

测试用合成向量标记 `deterministic-test`，运行时拒绝使用。运行时每次核验整份manifest、Chroma文档/来源/向量以及实际Ollama模型digest，再按条号精确读取或执行自然语言向量检索。这是优先保证可审查性的504条实现，尚未做大规模并发性能优化。

在独立验证进程显式启用：

```bash
export LAW_KNOWLEDGE_PROFILE=full
export LAW_FULL_INDEX_DIRECTORY="$PWD/.scratch/full-law-2026-09-30.2.integration1-qwen3"
export LAW_FULL_INDEX_COLLECTION=criminal_law_full
export LAW_FULL_EMBEDDING_DIGEST='<manifest和/api/tags中的实际digest>'
```

该索引不改变通用 `CHROMA_*` 用户上传索引，不调用其HyDE或reranker。没有 `user_id`时仍跳过RAG，JSON正文检索可以继续；索引未配置或不一致时记录RAG依赖失败，不自动改读旧六条索引。

更新应使用新语料文件、新目录和新manifest；先完整核验，再由主线程批准切换配置并重启。回滚是恢复上一套明确语料路径、索引目录、collection和digest后重启，或改回 `LAW_KNOWLEDGE_PROFILE=snapshot`。回滚配置不会回滚checkpoint、数据库或已生成报告；本任务没有更改任何正式服务配置。

## 评测与证据

```bash
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/run_full_eval.py offline \
  --output evaluation/results/full-offline.json
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/run_full_eval.py live-rag \
  --query-mode semantic --output evaluation/results/full-rag.json
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/run_full_eval.py live-lawref \
  --case dangerous-driving-derived --case core-theft --output evaluation/results/full-lawref.json
PYTHONNOUSERSITE=1 conda run -n Agent_dev python evaluation/verify_full_text.py \
  --corpus backend/data/law_knowledge/criminal_law_full.json \
  --pdf /path/to/official.pdf --output evaluation/results/full-text-check.json
```

2026-09-30的证据在 `evaluation/evidence/full_*_2026-09-30.json`：

- 提供的154页PDF经pypdf提取，移除空白和页码后505/505正文子串命中。源站本轮直接下载返回403，当前远端PDF字节hash没有确认；来源页面通过浏览核对。这只是程序文本证据。
- 离线真实关键词/读取工具8/8；正式LawRef适配、覆盖计算和编译图另有确定性回归，模型决策在测试中替代，不能称为真实LLM成功。
- 实际 `qwen3-embedding:0.6b` / 504条Chroma索引：初始纯向量条号测试2/8的失败保留；增加按条号精确路径后8/8；8个自然语言向量案例8/8。案例集合小且固定，不代表全量召回准确率或法律适用正确率。
- 实际 `qwen3.5:0.8b` LawRef两例0/2，均为 `duplicate_call`；安全终止、零覆盖，没有生成错误的可靠要件。这是默认启用前必须处理的真实模型限制。
- 记录了本机组件耗时，包括全量一致性核验和预热；不是并发吞吐、生产延迟或端到端咨询质量评测。

六条原live资产及input-only基线不与上述测试混合计分。2026-09-30记录为历史证据，2026-10-01起的评测另记录`annotation_mode=demo`。当前demo不以全量法律人工复核为门槛；真实模型稳定完成、Docker运行和生产部署仍只按实际执行结果报告。

2026-10-01放行验证：后台全量测试1377项通过，两项Redis连接受沙箱限制；获得本机连接权限后，两项原失败测试均通过。相关24项定向测试通过，demo离线工具评测8/8。放行记录见 `evaluation/evidence/full_demo_release_2026-10-01.json`。

2026-10-01主线程独立验收：从暂存区导出公开提交内容，显式配置Ollama并允许本机Redis后，后端全套1380项通过、评测测试49项通过、离线检索8/8、变更Python文件Ruff及`backend/app` Pyright通过。没有私人`.env`且未指定模型时，旧RAG样例因默认阿里云配置缺密钥失败，保留该环境边界。全仓Pyright仍有237条诊断（HEAD基线247条），本次未增加诊断；不能称全仓类型检查通过。该验证没有重跑真实LLM LawRef，也不把会捕获检索异常的旧样例视为真实检索成功。记录见 `evaluation/evidence/full_main_review_2026-10-01.json`。
