# LawRef 原始响应与最终生成协议诊断

这篇记录 LawRef 最终响应失败的诊断、协议对照和复现入口。
读完可以区分检索失败、thinking-only、格式错误与事实支持错误。
配置索引读 [全量资产说明](full_criminal_law.md)，评估口径读 [evaluation.md](../evaluation.md)，使用范围见 [limitations.md](../limitations.md)。

<a id="当前结论与阅读路线"></a>

## 当前职责与历史结论

当前主流程默认 `legacy` 工具循环，`native_candidate` 是显式候选协议。程序继续校验来源、读取资格、要件名称、重复调用与有限预算；检索专用配置见 [full 检索](full_law_retrieval.md)，上下文预算见 [Memory](../memory/README.md)。本页负责协议诊断与失败记录。

2026-10-15 已完成 `qwen3.5:9b` 一轮固定六例评测，结果 0/6。2026-10-16 修正多工具拒绝后的反馈和读取工具说明，只复测其中两条条号案例，真实评分 2/2；其余四例未重跑，没有新的六例总分。2026-10-01 的其他模型结果仍保留当时输入、配置、失败和指标。不同配置不能合并评分，也不能据组件评分判断完整咨询质量。

- 本地 `qwen3.5:9b` 与 legacy 流程：2026-10-15 一轮 0/6；2026-10-16 协议修正后的条号子集 2/2，其余四例的上下文预算问题仍未解决。历史小模型两轮 0/6，配置不同。
- 云模型与修订提示词：固定六例两轮 6/6，正式入口否认案例 1/1；主线程已核对这些局部证据。
- 以上结果不验证完整咨询闭环；当时的 PII 误遮盖与本地最终生成是两个独立问题，当前入口行为见下节。
- 更大本地模型是待验证方向，参数规模不是已证实的唯一原因。

优先看 [本地新提示词复测](#本地小模型与新提示词复测2026-10-01) 和 [修复后云对照及正式入口](#修复后两轮与正式复现入口)。下文保留试验先后、失败历史与当时配置，不把旧结果当作当前复跑结果。

## 本地有限对照入口

[本地 runner](../../evaluation/run_local_lawref.py)固定使用[原六例](../../evaluation/lawref_cloud_cases.jsonl)的字节与 SHA-256、当前 legacy 系统提示和原 `run_full_eval.py` 评分器。gold 仅在响应后评分；当前已读条文提示中的事实对象与初始事实字符串均按同一公开案例核对。该入口限 `qwen3.5:4b`、`qwen3.5:9b`，不下载模型、不写 `.env`，默认每个候选只跑一轮；第二轮需要先协调共享推理资源。

先读 `/api/version`、`/api/tags`、`/api/ps`，确认本地权重和 tools 能力后再执行；所有调用显式 `trust_env=False`。服务不可达、模型未安装或云代理 manifest 单独记录，未发生推理时不产生 0/6 分数。实际请求保留 `think=false`、temperature=0、seed=0、8192 上下文与 1024 输出 token；四步、单例 240 秒、工具 90 秒、网关总计 30 秒／单次 15 秒／最多两次保持固定。每模型每轮最多 48 次 chat 请求、整轮 1500 秒，最多两轮；该上限不含 embedding 与只读元数据请求。

对照使用向量 top5、关闭 reranker 与 HyDE，以免额外生成或模型下载干扰预算；这不是当前混合检索默认配置的完整质量评测。调用前核对 embedding digest，复制既有公共索引和 manifest，使用输出目录内的独立 SQLite 路径。原始 content、thinking、tool_calls、实际 HTTP 状态和完成 usage 逐请求保存；驻留快照的 `size_vram` 仅表示采样时的模型驻留量，不是整机或进程内存峰值。新窗口、思考与输出配置使本轮与历史 0.8b/云/native_candidate 数字不能合并；不同候选只在本入口同配置下比较。

从项目根运行，服务地址使用明确可达的既有 Ollama 地址，输出必须使用空的新目录：

```bash
backend/.venv/bin/python evaluation/run_local_lawref.py \
  --base-url "$OLLAMA_BASE_URL" --models qwen3.5:9b --rounds 1 \
  --index-directory backend/data/law_indexes/full-law \
  --output-dir evaluation/results/local-lawref-9b-new
```

可将模型改为 `qwen3.5:4b` 并使用另一新输出目录；退出码 2 表示依赖阻碍或候选未全部执行，1 表示真实六例评分存在失败，0 才表示选定候选各轮六例均通过。`preflight.json` 保留连通与能力诊断，`comparison.json` 保留各轮分母，逐轮 summary/raw 互相独立。输出可能含服务元数据与本地路径，公开前需按[维护规范](../maintenance.md)检查；保留失败与部分流，不覆盖历史文件。

整轮中断仍保留候选、轮次、summary/raw 文件和中断状态，`llm_executed` 按实际收到的响应记录；`candidate_llm_executed` 单独表示收到过指定候选的响应。返回模型身份或请求仪器校验失配单独记录，不能将其他模型的响应计为候选 0/6。中断或失配轮的 `passed`、`total` 为 null，退出码仍为 2；已完成且有效的其他轮证据保留。未收到响应时身份核验结果为 null；中断导致评分器未交回完整工具轨迹时，`rag_executed` 为 null，原始片段不充当完整评分。

2026-10-13 的[受控连通证据](../../evaluation/evidence/lawref_local_preflight_2026-10-13.json)保留当时主机名无法解析、回环端口拒绝连接的结果；该次未推理，没有分数。2026-10-15 服务已可达，版本为 0.40.4，核对已安装的 `qwen3.5:9b` 和 `qwen3-embedding:0.6b` 后运行一轮，没有下载新聊天模型，也没有运行未安装的 4b。

本轮[公开结果摘要](../../evaluation/evidence/lawref_local_9b_2026-10-15.json)保留逐例轨迹、精确预算、案例与提示指纹以及模型 digest，仅替换本地索引路径。12 次实际 chat 响应均为 HTTP 200，候选身份与请求参数核验通过，真实 embedding、检索工具及模型均执行，评分为 **0/6**、退出码 1。两例显式条号输入先检索到一条结果，随后模型反复同时返回 `get_article` 和 `search_elements`，被单工具协议拒绝，最终 `max_steps`；另四例先检索到五条结果，下一次模型调用前触发 `context_budget_exceeded`，汇总终止原因为 `budget_exceeded`。没有通过修改 gold、放宽预算或替换评分器消除失败。

这表明当前问题已从服务不可达转为输出协议与检索结果上下文预算；不能据此认定模型规模是唯一原因。本轮使用独立索引副本、关闭 reranker 与 HyDE，不代表完整咨询或法律专业复核。新入口的确定性协议回归见 [test_local_lawref.py](../../evaluation/test_local_lawref.py)。

2026-10-16 的有限修正继续保留每轮单工具限制。拒绝多调用后，程序对每个调用 ID 返回未执行的关联错误，并给出单工具纠偏；缺失或重复 ID 在反馈中规范化，不执行被拒绝工具，也不将条文记为已读取。`get_article` 的说明明确它同时返回正文与构成要件。新增回归覆盖纠正后四轮内完成、未读取条文不能直接入选、异常 ID 和持续违规仍受轮数限制；相关后端 145 项通过。

修正后的[真实两例结果](../../evaluation/evidence/lawref_9b_tool_feedback_two_2026-10-16.json)为 2/2，六次 chat 均 HTTP 200、候选身份与载荷核验通过，轨迹均为搜索、读取、最终答案。仍保持 8192 窗口、7168 输入预算及 1024 输出上限；这是评测主动设置的值，不代表模型能力上限。两例评分完成后的临时汇总脚本发生属性访问错误，原进程退出 1；原始评分和响应均已保存，汇总从原产物离线补齐，没有重新推理或改分。其余四例、完整咨询和 Docker 索引均未因此重新验收。

## 案情脱敏边界

当前已停用案情 PII 掩码。`backend/app/security/sensitive_filter.py` 保留 `mask_pii(text)`、`sanitize_input(text)` 和 `_mask_name(text)` 的函数入口与签名，字符串原样返回；姓名、地址、手机号、身份证号和车牌规则均移除。`sanitize_input` 没有独立文本清理职责，保留的日志只记录长度；高风险检测仍由 `detect_high_risk` 独立执行。

LawRef、事实摄取、Memory 背景和增量摘要保留原调用链，配置的模型会收到未掩码事实。当前不能宣称具备案情 PII 脱敏保护。日志过滤、trace 元数据不记录正文的约束及 JWT／RBAC 权限隔离独立保留；输入进入模型与日志不得泄漏原文是两个边界。Memory 的纯占位符标量处理仅兼容历史脱敏数据，不再生成新占位符。

[入口回归](../../backend/tests/security/test_sensitive_filter.py)核对各类合成 PII、普通案情、否认、空字符串及空白原样透传、幂等和风险检测独立性。[实际输入边界回归](../../backend/tests/consultation/test_pii_boundaries.py)核对 LawRef 工具循环及结构化提取、事实摄取、Memory 背景和增量摘要实际发送给供应商的消息仍包含原始合成事实及 PII；响应使用确定性替身，不证明真实模型质量或完整咨询。2026-10-07 主线程独立复跑相关五文件定向组，211 项通过；范围限入口及确定性调用契约。

下文模型、提示词和脱敏输入描述保留对应历史日期的事实，不代表停用掩码后的模型重跑结果。

## 历史实验：响应原因与对照

以下正文按试验发生顺序保留；其中“当前”“本轮”均指对应历史实验。

<details>
<summary>展开 2026-10-01 协议诊断、对照、失败与提交前回归原始记录</summary>

## 已确认的原因

在相同 `qwen3.5:0.8b`、公开口语事实、全量公共索引、4轮预算和网关30/15秒有限重试下，原始流式响应表明：

- 盗窃最终调用正常结束，`done_reason=stop`，输出199个token；`message.content` 为空，答案及JSON出现在 `message.thinking`。这是最终输出通道的问题，不能归类为已经返回了非法JSON。
- 醉驾最终调用与重试持续输出thinking，分别在约15秒／剩余deadline处被取消；采集到2737／2686个thinking字符，无content。没有完成响应，不能将输出token数或done_reason伪填为零或stop。
- 默认ChatOllama的reasoning为None，Ollama SDK会省略该HTTP参数，沿用模型默认思考行为。当前LangChain适配器只在reasoning显式开启时收集thinking为额外字段，最终内容仍独立。网关没有把思考文本当作最终答案。

该历史实验的 Ollama 版本为 0.35.0；安装版适配器源码、实际请求体与原始响应均已核对。

[Ollama thinking协议](https://ollama.com/blog/thinking)区分thinking和content；[ChatOllama reasoning说明](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama/reasoning)说明None沿用模型默认，False关闭。

这里的结论来自本机采集，不依赖文档推断。

## 对照结果与事实支持边界

| 对照 | 结果 | 说明 |
| --- | --- | --- |
| 原标准工具消息链、默认思考 | 自然两例0/2 | 盗窃thinking-only，醉驾最终思考超时 |
| 只关闭LawRef模型思考，仍开放全部工具 | 自然两例0/2 | 思考耗时减少，但再次重复读取，重复拦截仍生效 |
| 已读取结果的精简输入、无工具、原生JSON schema、think=false、1024token | 自然两例2/2 | 最终输出进入content；仍需检验事实支持 |
| 原生候选的完整六例，两次顺序复跑 | 两轮均4/6 | 原条号两例、原口语两例通过；事实不足及否认行为两例失败 |
| 同一输入去掉要件字符串enum | 控制两例仍全匹配 | 不能把默认全匹配归因于enum约束本身 |
| 同一输入改为逐要件布尔支持判断 | 控制两例仍全部true | 更换字段表达没有修复事实支持判断 |
| 同一精简输入和原生schema，显式think=true | 三例均无content，输出1024token后length | 在保留15秒和1024token预算内，未形成可用最终响应 |

原生候选能保证结构、条号与要件名称范围，却未可靠区分事实与提问／否认。不能把它的正例成功当成法律匹配正确性；仅提高步骤上限也不解决这个已观察到的问题。三个额外对照仅作诊断，没有合入业务判断。

明确的控制输入保存在[guardrail案例](../../evaluation/lawref_guardrail_cases.jsonl)。gold要求不进入模型或检索输入；runner在模型返回后检查匹配必须为空、两项盗窃要件必须留缺失。实际模型仍声称两项全匹配，所以两轮均被评测标为失败。原口语失败输入[保持不变](../../evaluation/lawref_natural_cases.jsonl)。

## 保留的实现与默认边界

应用默认仍为legacy工具循环；全量正文默认仍为full。`LAW_AGENT_FINAL_PROTOCOL=native_candidate` 是显式审阅候选，不写入本机 `.env`，也不作为 `.env.example` 默认启用。

候选在读取后只将脱敏事实与已读取条文送入最终生成，用动态JSON schema限制条号、要件名称及未放行正文的空要件。

Ollama只对该次模型副本关闭思考、限制1024token并绑定原生format，工厂缓存和其他节点不改。

其他供应商使用其structured output接口并保留raw usage；此适配只经过替代供应商回归，没有真实远端证明。

[Ollama原生结构化输出](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx)约束格式，不能证明事实推理正确。

来源、先读取后选择、要件名称验证仍由程序复核。

重复调用指纹、4轮默认／8轮配置上限、工具与整体deadline、网关有限重试仍保留。

原生候选最多一次纠偏，携带前次响应和具体错误；legacy仍受总轮数预算约束。

轨迹用 `rejection_reason` 区分empty_output、output_truncated、invalid_json、schema_error、unknown_article、unread_article、unknown_element及unexpected_tool_call，不保存案件原文或thinking。

[默认路径的最后实跑](../../evaluation/evidence/lawref_default_after_rework_2026-10-01.json)仍在口语盗窃最终空内容处安全终止，`applied_laws` 为空；这不是第二项已经解决的证据。

## 可复现证据

公开证据中的本地环境名称与个人绝对目录已脱敏；案例输入、模型响应、评分和运行统计未改动。脱敏副本的整文件字节可能不同于原始运行产物，历史整文件哈希不可直接用公开副本重算核对。复现时应按 [安装说明](../setup.md) 准备项目环境与独立索引。

2026-10-01 临时目录清理后，原始响应、诊断脚本与日志按原始字节私有归档，并保留逐文件路径与 hash 清单。公开评测证据未改写；旧路径只记录历史位置，不是当前依赖。当前索引说明见 [全量刑法索引说明](full_criminal_law.md)，新输出写入被忽略的 `evaluation/results/`。

- [显式候选六例第一轮](../../evaluation/evidence/lawref_candidate_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_candidate_six_run2_2026-10-01.json)：每例含选择、来源、匹配／缺失、轨迹、token及配置。两轮均4/6，不能合并宣称整体通过。
- [响应诊断摘要](../../evaluation/evidence/lawref_response_diagnosis_2026-10-01.json)：原始响应字符量、token、结束原因、请求参数与诊断路径。
- 完整原始content／tool_calls／thinking及不含请求头的HTTP请求体只用于公开合成案例，现已私有归档（原文件名为 `rework-*-raw.json`）。诊断脚本 `capture_raw.py` 同时归档，附带脚本hash；没有采集密钥或真实用户资料。

模型保持 `qwen3.5:0.8b`，digest为 `f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`。

embedding保持 `qwen3-embedding:0.6b`，digest为 `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。

语料及demo标注仍为 `2026-09-30.2.integration1`、`demo-reviewed-2026-10-01-v1`，索引仍使用隔离副本，没有下载／切换模型。

从根目录分别复跑（输出请选新路径）：

```bash
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --case dangerous-driving-derived --case core-theft \
  --output evaluation/results/protocol-original-new.json
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --query-mode semantic \
  --cases evaluation/lawref_natural_cases.jsonl --output evaluation/results/protocol-natural-new.json
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --query-mode semantic \
  --cases evaluation/lawref_guardrail_cases.jsonl --output evaluation/results/protocol-guardrail-new.json
```

## 先前选择记录（后续云对照见下文）

本轮回归为backend 1396 passed（345条warning）、evaluation 51 passed；app Pyright为0 errors，全量Pyright与基线均为237条，按文件、级别、规则和消息逐条比较无增减。

本任务修改的Python文件Ruff通过；扩大到全部app/tests时，其他会话新增的workflow.py导入区变更触发I001，已保留并排除在本任务补丁外。

扩大到全部evaluation还存在3处原有问题。

测试通过不能抵消上述真实控制案例失败。

1. 为LawRef最终事实支持判断授权一个更强的明确模型，保持当前检索、来源验证、deadline与控制案例，再做两轮真实复跑。模型名称、运行位置与成本需要明确授权；本轮没有擅自换模型或下载。
2. 若必须保留当前0.8b，可把demo范围收窄为检索、正文阅读与法条候选，暂不依赖该模型的要件全匹配结论。这需要用户接受覆盖能力的缩减，不能自行把它当成原验收要求已满足。

当前默认继续legacy，原生候选保持隔离。第二项整体仍WIP，待主线程独立审阅证据及选择方向；没有commit、push、合并、归档或Done变更。

## 授权云模型对照：入口被退役拒绝

2026-10-01用户授权仅用原公开合成案例试本机已有的 `deepseek-v3.1:671b-cloud`。

本机Ollama的tags仍包含405字节云代理manifest，remote_model为 `deepseek-v3.1:671b`，remote_host为 `https://ollama.com:443`，manifest digest为 `d3749919e45f955731da7a7e76849e20f7ed310725d3b8b52822e811f55d0a90`。

这是本地代理manifest标识，不能当作已运行云端权重的digest。

实际show与chat请求返回服务退役错误：`deepseek-v3.1:671b` 已于2026-07-15 00:00:00 -0700 PDT退役；chat为HTTP 410。在隔离进程中用原六个输入顺序复跑两轮，12次实际工具决策请求全部410，每轮0/6，均在搜索／读取前以dependency_failure终止。没有推理响应、最终答案或要件匹配／缺失判断可供验收，也不能据此证明账户已登录或云模型能力。没有静默回落本地模型。

本轮使用legacy工具消息协议，保留4轮、30/15秒网关预算与原评分契约，410视为不可重试的服务错误。

实际请求仅有temperature=0，没有发送本地候选的format、think或num_predict参数。

[官方结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)明确当前云端不支持原生结构化输出；[thinking文档](https://docs.ollama.com/capabilities/thinking)要求按模型能力核对控制值。

由于show及首轮工具决策已被拒绝，本轮没有实现或验证云端最终输出适配。

- [云入口诊断及请求摘要](../../evaluation/evidence/lawref_cloud_diagnosis_2026-10-01.json)
- [原六例第一轮](../../evaluation/evidence/lawref_cloud_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_six_run2_2026-10-01.json)

runner原模式标签仅表示运行意图。交付证据已补充actual_execution，并将llm_executed、rag_executed标为false；未补充前的runner文件、实际HTTP body/status/error及脚本已私有归档（原文件名为 `cloud-*-2026-10-01.*`），诊断附hash。没有采集请求头、账户身份或密钥，gold仍只在模型返回后评分。

V3.1阶段没有业务代码／永久配置变更，没有下载模型或启动／重启正式服务。全量默认和此前本地失败记录保留。

### 当前目录入口与云模型实测

主线程随后明确云试验授权涵盖当前可用入口。

[官方目录](https://ollama.com/library/deepseek-v4.1-flash)列出deepseek-v4.1-flash:cloud；本机show返回thinking控制值false、low、high、max，以及tools能力。

最小公开合成连通chat用已核实的think=false请求，实际返回HTTP 402，说明该模型不包含在当前free usage，需要添加usage credits或升级。

没有充值、升级、账户修改或下载；目录能力不等于账户调用权限。

追加候选gpt-oss:120b-cloud的show支持low、medium、high，默认medium；最小公开chat使用low返回HTTP 200、实际响应model为gpt-oss:120b、输出OK。

因此没有再探测20b。

[Ollama官方目录](https://ollama.com/library/gpt-oss)列出云tag。

实际请求、返回和usage均已采集，没有可核验的远端权重digest，不能用V3.1代理manifest替代该证明。

用原六例、原评分、legacy工具链和4轮／30-15秒预算，在隔离进程顺序跑两轮，均3/6。每轮19次请求均HTTP 200，返回model均gpt-oss:120b，已接受条文来源均rag_verified。两轮醉驾都把完整demo要件名缩为“醉酒驾驶机动车”，程序以unknown_element拒绝两次final；事实不足与否认案例仍错误地全匹配两个盗窃要件、missing为空。这说明模型切换本身未解决已观察到的提示／契约缺口。

- [GPT-OSS原协议第一轮](../../evaluation/evidence/lawref_cloud_gptoss120_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_gptoss120_six_run2_2026-10-01.json)

### legacy提示修复、审批与仪器纠正

只修改legacy系统提示：匹配表示本案明确肯定事实支持；条号／罪名咨询、正文／要件不是案件证据；未提供、否认或不确定留缺失；匹配时原样复制get_article或search_elements返回的required_elements完整name；空数组示例明确不代表案件事实。

没有关键词规则，没有改gold、验证器、预算或应用默认模型。

149项定向契约回归通过，owned Ruff及app Pyright通过，离线回归不证明事实判断已经修复。

额外精简schema文本协议和修复后legacy复测曾被自动审批拒绝，未执行；拒绝历史保留。主线程随后取得并核验人类对具体目的地、系统提示、工具协议、已读正文及要件、工具历史／纠错的外发授权，并通过同一正常审批机制执行。没有改传输入口或绕过审批。

第一次facts_v1记录中前三例有实际云HTTP 200，后三例在临时仪器的原query字面断言处失败，未发送HTTP。不能把该3/6解释为提示修复后模型结果。原因是应用先mask_pii，而仪器仍拿脱敏文本和原query比较；主线程只修仪器为原输入经现有mask_pii得到的完整facts精确allowset，保留第一次记录并用facts_v1_capture_v2新命名重跑，没有改业务脱敏。

过度脱敏是独立残余风险：地址regex实际吞掉“喝了酒，在城市道”，姓名regex将“任何案件／任何物品”中的“何案件／何物品”识别成姓名。原案例文件未变，但进入模型的事实已有这些变换；否认／未提供谓词仍保留，不能单凭脱敏解释全部误匹配。现有安全测试缺少普通刑事情境语义保留与跨标点吞行为的回归。此次只记录，不扩大PII业务修复。

### 修复后两轮与正式复现入口

主线程顺序执行的facts_v1_capture_v2两轮均6/6，子线程独立核对完整返回：条号咨询两例matches为空，盗窃正例2匹配／0缺失，醉驾正例以完整分支名称1匹配／0缺失，事实不足及否认两例0匹配／2缺失；来源均rag_verified，终止均final_answer。

醉驾两轮第3步仍曾输出缩写，被unknown_element拒绝，第4步在现有纠偏与轮数上限内改用完整名称成功。

第一轮首请求502由既有重试恢复（20次请求），第二轮19次请求均HTTP 200，返回model为gpt-oss:120b。

保留了原提示两轮3/6、首次仪器错误、审批拒绝、退役及付费拒绝历史。

- [修复后云第一轮](../../evaluation/evidence/lawref_cloud_gptoss120_facts_v1_capture_v2_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_gptoss120_facts_v1_capture_v2_six_run2_2026-10-01.json)
- [分阶段诊断与限界](../../evaluation/evidence/lawref_cloud_access_2026-10-01.json)

正式入口为[evaluation/run_cloud_lawref.py](../../evaluation/run_cloud_lawref.py)，输入[evaluation/lawref_cloud_cases.jsonl](../../evaluation/lawref_cloud_cases.jsonl)与原六例字节／hash一致。

CLI显式指定model/thinking/cases/summary/raw，先核实show能力；只在评测context内临时适配模型副本并恢复，不修改生产factory默认行为。

实际HTTP body/status、返回模型、usage与原始公开响应保存到raw（不采集请求头）；精确核对现有脱敏事实与公开工具快照，禁止模型回落、未经验证的原生format/num_predict和覆盖历史输出。

summary/raw同路径、空案例、空筛选均在外发前拒绝；rag_executed来自真实search_laws轨迹，不按mode推断。

适配器及评分13项定向回归、runner Pyright、owned Ruff均通过。

真实ChatOllama+HTTP替代传输验证low请求、工厂恢复、402/410、错误返回模型、脱敏allowset及上述拒绝路径；这些离线证据不替代两轮真实云结果。

主线程另通过正式CLI独立实跑[原否认案例](../../evaluation/evidence/lawref_cloud_formal_main_denial_2026-10-01.json)，1/1通过、3次HTTP 200、rag_verified、0匹配／2缺失。

从仓库根目录运行；先将终端变量 `LAW_INDEX_ABSOLUTE_PATH` 设为自己构建并核验的索引绝对路径，将 `LAW_INDEX_DIGEST` 设为对应 manifest 和实际 Ollama 模型的 digest。需要索引及其 manifest（历史索引现已迁入被忽略的数据目录），每轮换新的输出路径：

```bash
PYTHONNOUSERSITE=1 LLM_TYPE=OLLAMA EMBED_MODEL_TYPE=OLLAMA \
  LAW_KNOWLEDGE_PROFILE=full LAW_FULL_DEMO_ANNOTATIONS=on \
  LAW_FULL_INDEX_DIRECTORY="$LAW_INDEX_ABSOLUTE_PATH" \
  LAW_FULL_INDEX_COLLECTION=criminal_law_full \
  LAW_FULL_EMBEDDING_DIGEST="$LAW_INDEX_DIGEST" \
  TEXT_EMBEDDING_MODEL_NAME=qwen3-embedding:0.6b \
  LAW_AGENT_FINAL_PROTOCOL=legacy LAW_AGENT_MAX_STEPS=4 \
  LAW_AGENT_TOOL_TIMEOUT_SECONDS=90 LAW_AGENT_TIMEOUT_SECONDS=240 \
  LLM_TOTAL_TIMEOUT_SECONDS=30 LLM_ATTEMPT_TIMEOUT_SECONDS=15 \
  backend/.venv/bin/python evaluation/run_cloud_lawref.py \
  --cases evaluation/lawref_cloud_cases.jsonl --model gpt-oss:120b-cloud --thinking low \
  --base-url http://127.0.0.1:11434 \
  --output evaluation/results/cloud-formal-new-run1.json --raw-output evaluation/results/cloud-formal-new-run1-raw.json
```

回退方式是退出该评测进程；本机.env、正式聊天模型及其他节点配置没有永久改变。应用默认仍qwen3.5:0.8b，native候选仍显式隔离；该云复测阶段未重测新legacy提示在0.8b上的真实表现，不能宣称本地模型问题已解决。六例两轮证据不代表其他案件、并发、法律专家评审或生产部署验收。全量505／504／38保留，整体继续WIP，无commit／push／归档或Done。

## 本地小模型与新提示词复测（2026-10-01）

用户要求在本地小模型上复测新提示词。

主线程使用当前`_SYSTEM_PROMPT`、`qwen3.5:0.8b`、相同六例及全量公共索引，通过`evaluation/run_full_eval.py live-lawref --cases evaluation/lawref_cloud_cases.jsonl`顺序运行两轮。

协议明确为应用默认`legacy`，保持4轮、工具90秒、节点240秒与网关30/15秒预算，没有修改业务代码或永久模型配置。

本次未复跑`native_candidate`，不能与此前原生候选4/6混为同一配置。

| 案例 | 第一轮 | 第二轮 |
| --- | --- | --- |
| 条号咨询：危险驾驶 | agent_timeout | agent_timeout |
| 条号咨询：盗窃 | duplicate_call | duplicate_call |
| 口语盗窃事实 | agent_timeout | agent_timeout |
| 口语醉驾事实 | agent_timeout | agent_timeout |
| 事实不足 | max_steps | max_steps |
| 明确否认盗窃行为 | duplicate_call | duplicate_call |

两轮均0/6，评测退出码均为1。每例都成功检索并读取过条文，但未形成可验收的最终法律判断；日志还记录了无工具调用且content为空的响应。因此当前默认流程未通过，不能据此断言新提示词下否认事实仍被误匹配，也不能将问题归为单纯的模型规模不足。

- [本地第一轮](../../evaluation/evidence/lawref_local_new_prompt_legacy_six_live_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_local_new_prompt_legacy_six_live_run2_2026-10-01.json)
- [诊断与提示词快照](../../evaluation/evidence/lawref_local_new_prompt_diagnosis_2026-10-01.json)

首次受限环境运行全部ConnectError，未产生模型推理结果，保留为`lawref_local_new_prompt_legacy_six_run1_2026-10-01.json`，不计入上述两轮。

实际两轮日志已私有归档，原文件名分别为 `local-new-prompt-legacy-live-run1-2026-10-01.log` 与 `local-new-prompt-legacy-live-run2-2026-10-01.log`。

## 提交前回归（2026-10-01）

用户要求按内容分批提交整个工作树并同步main与开发分支；本地模型eval失败已在issue中记录，本次按用户要求留待后续处理。PII误遮盖风险仍保留，提交不代表该风险已修复或整体生产验收通过。

本次在当时的隔离环境重新运行后端全套：1396 passed、345条既有warning。首次受限运行的两项Redis测试因本机连接权限失败；允许本机连接后的完整重跑全部通过。evaluation代码测试61 passed；这些是runner及评分契约测试，不替代真实模型六例的结果。

backend/app与backend/tests的Ruff通过；本次变更的六个evaluation Python文件按backend规则并明确app导入归属后Ruff通过；backend/app Pyright为0 errors、0 warnings。

未宣称全仓Pyright通过。

逐条核对后将被多行格式化的30例cases.jsonl和六例lawref_cloud_cases.jsonl恢复为JSONL，输入与全部gold字段相同；六例文件恢复为公开复跑证据绑定的hash，并增加真实资产解析与hash回归。

历史六条快照仍保持原资产。

</details>
