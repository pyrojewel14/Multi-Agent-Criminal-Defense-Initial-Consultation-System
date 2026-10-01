# LawRef 原始响应与最终生成协议诊断

状态：本地模型评测已记录并暂缓，不阻断当前开发。全量默认已验证，云对照两轮6/6及正式入口否认案例1/1已通过主线程独立验收；本地Qwen使用新提示词与默认legacy流程复测两轮均0/6。后续计划在相同案例与评分契约下试用更大本地模型；参数规模只是待验证的可能原因，PII误遮盖和本地最终生成仍需分别核实。本记录保留诊断历史和正式复现方式。

## 已确认的原因

在相同 `qwen3.5:0.8b`、公开口语事实、全量公共索引、4轮预算和网关30/15秒有限重试下，原始流式响应表明：

- 盗窃最终调用正常结束，`done_reason=stop`，输出199个token；`message.content` 为空，答案及JSON出现在 `message.thinking`。这是最终输出通道的问题，不能归类为已经返回了非法JSON。
- 醉驾最终调用与重试持续输出thinking，分别在约15秒／剩余deadline处被取消；采集到2737／2686个thinking字符，无content。没有完成响应，不能将输出token数或done_reason伪填为零或stop。
- 默认ChatOllama的reasoning为None，Ollama SDK会省略该HTTP参数，沿用模型默认思考行为。当前LangChain适配器只在reasoning显式开启时收集thinking为额外字段，最终内容仍独立。网关没有把思考文本当作最终答案。

本机Ollama为0.35.0；安装版适配器源码、实际请求体与原始响应均已核对。[Ollama thinking协议](https://ollama.com/blog/thinking)区分thinking和content；[ChatOllama reasoning说明](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama/reasoning)说明None沿用模型默认，False关闭。这里的结论来自本机采集，不依赖文档推断。

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

候选在读取后只将脱敏事实与已读取条文送入最终生成，用动态JSON schema限制条号、要件名称及未放行正文的空要件。Ollama只对该次模型副本关闭思考、限制1024token并绑定原生format，工厂缓存和其他节点不改。其他供应商使用其structured output接口并保留raw usage；此适配只经过替代供应商回归，没有真实远端证明。[Ollama原生结构化输出](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx)约束格式，不能证明事实推理正确。

来源、先读取后选择、要件名称验证仍由程序复核。重复调用指纹、4轮默认／8轮配置上限、工具与整体deadline、网关有限重试仍保留。原生候选最多一次纠偏，携带前次响应和具体错误；legacy仍受总轮数预算约束。轨迹用 `rejection_reason` 区分empty_output、output_truncated、invalid_json、schema_error、unknown_article、unread_article、unknown_element及unexpected_tool_call，不保存案件原文或thinking。

[默认路径的最后实跑](../../evaluation/evidence/lawref_default_after_rework_2026-10-01.json)仍在口语盗窃最终空内容处安全终止，`applied_laws` 为空；这不是第二项已经解决的证据。

## 可复现证据

公开证据中的本地环境名称与个人绝对目录已脱敏；案例输入、模型响应、评分和运行统计未改动。脱敏副本的整文件字节可能不同于原始运行产物，历史整文件哈希不可直接用公开副本重算核对。复现时应按 [安装说明](../setup.md) 准备项目环境与独立索引。

- [显式候选六例第一轮](../../evaluation/evidence/lawref_candidate_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_candidate_six_run2_2026-10-01.json)：每例含选择、来源、匹配／缺失、轨迹、token及配置。两轮均4/6，不能合并宣称整体通过。
- [响应诊断摘要](../../evaluation/evidence/lawref_response_diagnosis_2026-10-01.json)：原始响应字符量、token、结束原因、请求参数与诊断路径。
- 完整原始content／tool_calls／thinking及不含请求头的HTTP请求体只用于公开合成案例，保存在忽略的 `.scratch/lawref_default_fix/rework-*-raw.json`。诊断脚本为同目录 `capture_raw.py`，附带脚本hash；没有采集密钥或真实用户资料。

模型保持 `qwen3.5:0.8b`，digest为 `f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4`。embedding保持 `qwen3-embedding:0.6b`，digest为 `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。语料及demo标注仍为 `2026-09-30.2.integration1`、`demo-reviewed-2026-10-01-v1`，索引仍使用隔离副本，没有下载／切换模型。

从根目录分别复跑（输出请选新路径）：

```bash
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --case dangerous-driving-derived --case core-theft \
  --output .scratch/protocol-original-new.json
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --query-mode semantic \
  --cases evaluation/lawref_natural_cases.jsonl --output .scratch/protocol-natural-new.json
LAW_AGENT_FINAL_PROTOCOL=native_candidate backend/.venv/bin/python \
  evaluation/run_full_eval.py live-lawref --query-mode semantic \
  --cases evaluation/lawref_guardrail_cases.jsonl --output .scratch/protocol-guardrail-new.json
```

## 先前选择记录（后续云对照见下文）

本轮回归为backend 1396 passed（345条warning）、evaluation 51 passed；app Pyright为0 errors，全量Pyright与基线均为237条，按文件、级别、规则和消息逐条比较无增减。本任务修改的Python文件Ruff通过；扩大到全部app/tests时，其他会话新增的workflow.py导入区变更触发I001，已保留并排除在本任务补丁外。扩大到全部evaluation还存在3处原有问题。测试通过不能抵消上述真实控制案例失败。

1. 为LawRef最终事实支持判断授权一个更强的明确模型，保持当前检索、来源验证、deadline与控制案例，再做两轮真实复跑。模型名称、运行位置与成本需要明确授权；本轮没有擅自换模型或下载。
2. 若必须保留当前0.8b，可把demo范围收窄为检索、正文阅读与法条候选，暂不依赖该模型的要件全匹配结论。这需要用户接受覆盖能力的缩减，不能自行把它当成原验收要求已满足。

当前默认继续legacy，原生候选保持隔离。第二项整体仍WIP，待主线程独立审阅证据及选择方向；没有commit、push、合并、归档或Done变更。

## 授权云模型对照：入口被退役拒绝

2026-10-01用户授权仅用原公开合成案例试本机已有的 `deepseek-v3.1:671b-cloud`。本机Ollama的tags仍包含405字节云代理manifest，remote_model为 `deepseek-v3.1:671b`，remote_host为 `https://ollama.com:443`，manifest digest为 `d3749919e45f955731da7a7e76849e20f7ed310725d3b8b52822e811f55d0a90`。这是本地代理manifest标识，不能当作已运行云端权重的digest。

实际show与chat请求返回服务退役错误：`deepseek-v3.1:671b` 已于2026-07-15 00:00:00 -0700 PDT退役；chat为HTTP 410。在隔离进程中用原六个输入顺序复跑两轮，12次实际工具决策请求全部410，每轮0/6，均在搜索／读取前以dependency_failure终止。没有推理响应、最终答案或要件匹配／缺失判断可供验收，也不能据此证明账户已登录或云模型能力。没有静默回落本地模型。

本轮使用legacy工具消息协议，保留4轮、30/15秒网关预算与原评分契约，410视为不可重试的服务错误。实际请求仅有temperature=0，没有发送本地候选的format、think或num_predict参数。[官方结构化输出文档](https://docs.ollama.com/capabilities/structured-outputs)明确当前云端不支持原生结构化输出；[thinking文档](https://docs.ollama.com/capabilities/thinking)要求按模型能力核对控制值。由于show及首轮工具决策已被拒绝，本轮没有实现或验证云端最终输出适配。

- [云入口诊断及请求摘要](../../evaluation/evidence/lawref_cloud_diagnosis_2026-10-01.json)
- [原六例第一轮](../../evaluation/evidence/lawref_cloud_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_six_run2_2026-10-01.json)

runner原模式标签仅表示运行意图。交付证据已补充actual_execution，并将llm_executed、rag_executed标为false；未补充前的runner文件、实际HTTP body/status/error及脚本保存在忽略的 `.scratch/lawref_default_fix/cloud-*-2026-10-01.*`，诊断附hash。没有采集请求头、账户身份或密钥，gold仍只在模型返回后评分。

V3.1阶段没有业务代码／永久配置变更，没有下载模型或启动／重启正式服务。全量默认和此前本地失败记录保留。

### 当前目录入口与云模型实测

主线程随后明确云试验授权涵盖当前可用入口。[官方目录](https://ollama.com/library/deepseek-v4.1-flash)列出deepseek-v4.1-flash:cloud；本机show返回thinking控制值false、low、high、max，以及tools能力。最小公开合成连通chat用已核实的think=false请求，实际返回HTTP 402，说明该模型不包含在当前free usage，需要添加usage credits或升级。没有充值、升级、账户修改或下载；目录能力不等于账户调用权限。

追加候选gpt-oss:120b-cloud的show支持low、medium、high，默认medium；最小公开chat使用low返回HTTP 200、实际响应model为gpt-oss:120b、输出OK。因此没有再探测20b。[Ollama官方目录](https://ollama.com/library/gpt-oss)列出云tag。实际请求、返回和usage均已采集，没有可核验的远端权重digest，不能用V3.1代理manifest替代该证明。

用原六例、原评分、legacy工具链和4轮／30-15秒预算，在隔离进程顺序跑两轮，均3/6。每轮19次请求均HTTP 200，返回model均gpt-oss:120b，已接受条文来源均rag_verified。两轮醉驾都把完整demo要件名缩为“醉酒驾驶机动车”，程序以unknown_element拒绝两次final；事实不足与否认案例仍错误地全匹配两个盗窃要件、missing为空。这说明模型切换本身未解决已观察到的提示／契约缺口。

- [GPT-OSS原协议第一轮](../../evaluation/evidence/lawref_cloud_gptoss120_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_gptoss120_six_run2_2026-10-01.json)

### legacy提示修复、审批与仪器纠正

只修改legacy系统提示：匹配表示本案明确肯定事实支持；条号／罪名咨询、正文／要件不是案件证据；未提供、否认或不确定留缺失；匹配时原样复制get_article或search_elements返回的required_elements完整name；空数组示例明确不代表案件事实。没有关键词规则，没有改gold、验证器、预算或应用默认模型。149项定向契约回归通过，owned Ruff及app Pyright通过，离线回归不证明事实判断已经修复。

额外精简schema文本协议和修复后legacy复测曾被自动审批拒绝，未执行；拒绝历史保留。主线程随后取得并核验人类对具体目的地、系统提示、工具协议、已读正文及要件、工具历史／纠错的外发授权，并通过同一正常审批机制执行。没有改传输入口或绕过审批。

第一次facts_v1记录中前三例有实际云HTTP 200，后三例在临时仪器的原query字面断言处失败，未发送HTTP。不能把该3/6解释为提示修复后模型结果。原因是应用先mask_pii，而仪器仍拿脱敏文本和原query比较；主线程只修仪器为原输入经现有mask_pii得到的完整facts精确allowset，保留第一次记录并用facts_v1_capture_v2新命名重跑，没有改业务脱敏。

过度脱敏是独立残余风险：地址regex实际吞掉“喝了酒，在城市道”，姓名regex将“任何案件／任何物品”中的“何案件／何物品”识别成姓名。原案例文件未变，但进入模型的事实已有这些变换；否认／未提供谓词仍保留，不能单凭脱敏解释全部误匹配。现有安全测试缺少普通刑事情境语义保留与跨标点吞行为的回归。此次只记录，不扩大PII业务修复。

### 修复后两轮与正式复现入口

主线程顺序执行的facts_v1_capture_v2两轮均6/6，子线程独立核对完整返回：条号咨询两例matches为空，盗窃正例2匹配／0缺失，醉驾正例以完整分支名称1匹配／0缺失，事实不足及否认两例0匹配／2缺失；来源均rag_verified，终止均final_answer。醉驾两轮第3步仍曾输出缩写，被unknown_element拒绝，第4步在现有纠偏与轮数上限内改用完整名称成功。第一轮首请求502由既有重试恢复（20次请求），第二轮19次请求均HTTP 200，返回model为gpt-oss:120b。保留了原提示两轮3/6、首次仪器错误、审批拒绝、退役及付费拒绝历史。

- [修复后云第一轮](../../evaluation/evidence/lawref_cloud_gptoss120_facts_v1_capture_v2_six_run1_2026-10-01.json)、[第二轮](../../evaluation/evidence/lawref_cloud_gptoss120_facts_v1_capture_v2_six_run2_2026-10-01.json)
- [分阶段诊断与限界](../../evaluation/evidence/lawref_cloud_access_2026-10-01.json)

正式入口为[evaluation/run_cloud_lawref.py](../../evaluation/run_cloud_lawref.py)，输入[evaluation/lawref_cloud_cases.jsonl](../../evaluation/lawref_cloud_cases.jsonl)与原六例字节／hash一致。CLI显式指定model/thinking/cases/summary/raw，先核实show能力；只在评测context内临时适配模型副本并恢复，不修改生产factory默认行为。实际HTTP body/status、返回模型、usage与原始公开响应保存到raw（不采集请求头）；精确核对现有脱敏事实与公开工具快照，禁止模型回落、未经验证的原生format/num_predict和覆盖历史输出。summary/raw同路径、空案例、空筛选均在外发前拒绝；rag_executed来自真实search_laws轨迹，不按mode推断。

适配器及评分13项定向回归、runner Pyright、owned Ruff均通过。真实ChatOllama+HTTP替代传输验证low请求、工厂恢复、402/410、错误返回模型、脱敏allowset及上述拒绝路径；这些离线证据不替代两轮真实云结果。主线程另通过正式CLI独立实跑[原否认案例](../../evaluation/evidence/lawref_cloud_formal_main_denial_2026-10-01.json)，1/1通过、3次HTTP 200、rag_verified、0匹配／2缺失。

从仓库根目录运行；需要已验证的隔离索引副本及其manifest（本轮保留原位置），每轮换新的输出路径：

```bash
PYTHONNOUSERSITE=1 LLM_TYPE=OLLAMA EMBED_MODEL_TYPE=OLLAMA \
  LAW_KNOWLEDGE_PROFILE=full LAW_FULL_DEMO_ANNOTATIONS=on \
  LAW_FULL_INDEX_DIRECTORY="$PWD/.scratch/lawref_default_fix/full-qwen3" \
  LAW_FULL_INDEX_COLLECTION=criminal_law_full \
  LAW_FULL_EMBEDDING_DIGEST=ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d \
  TEXT_EMBEDDING_MODEL_NAME=qwen3-embedding:0.6b \
  LAW_AGENT_FINAL_PROTOCOL=legacy LAW_AGENT_MAX_STEPS=4 \
  LAW_AGENT_TOOL_TIMEOUT_SECONDS=90 LAW_AGENT_TIMEOUT_SECONDS=240 \
  LLM_TOTAL_TIMEOUT_SECONDS=30 LLM_ATTEMPT_TIMEOUT_SECONDS=15 \
  backend/.venv/bin/python evaluation/run_cloud_lawref.py \
  --cases evaluation/lawref_cloud_cases.jsonl --model gpt-oss:120b-cloud --thinking low \
  --base-url http://127.0.0.1:11434 \
  --output .scratch/cloud-formal-new-run1.json --raw-output .scratch/cloud-formal-new-run1-raw.json
```

回退方式是退出该评测进程；本机.env、正式聊天模型及其他节点配置没有永久改变。应用默认仍qwen3.5:0.8b，native候选仍显式隔离；该云复测阶段未重测新legacy提示在0.8b上的真实表现，不能宣称本地模型问题已解决。六例两轮证据不代表其他案件、并发、法律专家评审或生产部署验收。全量505／504／38保留，整体继续WIP，无commit／push／归档或Done。

## 本地小模型与新提示词复测（2026-10-01）

用户要求在本地小模型上复测新提示词。主线程使用当前`_SYSTEM_PROMPT`、`qwen3.5:0.8b`、相同六例及全量公共索引，通过`evaluation/run_full_eval.py live-lawref --cases evaluation/lawref_cloud_cases.jsonl`顺序运行两轮。协议明确为应用默认`legacy`，保持4轮、工具90秒、节点240秒与网关30/15秒预算，没有修改业务代码或永久模型配置。本次未复跑`native_candidate`，不能与此前原生候选4/6混为同一配置。

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

首次受限环境运行全部ConnectError，未产生模型推理结果，保留为`lawref_local_new_prompt_legacy_six_run1_2026-10-01.json`，不计入上述两轮。实际两轮日志分别保存在`.scratch/lawref_default_fix/local-new-prompt-legacy-live-run1-2026-10-01.log`与`local-new-prompt-legacy-live-run2-2026-10-01.log`。

## 提交前回归（2026-10-01）

用户要求按内容分批提交整个工作树并同步main与开发分支；本地模型eval失败已在issue中记录，本次按用户要求留待后续处理。PII误遮盖风险仍保留，提交不代表该风险已修复或整体生产验收通过。

本次在当时的隔离环境重新运行后端全套：1396 passed、345条既有warning。首次受限运行的两项Redis测试因本机连接权限失败；允许本机连接后的完整重跑全部通过。evaluation代码测试61 passed；这些是runner及评分契约测试，不替代真实模型六例的结果。

backend/app与backend/tests的Ruff通过；本次变更的六个evaluation Python文件按backend规则并明确app导入归属后Ruff通过；backend/app Pyright为0 errors、0 warnings。未宣称全仓Pyright通过。逐条核对后将被多行格式化的30例cases.jsonl和六例lawref_cloud_cases.jsonl恢复为JSONL，输入与全部gold字段相同；六例文件恢复为公开复跑证据绑定的hash，并增加真实资产解析与hash回归。历史六条快照仍保持原资产。
