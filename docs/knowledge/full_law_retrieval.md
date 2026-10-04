# 全量公共刑法检索

本入口召回供后续核对的候选正文。排名分数表示检索相关性，不能证明案件事实、构成要件或有罪。`text_only`、`rag_unverified` 与经过项目标注核验的资格继续分别处理；重排和 HyDE 不会提高条文的要件资格。

## 真实调用路径

`LegalToolRegistry._search_laws` → `search_laws_by_rag` → `search_full_index`。

缺少认证 `user_id` 时跳过 RAG。full 路径只打开 `LAW_FULL_INDEX_DIRECTORY` 中显式配置的公共 collection，不打开 `CHROMA_*` 用户上传库。每次请求仍读回核验全量正文、metadata、向量、manifest、语料版本/hash，以及 Ollama `/api/tags` 中实际 embedding 模型 digest。任一不匹配均拒绝 full RAG，工具层如实标记依赖失败；不会切换到上传库。

精确条号继续通过 `article_id` 读取正文，不进行扩展或重排。母条与增补条分别处理，废止条和不存在的编号返回空；包含多个条号时，现有接口仍只提取第一个编号。

自然语言默认执行：

1. 原查询向量召回 top20；不使用 HyDE 替换原查询。
2. 原查询中文 BM25 召回 top20。中文连续片段拆成单字与相邻双字，英文/数字按词；无需空格，没有案例词白名单。
3. 按规范条号去重，以等权 RRF `1/(60+rank)` 融合。分数不跨模型直接相加。
4. Registry 用公共 JSON 核验来源与要件资格，保留 rank、score、recall_ranks 和执行状态；关键词召回最多补入一个通道的 top20，重叠候选也获得该通道的 RRF 贡献。
5. 对合并候选统一执行一次本地重排，最终取 top5。混合模式中来源类别不再覆盖相关性排序。

直接调用 `search_full_index` 或 `search_laws_by_rag` 默认在其候选池重排后返回 top5。Registry 通过 `candidate_pool=True` 获取扩大候选，合并关键词后才截断；不会先对原来五条重排后丢掉其他召回结果。

BM25 缓存仅存已核验的公共正文，进程内至多两份。缓存键包含语料 hash、中文切分版本 `cjk-char-bigram-v1` 与完整正文 tuple，任何一项变化即失效。缓存不跳过 manifest、向量或实际模型核验。

## 本地评分协议

full 专用适配器读取已有 `RERANKER_MODEL_PATH`，使用 `local_files_only=True`，不会自动下载。只支持本次配置的 Qwen3-Reranker-0.6B；legacy 上传库的评分器没有改变。

采用 [Qwen 官方 Transformers 示例](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#using-transformers) 的完整 system/user/assistant/think 模板，以 `yes` 和 `no` 最后位置 logits 的二元 softmax 给出 0..1 分数。完整特殊 token IDs 直接进入模型，不经 `decode(skip_special_tokens=True)` 往返。本地实测 token IDs 为 yes=9693、no=2152；证据记录权重、配置、tokenizer 与模板 hash。

当前 `RERANKER_MAX_LENGTH=512` 时，完整指令与原查询优先保留，余下 token 预算用于条文正文。超长条文尾部可能未参与评分，但返回内容仍是完整核验正文；该评分不能证明全文语义都被模型读取。原查询超预算时抛出明确失败，回退融合顺序，不静默丢失尾部否认语句。

单执行器、单实际推理，小批次为 2。超时或外层取消后，工作线程会自然结束；结束前新请求返回 `busy` 并回退，不积累后台排队任务。重排失败、非法分数或超时均保留融合排名，记录 `failed` / `timeout` / `busy` 与 `degraded=true`。

full 重排器使用 FP32，设备由 `LAW_FULL_RERANK_DEVICE` 控制。`auto` 依次选择可用的 CUDA、MPS、CPU；也可显式指定其中一种。显式设备不可用或推理失败时，保留融合顺序并记录降级，不静默替换设备。MPS 指 [PyTorch 的 Metal GPU 后端](https://docs.pytorch.org/docs/stable/notes/mps.html)，不是 CUDA 的 FlashAttention 2 配置。设备变更后应重启服务；上传库的 legacy 评分器不读取此配置。

## 配置与 HyDE

配置示例见 `backend/.env.example`，索引无需迁移或重建。

| 配置 | 默认 | 边界 |
| --- | --- | --- |
| `LAW_FULL_RETRIEVAL_MODE` | `hybrid` | `vector` 保留原查询向量 top5；同时设置 `LAW_FULL_RERANK_ENABLED=false` 才保留历史 Registry 来源优先排序 |
| `LAW_FULL_RECALL_K` | 20 | 5..50，分别限制原查询向量、BM25 和 Registry 关键词通道 |
| `LAW_FULL_RERANK_ENABLED` | true | 失败回退融合顺序 |
| `LAW_FULL_RERANK_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps`，仅作用于 full 重排器 |
| `LAW_FULL_RERANK_TIMEOUT_SECONDS` | 30 | 有限范围 0.01..120 秒 |
| `LAW_FULL_HYDE_ENABLED` | false | 只在 hybrid 模式增加额外召回 |
| `LAW_FULL_HYDE_K` | 10 | 1..20 |
| `LAW_FULL_HYDE_CALL_BUDGET` | 1 | 每次检索 0 或 1 次生成；同时受已有会话调用/token 预算约束 |
| `LAW_FULL_HYDE_TIMEOUT_SECONDS` | 10 | 0.01..30 秒，包含生成、额外 embedding 和查询 |

HyDE 只调用当前配置的本地 Ollama 模型，temperature=0、seed=0、最多生成 192 token，不重试。生成文本仅为额外向量通道，原查询 BM25 与向量始终保留；最终重排仍使用原查询。它不会写入案件事实、候选正文或要件。生成、预算、HTTP 或额外 embedding 失败均只丢弃该通道，并记录实际降级状态。默认关闭，是否开启需要更多独立样本的质量与耗时证据。

每条结果和工具 observation 记录 `retrieval_status`，包括各阶段 `executed` / `disabled` / `skipped_exact` / `failed` / `timeout` 等状态、召回数量与降级原因类型。错误只记录类别，不输出本地路径或秘密。即便候选为空，列表附带的状态仍会传到 observation。

## 复跑消融

从仓库根目录，在项目 Python 环境和现有本地模型准备好后运行：

```bash
PYTHONNOUSERSITE=1 python evaluation/run_full_retrieval_ablation.py \
  --output evaluation/results/full_retrieval_ablation.json
```

四组为向量基线、混合、混合+重排、混合+重排+HyDE。均调用真实 Registry → RAG → 公共索引，顺序执行、每条 deadline 为 120 秒、每条结果即时写盘。`--limit` 可进行小样本探针；`--modes vector` 可独立复核历史向量基线。输出文件必须新建，避免覆盖失败证据。

`evaluation/full_retrieval_cases.jsonl` 包含 30 条：口语、相近法条、多行为、否认、事实不足及条号边界。gold 只在返回后评分，不传入检索；Recall@1/@5 按每例相关条文集合计算，再做宏平均，MRR@5 使用首个相关条文名次。空 gold 不进入 Recall/MRR 分母，边界空结果单独核对。否认案例只评价被指控主题的正文关联，不能作为定罪支持。

固定先跑向量再跑重排会产生冷启动差异，证据同时提供逐条耗时、均值与 p95。该集合是项目维护的检索回归样本，未经过独立法律标注，也没有覆盖完整法律质量、LawRef 最终推理或生产稳定性。

`hyde_attempt_count` 是尝试进入扩展阶段的次数。共享会话预算可在发 HTTP 请求之前拒绝该尝试，因此该字段不表示实际模型调用量；应结合 `hyde` 状态及会话调用/usage 记录解读。各通道的原始分数分别保存在 `recall_scores`，HyDE 距离不会覆盖原查询向量距离。

## 2026-10-02 本地证据

本节是历史实验结果，本轮文档核对没有重跑检索或测试；历史代码指纹与现在工作区可能不同。

最终 [四组完整实测](../../evaluation/evidence/full_retrieval_ablation_final_2026-10-02.json) 使用相同语料、embedding digest、reranker 权重与 30 条查询，命令退出码 0。每组 26 条有 gold 的样本计分，另 4 条无 gold；表中耗时包含 30 条全部调用。

| 模式 | Recall@1 | Recall@5 | MRR@5 | 平均耗时 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原向量 top5 + 历史工具来源排序 | 0.5000 | 0.8269 | 0.6487 | 0.369 s | 0.392 s |
| 混合 + 融合排序 | 0.3654 | 0.7692 | 0.5186 | 0.374 s | 0.462 s |
| 混合 + 重排 | 0.7500 | 0.9615 | 0.8590 | 8.865 s | 17.192 s |
| 混合 + 重排 + HyDE | 0.7500 | 1.0000 | 0.8782 | 11.428 s | 17.302 s |

每组 25 条自然语言调用、4 条精确编号调用和 1 条空查询。两组重排均实际执行 25 次，最后一组 HyDE 实际执行 25 次；四组均无模型失败或降级。编号母条/增补条分别命中，废止、不存在及空查询的空结果检查全部通过。

去掉条号边界后，24 条有 gold 的自然语言样本 Recall@5 分别为 0.8125、0.7500、0.9583、1.0000，详见 [交付核验](../../evaluation/evidence/full_retrieval_verification_2026-10-02.json)。混合本身降低了本组最终 top5 质量；重排仍漏掉 `phone_taken` 口语案例，额外 HyDE 在本组补回。该结果不足以证明一般化提升或全量法律召回质量，HyDE 仍默认关闭。

最终证据包含检索代码、查询文件、语料、实际模型 digest 和评分协议指纹，并由交付核验重新核对 120 条结果。契约/索引定向测试 73 项通过，知识库/工具分组回归 394 项通过，独立评测测试 11 项通过。完整后端测试 1417 项通过、2 项 Redis 并发测试因沙箱禁止本地连接而失败；未将该边界宣称为完整测试通过。

历史产物继续保留：[沙箱连接失败](../../evaluation/evidence/full_retrieval_sandbox_blocked_2026-10-02.json)、[四条早期探针](../../evaluation/evidence/full_retrieval_probe_2026-10-02.json)、[首轮 30 条实测](../../evaluation/evidence/full_retrieval_ablation_2026-10-02.json)。后两者的 vector 工具排序当时为向量结果后追加关键词，尚未恢复历史来源优先；首轮 `hyde_call_count` 表示尝试次数。最终对照使用带代码指纹且恢复旧基线的 final 产物，不混用上述历史指标。
