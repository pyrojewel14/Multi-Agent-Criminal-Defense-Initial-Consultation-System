# FastAPI 接口与权限边界

这篇带你调用注册、会话消息和律师审核接口，并解释权限与错误响应。
读完可以串起一次本地 HTTP 联调，区分 workflow ID 和数据库 ID。
先启动服务请读 [安装说明](setup.md)，能力范围见 [边界说明](limitations.md)。

## 联调顺序

注册普通用户 → 登录并保存 token → 创建会话 → 确认同意 → 发消息/查状态。律师审核还需要管理员分配和律师账号。下文响应字段按当前代码契约说明，不是本轮真实 HTTP 实跑记录。

## 基础信息

- 服务地址：`http://127.0.0.1:8000`
- OpenAPI：`GET /openapi.json`
- Swagger UI：`GET /docs`
- 健康检查：`GET /health`
- 就绪检查：`GET /ready`，HTTP 200 的 JSON 状态需单独判断
- Bearer 认证：`Authorization: Bearer <access_token>`
- WebSocket：`/api/v1/sessions/{session_id}/ws?token=<access_token>`，仅允许会话所有者的 `client` access token

受角色保护的路由在 OpenAPI 中声明 `HTTPBearer`。路由和 operation 数量会随代码变化，本文不保存容易失效的固定计数。

`/health` 只报告存活。`/ready` 保留既有字段，并报告原文数据库与 checkpoint 的真实初始化和只读连接探测状态，以及 `memory` 下的摘要配置、结构化字段记忆开关和上下文预算。必要存储未初始化、关闭、查询失败或超时时，`status` 与 `api` 为 `not_ready`；必要存储可用而可选重排序器不可用时，`status=degraded`、`api=ready`。HTTP 状态仍为 200，调用方应检查 JSON。checkpoint 的 `restart_recovery` 表示持久化配置能力，范围限单 worker 正常重启，`recovery_verified_now=false` 表示本请求未执行恢复验证。探测限时且不调用模型、读取会话或返回存储路径与原始异常；具体字段和日志边界见 [Memory](memory/README.md#就绪状态与有限日志)。

## 核心接口

| 能力 | 方法与路径 | 权限 | 实际状态来源或落库 |
| --- | --- | --- | --- |
| 注册 | `POST /api/v1/auth/register` | 公开；只创建 `client` | SQLite `users` |
| 登录 | `POST /api/v1/auth/login` | 公开 | JWT；refresh token 写入 `users` |
| 刷新 token | `POST /api/v1/auth/refresh` | 公开，需有效 refresh token | SQLite 校验 refresh token |
| 当前用户 / 登出 | `GET /api/v1/auth/me`、`POST /api/v1/auth/logout` | 已认证 | SQLite；登出只撤销 refresh token |
| 创建会话 | `POST /api/v1/sessions` | 已认证 | SQLite 创建 `consultations`；LangGraph 启动状态 |
| 知情同意 | `POST /api/v1/sessions/{session_id}/confirm-consent` | 会话所有者 | LangGraph checkpoint；`consent_given` 同步 SQLite |
| 发送消息 | `POST /api/v1/sessions/{session_id}/message` | 会话所有者 | checkpoint；HTTP/WS 外部原文与回复写入业务 SQLite |
| 查询实时状态 | `GET /api/v1/sessions/{session_id}/state` | 所有者、已分配律师、管理员 | LangGraph checkpoint；不从 Redis/内存猜测 pending node |
| 获取报告草案 | `GET /api/v1/sessions/{session_id}/report-draft` | 已分配律师、管理员 | LangGraph checkpoint state，不从业务 SQLite 的报告列恢复 |
| workflow 律师审核 | `PUT /api/v1/sessions/{session_id}/review` | 已分配律师、管理员 | application command；checkpoint + SQLite 审计 |
| 活跃会话列表 / 关闭 | `GET /api/v1/sessions`、`POST /api/v1/sessions/{session_id}/close` | 按角色和资源过滤 | application command；checkpoint + SQLite `cancelled` 审计 |
| 历史记录 | `GET /api/v1/consultations/list`、`GET /api/v1/consultations/{id}`、`GET /api/v1/consultations/{id}/messages` | client 仅本人；lawyer 仅已分配；admin 全部 | SQLite |
| 分配律师 / 更新状态 | `POST /api/v1/consultations/assign`、`PUT /api/v1/consultations/{id}/status` | admin 分配；lawyer/admin 更新允许范围 | SQLite；分配同步到仍活跃的 workflow checkpoint/cache |
| 律师工作台 | `/api/v1/lawyer/sessions*`、`/api/v1/lawyer/alerts*` | lawyer/admin 角色门槛；资源仍按 DB 分配校验 | SQLite |
| 用户 / 律师管理 | `/api/v1/users*`、`/api/v1/lawyers*` | admin | SQLite |
| 知识库管理 | `/api/v1/knowledge*` | admin | Chroma、MD5 store 与文档处理链；依赖本地模型/解析组件 |

`session_id` 是 LangGraph 的工作流 ID；`consultation_id` 是 SQLite 主键，二者通过 `consultations.workflow_session_id` 关联。创建会话响应同时返回两者；业务历史的 `ConsultationResponse` 也返回可空的 `workflow_session_id`，旧记录可能没有映射。律师工作台和历史接口中的路径 ID 实际使用 `consultation_id`，不能与 workflow `session_id` 混用。

## 权限矩阵

| 操作 | 匿名 | client | lawyer | admin |
| --- | --- | --- | --- | --- |
| 注册 / 登录 / 刷新 | 允许 | 允许 | 允许 | 允许 |
| 创建会话 | 401 | 允许 | 允许 | 允许 |
| 同意 / 发送消息 | 401 | 仅本人会话 | 403 | 403 |
| 实时状态 / 关闭 | 401 | 仅本人会话 | 仅已分配会话 | 允许 |
| 报告草案 / workflow 审核 | 401 | 403 | 仅已分配会话 | 允许 |
| 历史记录 | 401 | 仅本人记录 | 仅已分配记录 | 全部 |
| 用户、律师、知识库管理 | 401 | 403 | 403 | 允许 |
| WebSocket 消息 | 4401 | 仅本人会话 | 4403 | 4403 |

JWT 会校验签名、过期时间、token 类型、非空 `sub` 和 `client/lawyer/admin` 角色枚举。未知角色即使使用合法签名也按 401 拒绝。当前 access token 不做服务端撤销列表：登出、禁用用户或修改角色后，已签发 access token 在到期前不会被即时撤销。

## 可复制 HTTP 示例

先按 [setup.md](setup.md) 启动服务，再打开 `http://127.0.0.1:8000/docs`。
下面的 curl 用占位密码与 token 演示请求结构；请在本地替换，不要原样发送占位值。`ACCESS_TOKEN`、`SESSION_ID` 等变量需手动从响应中取得。

注册（真实 HTTP 状态为 201）：

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo_client","password":"<strong-password-entered-locally>","email":"demo@example.com"}'
```

成功响应结构：

```json
{
  "code": 201,
  "message": "注册成功",
  "data": {
    "id": "<user_id>",
    "username": "demo_client",
    "role": "client",
    "is_active": true
  }
}
```

登录并保存 access token：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo_client","password":"<strong-password-entered-locally>"}'
```

预期 HTTP 200，成功 wrapper 的 `data` 中包含 `access_token`、`refresh_token`、`token_type`、`expires_in` 和 `user`。将 `data.access_token` 保存为终端变量 `ACCESS_TOKEN`；不要把完整 token 写进文档。

创建会话。`client_id` 是兼容字段，可以省略；所有者始终取自 token：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"user_type":"suspect"}'
```

预期 HTTP 200，响应是裸会话对象；把 `session_id` 和 `consultation_id` 分别保存为 `SESSION_ID` 和 `CONSULTATION_ID`：

```json
{
  "session_id": "<workflow_session_id>",
  "consultation_id": "<database_consultation_id>",
  "welcome_message": "<含免责声明的欢迎语>",
  "current_agent": "Receptionist",
  "created_at": "<ISO-8601 timestamp>"
}
```

确认知情同意。先设置 `CONSENT_TIMESTAMP` 为本次同意的 ISO-8601 时间，不要复用历史时间：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/confirm-consent" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"consent_given\":true,\"consent_timestamp\":\"${CONSENT_TIMESTAMP}\",\"consent_version\":\"v1\"}"

```

预期 HTTP 200，响应含 `success=true`、`current_agent`、`next_prompt` 和 `conversation_started`。随后发送合成输入：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/message" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"content\":\"事情发生在昨天晚上，请继续询问。\",\"idempotency_key\":\"client-message-001\"}"
```

消息成功时返回 `message_id`、`agent_name`、`response_content`、`is_complete`、`pending_questions` 和 `alert_triggered`。值取决于实际流程；模型超时等情况按下方错误表处理。

查询状态：

```bash
curl -s "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/state" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

预期 HTTP 200，状态响应含 `session_id`、`consultation_id`、`current_agent`、`consent_given`、`pending_questions` 和 `status`。它未暴露全部底层状态字段；不能用业务角色推断 pending node。

管理员先用数据库 `consultation_id` 分配律师；之后已分配律师用 workflow `session_id` 读取草案并审核：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/consultations/assign \
  -H "Authorization: Bearer ${ADMIN_ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"consultation_id\":\"${CONSULTATION_ID}\",\"lawyer_id\":\"${LAWYER_ID}\"}"

curl -s "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/report-draft" \
  -H "Authorization: Bearer ${LAWYER_ACCESS_TOKEN}"

curl -s -X PUT "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/review" \
  -H "Authorization: Bearer ${LAWYER_ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approved","feedback":"同意该草案","final_output":"律师确认后的报告","idempotency_key":"lawyer-review-001"}'
```

分配成功返回 `{code,message,data}` wrapper；草案响应含 `report_draft`。首次审核必须在真实 `human_review` 断点：否则 409，跨存储提交失败为 503。具体重试方式见 [命令一致性](architecture.md#生命周期命令先推进图再提交审计)。

公开注册只创建 `client`。项目没有公开的管理员初始化接口；管理员 token 和律师账号必须来自已有受信任初始化数据或管理员接口，不能通过修改注册请求中的 `role` 获得。

## 统一错误响应

`AppException`、FastAPI/Starlette `HTTPException`、请求验证错误和未处理异常统一使用：

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "无权访问此会话"
  }
}
```

| HTTP 状态 | code | 含义 |
| --- | --- | --- |
| 400 | `BAD_REQUEST` | 业务请求不合法 |
| 401 | `UNAUTHORIZED` | 缺少、过期或声明无效的 access token |
| 403 | `FORBIDDEN` | 角色或资源权限不足 |
| 404 | `NOT_FOUND` | 路由、会话、记录或草案不存在 |
| 409 | `HTTP_ERROR` | 状态、幂等载荷或操作冲突 |
| 503 | `INTERNAL_ERROR` | 生命周期审计待修复；按响应要求重试相同操作 |
| 413 | `LLM_SERVICE_ERROR` | 必要完整输入超出上下文预算；拆分输入后补充 |
| 422 | `VALIDATION_ERROR` | Pydantic/FastAPI 请求校验失败 |
| 429 | `RATE_LIMITED` | 请求超过限流窗口 |
| 500 | `INTERNAL_ERROR` | 未处理错误或路由显式内部错误 |
| 502 | `LLM_SERVICE_ERROR` | 上游 LLM 服务失败 |
| 504 | `LLM_TIMEOUT` | 上游 LLM 超时 |

高风险短路目前通过 HTTP 403 和谨慎提示返回；它不等同于外部律师工单或通知已发送。

## 状态、消息与审计

- 实时状态、草案和恢复读取 LangGraph checkpoint；业务库不能替代执行状态。详见 [存储与恢复](architecture.md#存储与恢复)。
- `Consultation` 虽定义事实、法条和报告等列，自然 workflow 尚未统一回写这些列；草案接口因此读取实时 state。
- HTTP 与 WebSocket 共用 `process_message`；缺少数据库会话时由服务创建 `AsyncSessionLocal`，再进入 `process_external`。完整输入先提交业务消息表，随后执行图并记录实际回复；checkpoint 回执与跨存储修复统一见 [架构](architecture.md#并发与幂等)。
- WebSocket 可用 `message_id` 兼容提供幂等键。
- Redis 在 lifespan 中强制连接；`persist_state()` 的兼容投影不能用来推断待执行节点。
- SQLAlchemy 异步连接需要 `greenlet`，依赖已列入安装清单。

## OpenAPI 与运行时已知差异

- 部分历史路由使用 `success_response()` 返回 `{code,message,data}` wrapper，但装饰器的 `response_model` 仍描述裸 data 模型；运行时成功响应以上述 wrapper 为准。
- FastAPI 自动生成的 422 OpenAPI schema 仍可能显示默认 `detail` 结构；运行时已由全局处理器转换为统一 `error` envelope。
- ASGI 测试会 mock LLM、RAG、Redis 或路由数据库依赖；它证明路由、鉴权和错误契约，不证明外部模型、Chroma 或真实 Redis 在线。
- 当前没有 Alembic migration、checkpoint 备份恢复演练、access-token 撤销列表或多实例状态一致性；已有确定性证据覆盖 SQLite saver 跨进程恢复；[阶段 3 正常停启验证](memory/phase3-verification.md)已于 2026-10-04 由主线程独立验收，限固定合成样例、单 worker 正常停启，运行中崩溃和多 worker 尚未验证。已知 workflow ID 的恢复与旧会话发现见 [Memory 说明](memory/README.md#恢复与失败边界)。

## 术语速查

| 术语 | 含义 |
| --- | --- |
| JWT / Bearer | 签名访问令牌 / 在 Authorization 请求头携带令牌的方式 |
| RBAC | 按角色校验访问权限，资源归属仍需单独检查 |
| OpenAPI / ASGI | 接口描述标准 / Python 异步 Web 服务接口 |
| Pydantic | 字段、类型和约束校验库 |
| greenlet | SQLAlchemy 异步适配需要的协作执行组件 |
