# FastAPI 接口与权限边界

本文以 `backend/main.py` 和当前路由实现为准。它说明可调用接口及已知边界，不代表生产部署、真实 LLM/RAG 质量或完整故障恢复能力。运行时 schema 应通过目标提交的 `/openapi.json` 重新取得。

## 基础信息

- 服务地址：`http://127.0.0.1:8000`
- OpenAPI：`GET /openapi.json`
- Swagger UI：`GET /docs`
- 健康检查：`GET /health`
- Bearer 认证：`Authorization: Bearer <access_token>`
- WebSocket：`/api/v1/sessions/{session_id}/ws?token=<access_token>`，仅允许会话所有者的 `client` access token

受角色保护的路由在 OpenAPI 中声明 `HTTPBearer`。路由和 operation 数量会随代码变化，本文不保存容易失效的固定计数。

## 核心接口

| 能力 | 方法与路径 | 权限 | 实际状态来源或落库 |
| --- | --- | --- | --- |
| 注册 | `POST /api/v1/auth/register` | 公开；只创建 `client` | SQLite `users` |
| 登录 | `POST /api/v1/auth/login` | 公开 | JWT；refresh token 写入 `users` |
| 刷新 token | `POST /api/v1/auth/refresh` | 公开，需有效 refresh token | SQLite 校验 refresh token |
| 当前用户 / 登出 | `GET /api/v1/auth/me`、`POST /api/v1/auth/logout` | 已认证 | SQLite；登出只撤销 refresh token |
| 创建会话 | `POST /api/v1/sessions` | 已认证 | SQLite 创建 `consultations`；LangGraph 启动状态 |
| 知情同意 | `POST /api/v1/sessions/{session_id}/confirm-consent` | 会话所有者 | LangGraph checkpoint；`consent_given` 同步 SQLite |
| 发送消息 | `POST /api/v1/sessions/{session_id}/message` | 会话所有者 | workflow；HTTP 用户/Agent 消息写入 SQLite |
| 查询实时状态 | `GET /api/v1/sessions/{session_id}/state` | 所有者、已分配律师、管理员 | LangGraph checkpoint；不从 Redis/内存猜测 pending node |
| 获取报告草案 | `GET /api/v1/sessions/{session_id}/report-draft` | 已分配律师、管理员 | 实时 workflow state，不从 SQLite 恢复 |
| workflow 律师审核 | `PUT /api/v1/sessions/{session_id}/review` | 已分配律师、管理员 | application command；checkpoint + SQLite 审计 |
| 活跃会话列表 / 关闭 | `GET /api/v1/sessions`、`POST /api/v1/sessions/{session_id}/close` | 按角色和资源过滤 | application command；checkpoint + SQLite `cancelled` 审计 |
| 历史记录 | `GET /api/v1/consultations/list`、`GET /api/v1/consultations/{id}`、`GET /api/v1/consultations/{id}/messages` | client 仅本人；lawyer 仅已分配；admin 全部 | SQLite |
| 分配律师 / 更新状态 | `POST /api/v1/consultations/assign`、`PUT /api/v1/consultations/{id}/status` | admin 分配；lawyer/admin 更新允许范围 | SQLite；分配同步到仍活跃的 workflow checkpoint/cache |
| 律师工作台 | `/api/v1/lawyer/sessions*`、`/api/v1/lawyer/alerts*` | lawyer/admin 角色门槛；资源仍按 DB 分配校验 | SQLite |
| 用户 / 律师管理 | `/api/v1/users*`、`/api/v1/lawyers*` | admin | SQLite |
| 知识库管理 | `/api/v1/knowledge*` | admin | Chroma、MD5 store 与文档处理链；依赖本地模型/解析组件 |

`session_id` 是 LangGraph 的工作流 ID；`consultation_id` 是 SQLite 主键，二者通过 `consultations.workflow_session_id` 关联。创建会话响应同时返回两者。律师工作台和历史接口中的路径 ID 实际使用 `consultation_id`，不能与 workflow `session_id` 混用。

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

以下命令从仓库根目录启动服务；启动阶段要求 SQLite 可写、Redis 可连接，并需要能加载当前配置：

```bash
cd backend
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

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

创建会话。`client_id` 是兼容字段，可以省略；所有者始终取自 token：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"user_type":"suspect"}'
```

```json
{
  "session_id": "<workflow_session_id>",
  "consultation_id": "<database_consultation_id>",
  "welcome_message": "<含免责声明的欢迎语>",
  "current_agent": "Receptionist",
  "created_at": "<ISO-8601 timestamp>"
}
```

确认知情同意并发送消息：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/confirm-consent" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"consent_given\":true,\"consent_timestamp\":\"2026-07-16T08:00:00Z\",\"consent_version\":\"v1\"}"

curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/message" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"content\":\"事情发生在昨天晚上，请继续询问。\",\"idempotency_key\":\"client-message-001\"}"
```

查询状态：

```bash
curl -s "http://127.0.0.1:8000/api/v1/sessions/${SESSION_ID}/state" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

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
| 422 | `VALIDATION_ERROR` | Pydantic/FastAPI 请求校验失败 |
| 429 | `RATE_LIMITED` | 请求超过限流窗口 |
| 500 | `INTERNAL_ERROR` | 未处理错误或路由显式内部错误 |
| 502 | `LLM_SERVICE_ERROR` | 上游 LLM 服务失败 |
| 504 | `LLM_TIMEOUT` | 上游 LLM 超时 |

高风险短路目前通过 HTTP 403 和谨慎提示返回；它不等同于外部律师工单或通知已发送。

## Redis 与 SQLAlchemy 边界

- 应用 lifespan 依次执行 `init_db()` 和 `init_redis()`。Redis 在启动时是硬依赖，连接失败会阻断应用启动。
- Redis 仍可作为可选缓存/观测依赖，但不会作为执行状态回退；API 查询和恢复只读取 LangGraph checkpoint。
- `persist_state()` 对 LangGraph 已写入的完整 state 只刷新进程内兼容投影；只有调用方明确给出新增投影字段时才最小更新 checkpoint，且不能据此推断 pending node。
- SQLite 持久化用户、咨询记录和 HTTP 消息，并作为 approve/reject/close 的业务审计源；定向测试使用内存 SQLite 验证相关落库契约。
- 完整 workflow state 不从 SQLite 恢复。`facts_structured`、`applied_laws`、`risk_assessment`、`service_plan`、`report_draft` 等虽在 `Consultation` 模型定义，但当前自然 workflow 尚未统一回写这些列。新增报告草案接口因此明确读取实时 state，而不是宣称数据库已完整持久化。
- HTTP 与 WebSocket 消息都接受 `idempotency_key`；WebSocket 也兼容把 `message_id` 作为该键。HTTP 的 workflow 推进、checkpoint history 与 `consultation_messages` 写入位于同一 service 命令边界；WebSocket 仍不写 `consultation_messages`。生命周期命令先推进 checkpoint，再提交 SQLite；SQLite 失败时保留真实执行位置并标记 `repair_required`，503 会要求使用相同操作和相同 key 重试修复，普通 resume 在修复前被阻断。
- 幂等键最长 128 字符，作用域为 `(session_id, command_type, idempotency_key)`。同 key 不同载荷返回 409。可安全重放结果只在当前进程缓存一小时且总量最多 2048 条；消息若已推进 workflow、但随后 SQLite 写入失败，同 key 会重放原错误而不再次 resume，也不会自动补写缺失消息。重启或多 worker 不共享，跨进程部署必须增加持久化幂等表和共享并发控制。
- 首次 approve/reject 仅接受处于 `human_review` 断点的工作流；其他执行位置返回 409，不推进工作流也不写 SQLite。该限制不阻止已标记 `repair_required` 的同 action 审计修复。
- SQLAlchemy async 连接需要 `greenlet`；依赖已列入 `pyproject.toml` 和 `requirements.txt`。

## OpenAPI 与运行时已知差异

- 部分历史路由使用 `success_response()` 返回 `{code,message,data}` wrapper，但装饰器的 `response_model` 仍描述裸 data 模型；运行时成功响应以上述 wrapper 为准。
- FastAPI 自动生成的 422 OpenAPI schema 仍可能显示默认 `detail` 结构；运行时已由全局处理器转换为统一 `error` envelope。
- ASGI 测试会 mock LLM、RAG、Redis 或路由数据库依赖；它证明路由、鉴权和错误契约，不证明外部模型、Chroma 或真实 Redis 在线。
- 当前没有 Alembic migration、默认生产级持久化 checkpointer、access-token 撤销列表或多实例状态一致性。
