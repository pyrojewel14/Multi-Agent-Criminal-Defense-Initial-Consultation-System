# 前端 MVP

React + TypeScript + Vite 单页应用，面向刑事辩护初次咨询展示。开发环境通过 Vite 将 `/api` 和 `/health` 代理到 `http://127.0.0.1:8000`。

```bash
npm install
npm run dev
```

质量检查：

```bash
npm run typecheck
npm test -- --run
npm run build
```

主要目录：

- `src/api/`：FastAPI 类型、成功/错误 envelope、JWT refresh。
- `src/auth/`：登录和公开 client 注册。
- `src/pages/ClientWorkspace.tsx`：workflow session 咨询主线。
- `src/pages/LawyerWorkspace.tsx`：已分配律师与管理员的 workflow 审核；律师队列按数据库分配记录关联 workflow，缺少可关联 workflow 的记录只读展示。
- `src/components/`：ID、状态、错误、结构化数据与空状态组件。

`session_id` 标识 LangGraph workflow，执行状态保存于独立 checkpoint SQLite，Redis 仅用于缓存/投影；`consultation_id` 是业务 SQLite 咨询记录主键。两者不能混用。公开注册只创建 client，律师/admin 账号与案件分配依赖后端受信任初始化或管理接口。
