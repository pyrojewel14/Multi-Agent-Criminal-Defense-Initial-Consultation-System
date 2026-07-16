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
- `src/pages/LawyerWorkspace.tsx`：assigned lawyer 数据库审核与 admin workflow 审核。
- `src/components/`：ID、状态、错误、结构化数据与空状态组件。

`session_id` 用于 LangGraph/Redis workflow；`consultation_id` 是 SQLite 咨询记录主键。两者不能混用。公开注册只创建 client，律师/admin 账号与案件分配依赖后端受信任初始化或管理接口。
