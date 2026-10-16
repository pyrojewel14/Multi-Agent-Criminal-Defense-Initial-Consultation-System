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
- `src/pages/LawyerWorkspace.tsx`：已分配律师的 workflow 审核；律师队列按数据库分配记录关联 workflow，缺少可关联 workflow 的记录只读展示。
- `src/pages/AdminWorkspace.tsx`：管理员通过界面创建律师、浏览咨询记录、分配承办律师，并切换到管理员实时审核队列。
- `src/components/`：ID、状态、错误、结构化数据与空状态组件。

`session_id` 标识 LangGraph workflow，执行状态保存于独立 checkpoint SQLite，Redis 仅用于缓存/投影；`consultation_id` 是业务 SQLite 咨询记录主键。两者不能混用。公开注册只创建 client，律师/admin 账号与案件分配依赖后端受信任初始化或管理接口。

## 不使用手动 API 的双角色演示

1. 从仓库根目录运行 `(cd backend && .venv/bin/python scripts/create_local_admin.py --username local-admin)`，交互式初始化首个管理员（如数据库已存在管理员则跳过此步；公开注册仍只允许 client）。用管理员账号登录前端，在「律师账号」页创建一个 lawyer 账号。
2. 在普通窗口以 client 身份注册或登录，创建咨询并同意告知；可以开始描述合成案情。
3. 管理员在「案件分配」页点击刷新，选中该咨询，选择刚创建的律师并点击「确认分配」。无需手写 POST 请求。
4. 在独立的无痕窗口或不同浏览器以 lawyer 身份登录。当工作流进入 HumanReview 后点击「刷新审核队列」，选择草案并批准/退回。
5. client 页面会轮询实时状态，可查看审核结果。普通标签页共用 localStorage 登录信息，因此多角色不能使用同一浏览器配置文件中的普通标签页。

注意：管理员初始化属于受信任的部署准备，不在公开注册页面提供；全量索引、模型服务和真实 LawRef 能否生成草案仍需单独联调。已重启服务时，活跃会话列表目前仍有进程内边界，不能将历史 SQLite 记录等同于可恢复的 workflow 审核草案。
