# Multi-Agent Criminal Defense Initial Consultation System

基于多 Agent 协作的刑事辩护初期智能咨询系统，通过 LangGraph 编排六个专业 Agent，为刑事案件当事人提供结构化的法律咨询服务，并支持律师审核与人工介入。

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [使用说明](#使用说明)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目概述

本系统面向刑事案件当事人（嫌疑人、受害者、家属），提供智能化的初期法律咨询服务。系统通过六个专业 Agent 的协作，完成从接待、事实挖掘、法条检索、风险评估到服务方案生成的完整咨询流程，同时内置安全合规机制（PII 脱敏、高风险检测、免责声明注入），保障咨询过程的法律合规性。

**核心价值：**

- 结构化的事实挖掘，基于构成要件覆盖度驱动追问
- 两阶段法条检索（RAG 语义召回 + 知识库精确验证），避免 LLM 幻觉
- 量化风险评估与个性化服务方案
- 律师审核与高风险人工介入机制
- 多角色支持（咨询者、律师、管理员）

## 核心功能

### 六 Agent 协作工作流

| Agent | 职责 |
|-------|------|
| **Receptionist** 接待 Agent | 欢迎语生成、权利义务告知、用户身份确认、案件城市收集 |
| **FactDigger** 事实挖掘 Agent | LLM Function Calling 结构化事实提取、构成要件覆盖度分析、智能追问 |
| **LawRef** 法条检索 Agent | RAG 语义检索 + JSON 知识库精确匹配验证、法条编号归一化 |
| **RiskAssessor** 风险评估 Agent | 量刑预测、强制措施风险、证据风险点、程序风险综合评估 |
| **ServicePlanner** 服务方案 Agent | 个性化服务方案与《初期咨询报告》草案生成 |
| **HumanAlert** 高风险人工介入 Agent | 高风险检测触发时暂停自动流程，通知律师介入 |

### 安全合规

- **PII 脱敏** — 自动掩码身份证号、手机号、姓名、地址、车牌号
- **高风险语句检测** — 识别自认其罪、串供意图、伪造/销毁证据等敏感表述
- **免责声明注入** — 所有 Agent 输出自动附加法律免责前缀
- **RBAC 权限控制** — admin / lawyer / client 三级角色权限

### 多角色界面

- **咨询者端** — 在线咨询对话、同意确认、报告查看
- **律师端** — 会话审核、报告批准/退回、高风险告警处理、人工接管
- **管理员端** — 用户管理、律师管理、咨询监控、知识库维护

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3 + TDesign)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │ 咨询者端  │  │  律师端   │  │ 管理员端  │                         │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘                         │
│        └──────────────┼──────────────┘                              │
│                       │  HTTP / WebSocket                           │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────────┐
│                  Backend (FastAPI)                                   │
│                       │                                              │
│  ┌────────────────────┼────────────────────────────────────────┐    │
│  │              API Layer (v1/routers)                          │    │
│  │  auth │ sessions │ consultations │ lawyer │ knowledge │ ...  │    │
│  └────────────────────┬────────────────────────────────────────┘    │
│                       │                                              │
│  ┌────────────────────┼────────────────────────────────────────┐    │
│  │          LangGraph Orchestrator (workflow.py)                │    │
│  │                                                              │    │
│  │  START → Receptionist → FactDigger ⇄ LawRef → RiskAssessor │    │
│  │                    → ServicePlanner → HumanReview → END      │    │
│  │                    ↘ HumanAlert → END (高风险)               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                       │                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Security │  │   RAG    │  │   LLM    │  │   Core Utils     │    │
│  │ JWT/RBAC │  │ ChromaDB │  │ Gateway  │  │ Logger/RateLimit │    │
│  │ PII/免责  │  │ Reranker │  │ Factory  │  │                  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
│                       │                                              │
│  ┌────────────┐  ┌────┴─────┐                                      │
│  │  SQLite    │  │  Redis   │                                      │
│  └────────────┘  └──────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 技术栈

### 后端

| 类别 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI >=0.104.0 | 异步高性能 API 框架 |
| ASGI 服务器 | Uvicorn >=0.24.0 | 支持 WebSocket |
| 数据验证 | Pydantic >=2.5.0 | 请求/响应模型 |
| ORM | SQLAlchemy >=2.0.0 | 异步 ORM |
| 数据库 | SQLite (aiosqlite) | 零配置嵌入式数据库 |
| 缓存 | Redis >=5.0.0 | 会话缓存与限流 |
| 认证 | PyJWT + passlib[bcrypt] | JWT 令牌 + 密码哈希 |
| LLM 编排 | LangChain + LangGraph >=0.1.0 | 多 Agent 工作流 |
| 向量数据库 | ChromaDB >=0.4.22 | RAG 语义检索 |
| LLM 后端 | OpenAI SDK >=1.3.0 | 兼容阿里云百炼 / Ollama |
| 文档解析 | pypdf / python-docx / python-pptx | 知识库文档处理 |

### 前端

| 类别 | 技术 | 版本 |
|------|------|------|
| 框架 | Vue 3 | ^3.4.15 |
| 语言 | TypeScript | ~5.3.3 |
| 构建工具 | Vite | ^5.0.12 |
| UI 组件库 | TDesign Vue Next | ^1.9.8 |
| 状态管理 | Pinia | ^2.1.7 |
| 路由 | Vue Router | ^4.2.5 |
| HTTP 客户端 | Axios | ^1.7.2 |
| 图标 | Lucide Vue Next | ^0.511.0 |
| Markdown 渲染 | markdown-it | ^14.1.0 |
| 代码高亮 | highlight.js | ^11.9.0 |
| CSS 预处理 | Less | ^4.2.0 |

## 环境要求

- **Python** >= 3.10
- **Node.js** >= 18
- **Redis** >= 5.0
- **LLM 服务**（二选一）：
  - 阿里云百炼 API Key（推荐，用于 Qwen 系列模型）
  - Ollama 本地服务（需提前下载模型）

## 安装与配置

### 1. 克隆项目

```bash
git clone <repository-url>
cd "Multi-Agent Criminal Defense Initial Consultation System"
```

### 2. 后端配置

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，关键配置项如下：

```ini
# JWT 密钥（生产环境必须修改）
JWT_SECRET_KEY="CHANGE_ME_IN_PRODUCTION"

# LLM 类型：ALIYUN | OLLAMA
LLM_TYPE="ALIYUN"

# 阿里云百炼配置（LLM_TYPE=ALIYUN 时）
ALIYUN_ACCESS_KEY_SECRET="your_api_key"
ALIYUN_MODEL_NAME="qwen3-max"

# Ollama 配置（LLM_TYPE=OLLAMA 时）
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL_NAME="qwen3.5:0.8b"

# Embedding 模型类型：ALIYUN | OLLAMA
EMBED_MODEL_TYPE="ALIYUN"

# Redis 配置
REDIS_HOST="localhost"
REDIS_PORT=6379
```

### 3. 前端配置

```bash
cd frontend

# 安装依赖
npm install
```

### 4. 启动 Redis

确保 Redis 服务已启动：

```bash
# macOS
brew services start redis

# Linux
sudo systemctl start redis
```

## 使用说明

### 启动后端

```bash
cd backend
source venv/bin/activate
python main.py
```

后端服务启动在 `http://localhost:8000`，API 文档可访问 `http://localhost:8000/docs`。

### 启动前端

```bash
cd frontend
npm run dev
```

前端开发服务器启动在 `http://localhost:3000`，自动代理 API 请求到后端。

### 构建前端生产版本

```bash
cd frontend
npm run build
```

构建产物输出到 `frontend/dist/` 目录。

### 快速体验

1. 访问 `http://localhost:3000`，注册新账号
2. 以咨询者身份登录，创建咨询会话
3. 系统自动启动接待 Agent，引导完成知情同意
4. 进入事实挖掘阶段，Agent 将基于构成要件覆盖度智能追问
5. 完成事实收集后，系统自动进行法条检索、风险评估和服务方案生成
6. 生成报告后进入律师审核环节

## API 文档

所有 API 挂载在 `/api/v1` 前缀下。启动后端后可访问 Swagger 文档：`http://localhost:8000/docs`

### 认证模块 `/api/v1/auth`

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/auth/register` | 用户注册 | 公开 |
| POST | `/auth/login` | 用户登录 | 公开 |
| POST | `/auth/refresh` | 刷新令牌 | 公开 |
| GET | `/auth/me` | 获取当前用户信息 | 已认证 |
| POST | `/auth/logout` | 用户登出 | 已认证 |

### 咨询会话模块 `/api/v1/sessions`

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/sessions` | 创建新会话 | client |
| POST | `/sessions/{session_id}/message` | 发送消息 | client |
| POST | `/sessions/{session_id}/confirm-consent` | 确认隐私同意 | client |
| GET | `/sessions/{session_id}/state` | 获取会话状态 | client |
| PUT | `/sessions/{session_id}/review` | 律师审核反馈 | lawyer |
| GET | `/sessions` | 获取会话列表 | client |
| POST | `/sessions/{session_id}/close` | 关闭会话 | client |
| WebSocket | `/sessions/{session_id}/ws` | 实时双向通信 | client |

### 咨询历史模块 `/api/v1/consultations`

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| GET | `/consultations/list` | 获取咨询列表 | lawyer/admin |
| GET | `/consultations/{id}` | 获取咨询详情 | lawyer/admin |
| GET | `/consultations/{id}/messages` | 获取消息记录 | lawyer/admin |
| POST | `/consultations/assign` | 分配律师 | admin |
| PUT | `/consultations/{id}/status` | 更新咨询状态 | admin |

### 律师模块 `/api/v1/lawyer`

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| GET | `/lawyer/sessions` | 获取分配的会话 | lawyer |
| GET | `/lawyer/sessions/{id}` | 获取会话详情 | lawyer |
| PUT | `/lawyer/sessions/{id}/report` | 审核批准报告 | lawyer |
| POST | `/lawyer/sessions/{id}/reject` | 退回会话重做 | lawyer |
| POST | `/lawyer/sessions/{id}/intervene` | 人工接管会话 | lawyer |
| GET | `/lawyer/alerts` | 获取高风险告警 | lawyer |
| PUT | `/lawyer/alerts/{id}/read` | 标记告警已读 | lawyer |

### 知识库模块 `/api/v1/knowledge`

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/knowledge/add/single` | 上传单个文件 | admin |
| POST | `/knowledge/add/multiple` | 上传多个文件 | admin |
| DELETE | `/knowledge/clean` | 清空知识库 | admin |
| GET | `/knowledge/list` | 获取文档列表 | admin |
| GET | `/knowledge/chunks` | 获取文档切片信息 | admin |

### 用户/律师管理模块

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| GET | `/api/v1/users/` | 获取用户列表 | admin |
| POST | `/api/v1/lawyers/` | 创建律师账号 | admin |
| PUT | `/api/v1/lawyers/{id}` | 更新律师信息 | admin |

## 项目结构

```
├── backend/
│   ├── main.py                      # FastAPI 应用入口
│   ├── requirements.txt             # Python 依赖
│   ├── pyproject.toml               # 项目配置 & 构建系统
│   ├── .env.example                 # 环境变量模板
│   ├── docs/                        # 项目文档
│   └── app/
│       ├── agents/                  # 六个 Agent 节点
│       │   ├── receptionist.py      #   接待 Agent
│       │   ├── fact_digger.py       #   事实挖掘 Agent
│       │   ├── law_ref.py           #   法条检索 Agent
│       │   ├── risk_assessor.py     #   风险评估 Agent
│       │   ├── service_planner.py   #   服务方案 Agent
│       │   └── human_alert.py       #   高风险人工介入 Agent
│       ├── orchestrator/
│       │   └── workflow.py          # LangGraph 工作流编排器
│       ├── state/
│       │   └── consultation_state.py # 全局共享状态定义
│       ├── tools/
│       │   └── fact_tools.py        # LLM Function Calling 工具
│       ├── prompts/                 # Agent 提示词文件
│       ├── config/                  # YAML 配置文件
│       ├── v1/                      # API v1 层
│       │   ├── router/              #   路由定义
│       │   ├── schemas/             #   请求/响应模型
│       │   └── service/             #   业务逻辑服务
│       ├── rag/                     # RAG 检索增强生成
│       ├── security/                # 安全模块 (JWT/RBAC/PII/免责)
│       ├── models/                  # SQLAlchemy ORM 模型
│       ├── db/                      # 数据库 & Redis 配置
│       ├── errors/                  # 统一错误处理
│       ├── core/                    # 核心工具 (限流/响应封装)
│       ├── schemas/                 # 通用数据模型
│       └── utils/                   # 工具类 (LLM Gateway/Factory/Logger)
├── frontend/
│   ├── package.json                 # 前端依赖
│   ├── vite.config.ts               # Vite 配置
│   └── src/
│       ├── main.ts                  # Vue 应用入口
│       ├── App.vue                  # 根组件
│       ├── router/                  # 路由配置
│       ├── stores/                  # Pinia 状态管理
│       ├── api/                     # API 调用层
│       ├── views/                   # 页面视图
│       ├── components/              # UI 组件
│       ├── composables/             # Vue 组合式函数
│       ├── layouts/                 # 布局组件
│       ├── styles/                  # 样式
│       └── utils/                   # 工具函数
└── LICENSE                          # MIT 许可证
```

## 贡献指南

### 开发环境设置

```bash
# 后端开发依赖
cd backend
pip install -e ".[dev]"

# 代码检查
ruff check .

# 运行测试
pytest
```

### 代码规范

- **后端**：使用 Ruff 进行代码检查，行宽上限 120，目标 Python 3.10
- **前端**：使用 TypeScript 严格模式，Vue 3 Composition API 风格
- 提交信息请使用简洁准确的描述，说明变更内容和目的

### 开发流程

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/your-feature`)
3. 提交变更 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

### 关键设计约定

- Agent 节点位于 `backend/app/agents/`，每个 Agent 独立一个文件
- 全局状态通过 `ConsultationState` TypedDict 在 Agent 间传递
- 工作流编排集中在 `backend/app/orchestrator/workflow.py`
- LLM 调用统一通过 `LLMGateway`，法律场景强制 `temperature=0`
- 所有 Agent 输出自动经过 PII 脱敏和免责声明注入

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

Copyright (c) 2026 muding mao
