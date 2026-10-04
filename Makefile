PYTHON ?= .venv/bin/python
UV_PYTHON ?= 3.12
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 5173
LAW_INDEX_DIR ?= backend/data/law_indexes/full-law
LAW_COLLECTION ?= criminal_law_full

.PHONY: help install run-backend run-frontend build-law-index test test-backend test-frontend eval compose-config compose-up compose-down

help:
	@echo "可用目标："
	@echo "  make install       使用 uv 创建后端环境并安装前后端依赖"
	@echo "  make run-backend   启动 FastAPI（默认 127.0.0.1:8000）"
	@echo "  make run-frontend  启动 Vite（默认 127.0.0.1:5173）"
	@echo "  make build-law-index 构建全量公共法条索引（目标目录必须不存在）"
	@echo "  make test          运行定向后端回归与全部前端测试"
	@echo "  make eval          运行 30 条确定性离线评估"
	@echo "  make compose-config 校验 Docker Compose 配置"
	@echo "  make compose-up    启动 backend + Redis 容器"
	@echo "  make compose-down  停止 Compose 服务（保留命名卷）"

install:
	uv venv --python $(UV_PYTHON) --allow-existing backend/.venv
	uv pip install --python backend/.venv/bin/python -r backend/requirements.txt
	uv pip install --python backend/.venv/bin/python "pytest>=7.4.0" "pytest-asyncio>=0.21.0" "httpx>=0.25.0" "ruff>=0.1.0"
	npm --prefix frontend ci

run-backend:
	cd backend && $(PYTHON) -m uvicorn main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT)

run-frontend:
	npm --prefix frontend run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

build-law-index:
	cd backend && $(PYTHON) ../evaluation/build_full_index.py \
		--index-dir "$(if $(filter /%,$(word 1,$(LAW_INDEX_DIR))),$(LAW_INDEX_DIR),$(CURDIR)/$(LAW_INDEX_DIR))" \
		--collection "$(LAW_COLLECTION)"
	@echo "将索引绝对路径和输出中的 embedding_model_digest 填入 backend/.env；详见 docs/setup.md。"

test: test-backend test-frontend

test-backend:
	cd backend && $(PYTHON) -m pytest -q \
		tests/infrastructure/config/test_deployment_config.py \
		tests/security/test_jwt.py \
		tests/infrastructure/database/test_db_config.py \
		tests/infrastructure/database/test_redis_config.py \
		tests/integration/test_main_startup.py \
		tests/api/v1/routers/test_phase6_api_contract.py

test-frontend:
	npm --prefix frontend test -- --run

eval:
	cd backend && $(PYTHON) ../evaluation/run_eval.py

compose-config:
	docker compose config --quiet

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
