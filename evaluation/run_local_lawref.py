"""固定公开六例的有限本地 Ollama 对照；不下载模型或修改服务配置。"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
CASES = Path(__file__).with_name("lawref_cloud_cases.jsonl")
CASES_SHA256 = "b5958b42d62ede849bafc245d3e31a4e00ac6c653806471908b3f10bcb7c1985"
MODELS = ("qwen3.5:4b", "qwen3.5:9b")
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1024}
FIXED_ENV = {
    "LLM_TYPE": "OLLAMA",
    "EMBED_MODEL_TYPE": "OLLAMA",
    "LAW_KNOWLEDGE_PROFILE": "full",
    "LAW_FULL_DEMO_ANNOTATIONS": "on",
    "LAW_AGENT_FINAL_PROTOCOL": "legacy",
    "LAW_AGENT_MAX_STEPS": "4",
    "LAW_AGENT_TIMEOUT_SECONDS": "240",
    "LAW_AGENT_TOOL_TIMEOUT_SECONDS": "90",
    "LLM_TOTAL_TIMEOUT_SECONDS": "30",
    "LLM_ATTEMPT_TIMEOUT_SECONDS": "15",
    "LLM_MAX_ATTEMPTS": "2",
    "LLM_RETRY_BACKOFF_SECONDS": "0.25",
    "MEMORY_MODEL_WINDOW": "8192",
    "MEMORY_OUTPUT_RESERVE": "1024",
    "MEMORY_CONTEXT_TOKEN_BUDGET": "7168",
    "LAW_FULL_RETRIEVAL_MODE": "vector",
    "LAW_FULL_RERANK_ENABLED": "false",
    "LAW_FULL_HYDE_ENABLED": "false",
}


def write_json(path: Path, value) -> None:
    """写入本次独立输出；路径的新旧检查由入口统一完成。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cases(path: Path) -> list[dict]:
    """固定原六例字节，避免改输入或 gold 后仍沿用原对照分母。"""
    if hashlib.sha256(path.read_bytes()).hexdigest() != CASES_SHA256:
        raise ValueError("输入必须保持固定六例原字节")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_args(args) -> None:
    """在联网或复制索引前拒绝无界轮数、未知模型及历史输出覆盖。"""
    if args.rounds not in {1, 2}:
        raise ValueError("轮数只能是1或2；默认首轮后协调资源")
    if not args.models or len(set(args.models)) != len(args.models) or set(args.models) - set(MODELS):
        raise ValueError("只允许明确的4b/9b本地候选，且不能重复")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("输出目录已存在内容，请用新路径保留历史证据")


async def probe(client: httpx.AsyncClient, base_url: str, models: list[str]) -> dict:
    """只读服务与模型元数据；不可达不生成六例分数，也不触发下载。"""
    report = {
        "base_url": base_url,
        "trust_env": False,
        "llm_executed": False,
        "available_models": [],
        "models": {},
        "requests": [],
    }

    async def request(method, endpoint, **kwargs):
        row = {"endpoint": endpoint}
        report["requests"].append(row)
        try:
            response = await client.request(method, base_url.rstrip("/") + endpoint, **kwargs)
            row["http_status"] = response.status_code
            response.raise_for_status()
            row["body"] = response.json()
            return row["body"]
        except (httpx.HTTPError, ValueError) as exc:
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            return None

    report["version"] = await request("GET", "/api/version")
    tags = await request("GET", "/api/tags")
    report["residency"] = await request("GET", "/api/ps")
    for model in models:
        tag = next((item for item in (tags or {}).get("models", []) if item.get("name") == model), None)
        row = report["models"][model] = {"catalog_entry": tag}
        if tags is None:
            row["status"] = "service_unreachable"
        elif tag is None:
            row["status"] = "not_installed"
        elif tag.get("remote_host") or tag.get("remote_model"):
            row["status"] = "cloud_manifest_rejected"
        else:
            show = await request("POST", "/api/show", json={"model": model})
            row["show"] = show
            row["status"] = "available" if show and "tools" in show.get("capabilities", []) else "tools_unverified"
            if row["status"] == "available":
                report["available_models"].append(model)
    return report


class LocalCapture:
    """评测进程内显式旁路代理，核对公开载荷并保存部分流及实际 usage。"""

    def __init__(self, model, base_url, raw_output, cases, articles, *, max_requests=48):
        self.model, self.base_url, self.raw_output = model, base_url, raw_output
        self.facts = [{"behavior_sequence": [case["query"]], "consequence": ""} for case in cases]
        self.articles = articles
        self.max_requests = max_requests
        self.records = []
        self.client_kwargs = {"trust_env": False}

    def verify_payload(self, request: dict) -> None:
        """核对当前 legacy 消息、固定事实与已读快照；gold不进入模型。"""
        from app.consultation.agents.legal_research import _SYSTEM_PROMPT

        if request.get("model") != self.model or request.get("think") is not False:
            raise ValueError("本地模型或思考协议不一致")
        if "format" in request or any(request.get("options", {}).get(k) != v for k, v in OPTIONS.items()):
            raise ValueError("本地对照参数不一致")
        messages = request["messages"]
        if messages[0]["content"] != _SYSTEM_PROMPT:
            raise ValueError("系统提示偏离当前legacy协议")
        prompt = json.loads(messages[1]["content"])
        facts = prompt["facts"]
        facts = json.loads(facts) if isinstance(facts, str) else facts
        if facts not in self.facts:
            raise ValueError("模型输入不属于固定公开案例")
        if set(prompt) - {"facts", "read_articles"}:
            raise ValueError("模型输入包含额外评分字段")
        for article in prompt.get("read_articles", []):
            known = self.articles[article["article_id"]]
            if article["content"] != known["content"] or article["required_elements"] != [
                item["name"] if isinstance(item, dict) else item for item in known.get("elements", [])
            ]:
                raise ValueError("已读正文或要件超出公开快照")
        for message in messages:
            if message["role"] != "tool":
                continue
            result = json.loads(message["content"])["result"]
            article = result.get("article")
            if article:
                known = self.articles[article["article_id"]]
                if article["content"] != known["content"] or article["required_elements"] != known.get("elements", []):
                    raise ValueError("工具正文或要件超出公开快照")
            if "required_elements" in result and result["required_elements"] != self.articles[result["article_id"]].get(
                "elements", []
            ):
                raise ValueError("工具要件超出公开快照")

    def __enter__(self):
        from app.infrastructure.llm.factory import chat_model_factory
        from app.infrastructure.llm.gateway import LLMCallPolicy, llm_gateway
        from langchain_ollama import ChatOllama

        self.original_factory = chat_model_factory.create_precise_model
        self.original_stream = ChatOllama._acreate_chat_stream
        self.original_policy = llm_gateway._policy
        owner = self

        def factory(temperature=0):
            return ChatOllama(
                model=owner.model,
                base_url=owner.base_url,
                reasoning=False,
                client_kwargs=owner.client_kwargs,
                **OPTIONS,
            )

        async def stream(model, messages, stop=None, **kwargs):
            async for chunk in owner.capture(model, messages, stop, **kwargs):
                yield chunk

        chat_model_factory.create_precise_model = factory
        ChatOllama._acreate_chat_stream = stream
        llm_gateway._policy = LLMCallPolicy()
        return self

    def __exit__(self, *_):
        from app.infrastructure.llm.factory import chat_model_factory
        from app.infrastructure.llm.gateway import llm_gateway
        from langchain_ollama import ChatOllama

        chat_model_factory.create_precise_model = self.original_factory
        ChatOllama._acreate_chat_stream = self.original_stream
        llm_gateway._policy = self.original_policy

    async def capture(self, model, messages, stop=None, **kwargs):
        """流取消时保留原始片段；不把未完成请求的token或结束原因伪填为零。"""
        if len(self.records) >= self.max_requests:
            raise ValueError("达到本轮请求预算")
        row = {
            "wire_request": None,
            "http_status": None,
            "content": "",
            "thinking": "",
            "tool_calls": [],
            "last_response": None,
            "candidate_response_observed": False,
            "returned_models": [],
        }
        self.records.append(row)
        started = time.monotonic()
        client = model._async_client._client
        hooks = {key: list(value) for key, value in client.event_hooks.items()}

        async def request_hook(request):
            row["wire_request"] = json.loads(request.content)
            try:
                self.verify_payload(row["wire_request"])
            except (ValueError, KeyError, TypeError):
                row["validation_error"] = "payload_mismatch"
                raise

        async def response_hook(response):
            row["http_status"] = response.status_code
            if response.status_code >= 400:
                await response.aread()
                row["error_response_body"] = response.text

        client.event_hooks["request"] = hooks.get("request", []) + [request_hook]
        client.event_hooks["response"] = hooks.get("response", []) + [response_hook]
        try:
            async for chunk in self.original_stream(model, messages, stop=stop, **kwargs):
                raw = chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else dict(chunk)
                row["last_response"] = raw
                if raw.get("model") not in row["returned_models"]:
                    row["returned_models"].append(raw.get("model"))
                if raw.get("model") != self.model:
                    row["validation_error"] = "model_identity_mismatch"
                    raise ValueError("服务返回模型与指定本地模型不一致")
                row["candidate_response_observed"] = True
                message = raw.get("message") or {}
                row["content"] += message.get("content") or ""
                row["thinking"] += message.get("thinking") or ""
                row["tool_calls"].extend(message.get("tool_calls") or [])
                yield chunk
        except BaseException as exc:
            row["exception_type"] = type(exc).__name__
            raise
        finally:
            client.event_hooks = hooks
            row["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            write_json(self.raw_output, self.records)


@contextmanager
def environment(values):
    """隔离当前评测配置并在失败时恢复，不写入.env或其他进程。"""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def residency(base_url: str) -> dict:
    """驻留采样失败只记录诊断，不丢弃已经完成的模型评测。"""
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
            response = await client.get(base_url.rstrip("/") + "/api/ps")
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error_type": type(exc).__name__}


async def run(args) -> dict:
    """先探测后逐模型逐轮执行，固定评分器、独立索引副本及SQLite路径。"""
    validate_args(args)
    cases = load_cases(CASES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
        preflight = await probe(client, args.base_url, args.models)
    write_json(args.output_dir / "preflight.json", preflight)
    result = {
        "status": "blocked_service_or_models",
        "preflight": preflight,
        "runs": [],
        "llm_executed": False,
        "candidate_llm_executed": False,
        "legal_review_completed": False,
        "production_deployment_verified": False,
    }
    if not preflight["available_models"]:
        write_json(args.output_dir / "comparison.json", result)
        return result

    # 只在服务可达后复制既有公共索引；Chroma读取可能写SQLite，不打开原索引。
    manifest_path = args.index_directory.parent / f"{args.index_directory.name}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    embedding = next(
        (
            item
            for row in preflight["requests"]
            if row["endpoint"] == "/api/tags"
            for item in row.get("body", {}).get("models", [])
            if item.get("name") == manifest["embedding_model"]
        ),
        None,
    )
    if not embedding or embedding.get("digest") != manifest["embedding_model_digest"]:
        result["status"] = "blocked_embedding_digest"
        write_json(args.output_dir / "comparison.json", result)
        return result
    state = args.output_dir / "state"
    isolated_index = state / args.index_directory.name
    shutil.copytree(args.index_directory, isolated_index)
    shutil.copy2(manifest_path, isolated_index.parent / f"{isolated_index.name}.manifest.json")
    config = {
        **FIXED_ENV,
        "OLLAMA_BASE_URL": args.base_url,
        "LAW_FULL_INDEX_DIRECTORY": isolated_index.resolve(),
        "LAW_FULL_INDEX_COLLECTION": manifest["collection"],
        "TEXT_EMBEDDING_MODEL_NAME": manifest["embedding_model"],
        "LAW_FULL_EMBEDDING_DIGEST": manifest["embedding_model_digest"],
        "DATABASE_PATH": (state / "eval.sqlite").resolve(),
        "LANGGRAPH_CHECKPOINT_DB_PATH": (state / "checkpoints.sqlite").resolve(),
    }
    with environment(config):
        from app.consultation.agents.legal_research import _SYSTEM_PROMPT
        from app.knowledge.law.repository import _build_article_index
        from app.knowledge.law.corpus import load_criminal_law_data

        from evaluation.run_full_eval import run_evaluation

        articles = _build_article_index(load_criminal_law_data(profile="full"))
        for model in preflight["available_models"]:
            for number in range(1, args.rounds + 1):
                name = model.replace(":", "-") + f"-run{number}"
                output = args.output_dir / f"{name}.json"
                raw_output = args.output_dir / f"{name}-raw.json"
                before = await residency(args.base_url)
                interruption = None
                with (
                    environment({"OLLAMA_MODEL_NAME": model}),
                    LocalCapture(model, args.base_url, raw_output, cases, articles) as capture,
                ):
                    # 六例每例240秒，另留60秒给评分/索引核验；HTTP尝试上限为6*4*2。
                    try:
                        summary = await asyncio.wait_for(
                            run_evaluation("live-lawref", output, query_mode="article", cases_path=CASES), timeout=1500
                        )
                    except Exception as exc:  # noqa: BLE001 — 评测中断必须落盘且恢复进程配置
                        interruption = type(exc).__name__
                        summary = {
                            "evidence_kind": "local-lawref-interrupted",
                            "cases": [],
                            "passed": None,
                            "total": None,
                            "error_type": interruption,
                            "legal_review_completed": False,
                            "production_deployment_verified": False,
                        }
                if not raw_output.exists():
                    write_json(raw_output, capture.records)
                after = await residency(args.base_url)
                summary["actual_execution"] = {
                    "model": model,
                    "digest": preflight["models"][model]["catalog_entry"]["digest"],
                    "protocol": "legacy",
                    "thinking": False,
                    "options": OPTIONS,
                    "trust_env": False,
                    "max_http_requests": 48,
                    "round_timeout_seconds": 1500,
                    "system_prompt_sha256": hashlib.sha256(_SYSTEM_PROMPT.encode()).hexdigest(),
                    "scorer_sha256": hashlib.sha256((ROOT / "evaluation/run_full_eval.py").read_bytes()).hexdigest(),
                    "http_statuses": [row["http_status"] for row in capture.records],
                    "raw_sha256": hashlib.sha256(raw_output.read_bytes()).hexdigest() if raw_output.exists() else None,
                    "residency_before": before,
                    "residency_after": after,
                    "memory_scope": "Ollama模型驻留快照size_vram；不是进程峰值或整机内存峰值",
                    "identity_verified": (
                        not any(row.get("validation_error") == "model_identity_mismatch" for row in capture.records)
                    )
                    if any(row["last_response"] for row in capture.records)
                    else None,
                    "payload_verified": (
                        not any(row.get("validation_error") == "payload_mismatch" for row in capture.records)
                    )
                    if any(row["wire_request"] for row in capture.records)
                    else None,
                    "returned_models": sorted(
                        {
                            returned
                            for row in capture.records
                            for returned in row["returned_models"]
                            if returned is not None
                        }
                    ),
                }
                summary["llm_executed"] = any(row["last_response"] for row in capture.records)
                summary["candidate_llm_executed"] = any(row["candidate_response_observed"] for row in capture.records)
                summary["rag_executed"] = any(
                    step["tool_name"] == "search_laws"
                    and step["tool_status"]
                    in {"success", "empty", "partial_dependency_failure", "dependency_failure", "timeout"}
                    for case in summary["cases"]
                    for step in case.get("trajectory", [])
                )
                summary["actual_execution"]["rag_execution_basis"] = "search_laws实际尝试轨迹；不等于召回成功"
                if interruption:
                    # 评分器未交回完整轨迹时，不把无法确认的检索状态写成未执行。
                    summary["rag_executed"] = None
                    summary["actual_execution"]["rag_execution_basis"] = "中断轮未交回完整工具轨迹，检索状态未知"
                round_status = (
                    "runner_interrupted"
                    if interruption
                    else "identity_mismatch"
                    if summary["actual_execution"]["identity_verified"] is False
                    else "payload_mismatch"
                    if summary["actual_execution"]["payload_verified"] is False
                    else "no_inference"
                    if not summary["candidate_llm_executed"]
                    else "completed"
                )
                summary["round_status"] = round_status
                if round_status != "completed":
                    # 中断或仪器失配保留执行事实与逐例诊断，不生成该候选的六例评分。
                    summary["attempted_case_count"] = summary["total"]
                    summary["passed"] = summary["total"] = None
                write_json(output, summary)
                result["runs"].append(
                    {
                        "model": model,
                        "round": number,
                        "passed": summary["passed"],
                        "total": summary["total"],
                        "summary": output.name,
                        "raw": raw_output.name,
                        "status": round_status,
                        "llm_executed": summary["llm_executed"],
                        "candidate_llm_executed": summary["candidate_llm_executed"],
                    }
                )
                result["llm_executed"] |= summary["llm_executed"]
                result["candidate_llm_executed"] |= summary["candidate_llm_executed"]
                if interruption:
                    result.update(status="runner_interrupted", error_type=interruption)
                write_json(args.output_dir / "comparison.json", result)
                if interruption:
                    return result
    if any(row["status"] == "identity_mismatch" for row in result["runs"]):
        result["status"] = "blocked_identity_mismatch"
    elif any(row["status"] == "payload_mismatch" for row in result["runs"]):
        result["status"] = "blocked_payload_mismatch"
    elif any(row["total"] is None for row in result["runs"]):
        result["status"] = "blocked_no_inference"
    else:
        result["status"] = (
            "evaluated" if len(preflight["available_models"]) == len(args.models) else "partially_evaluated"
        )
    write_json(args.output_dir / "comparison.json", result)
    return result


def main() -> int:
    """服务不可达退出2，实际运行有失败退出1，全部六例通过才退出0。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--rounds", type=int, choices=[1, 2], default=1)
    parser.add_argument("--index-directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({"status": result["status"], "runs": result["runs"]}, ensure_ascii=False))
    if result["status"] != "evaluated":
        return 2
    return 0 if all(row["passed"] == row["total"] == 6 for row in result["runs"]) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
