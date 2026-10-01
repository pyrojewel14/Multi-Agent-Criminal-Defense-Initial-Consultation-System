"""公开合成案例的 Ollama 云对照入口；仅修改评测进程的模型副本。"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def masked_case_facts(cases: list[dict], query_mode: str) -> set[str]:
    """只允许原案例经过现有脱敏得到的模型事实，不更改原输入或评分。"""
    from app.security.sensitive_filter import mask_pii

    key = "semantic_query" if query_mode == "semantic" else "query"
    return {mask_pii(json.dumps({"behavior_sequence": [case[key]], "consequence": ""},
                               ensure_ascii=False, default=str)) for case in cases}


class CloudCapture:
    """临时适配思考级别并记录实际云请求；退出时恢复工厂、流方法和事件钩子。"""

    def __init__(self, model: str, thinking: str | bool, raw_output: Path,
                 allowed_facts: set[str], articles: dict[str, Any]):
        self.model = model
        self.thinking = thinking
        self.raw_output = raw_output
        self.allowed_facts = allowed_facts
        self.articles = articles
        self.records: list[dict[str, Any]] = []

    def verify_payload(self, request: dict) -> None:
        """拒绝模型回落、未批准事实、原生schema及快照外正文；gold不进入请求。"""
        if request.get("model") != self.model or request.get("think") != self.thinking:
            raise ValueError("云模型或思考配置不一致")
        if "format" in request or "num_predict" in request.get("options", {}):
            raise ValueError("云对照不得使用未经核验的本地输出参数")
        facts = json.loads(request["messages"][1]["content"])["facts"]
        if facts not in self.allowed_facts:
            raise ValueError("模型输入不属于批准的公开合成案例")
        for message in request["messages"]:
            if message["role"] != "tool":
                continue
            result = json.loads(message["content"])["result"]
            article = result.get("article")
            if article:
                known = self.articles[article["article_id"]]
                if article["content"] != known["content"] or article["required_elements"] != known.get("elements", []):
                    raise ValueError("工具正文或要件超出公开快照")
            if "required_elements" in result and result["required_elements"] != self.articles[result["article_id"]].get("elements", []):
                raise ValueError("工具要件超出公开快照")

    def __enter__(self):
        from langchain_ollama import ChatOllama

        from app.infrastructure.llm.factory import chat_model_factory

        self.original_factory = chat_model_factory.create_precise_model
        self.original_stream = ChatOllama._acreate_chat_stream

        def factory(temperature: float = 0.1):
            model = self.original_factory(temperature)
            if not isinstance(model, ChatOllama) or model.model != self.model:
                raise ValueError("禁止回落至其他供应商或本地模型")
            return model.model_copy(update={"reasoning": self.thinking})

        owner = self

        async def stream(self, messages, stop=None, **kwargs):
            async for chunk in owner.capture(self, messages, stop, **kwargs):
                yield chunk

        chat_model_factory.create_precise_model = factory
        ChatOllama._acreate_chat_stream = stream
        return self

    def __exit__(self, *_):
        from langchain_ollama import ChatOllama

        from app.infrastructure.llm.factory import chat_model_factory

        chat_model_factory.create_precise_model = self.original_factory
        ChatOllama._acreate_chat_stream = self.original_stream

    async def capture(self, model, messages, stop=None, **kwargs):
        """保存无请求头的实际HTTP载荷与原始公开案例响应，部分流不伪填token。"""
        started = time.monotonic()
        row = {"requested_model": model.model, "request_url": None, "wire_request": None,
               "http_status": None, "content": "", "thinking": "", "tool_calls": [],
               "last_response": None, "chunks": 0}
        self.records.append(row)
        client = model._async_client._client
        original_request_hooks = list(client.event_hooks.get("request", []))
        original_response_hooks = list(client.event_hooks.get("response", []))

        async def request_hook(request):
            if request.url.path == "/api/chat":
                row["request_url"] = str(request.url)
                row["wire_request"] = json.loads(request.content)
                self.verify_payload(row["wire_request"])

        async def response_hook(response):
            if response.request.url.path == "/api/chat":
                row["http_status"] = response.status_code
                if response.status_code >= 400:
                    await response.aread()
                    row["error_response_body"] = response.json()

        client.event_hooks["request"] = original_request_hooks + [request_hook]
        client.event_hooks["response"] = original_response_hooks + [response_hook]
        try:
            async for chunk in self.original_stream(model, messages, stop=stop, **kwargs):
                value: Any = chunk
                raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
                row["last_response"] = raw
                expected = self.model.removesuffix("-cloud").removesuffix(":cloud")
                if raw.get("model") not in {self.model, expected}:
                    raise ValueError("云服务返回模型与请求模型不一致")
                message = raw.get("message") or {}
                row["content"] += message.get("content") or ""
                row["thinking"] += message.get("thinking") or ""
                row["tool_calls"].extend(message.get("tool_calls") or [])
                row["chunks"] += 1
                yield chunk
        except BaseException as exc:
            row["exception_type"] = type(exc).__name__
            row["exception_status_code"] = getattr(exc, "status_code", None)
            raise
        finally:
            client.event_hooks["request"] = original_request_hooks
            client.event_hooks["response"] = original_response_hooks
            row["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
            self.raw_output.parent.mkdir(parents=True, exist_ok=True)
            self.raw_output.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n")


async def evaluate(args) -> dict:
    """核实云能力后复用原正式评分；外部拒绝不回落，本机配置不永久修改。"""
    from app.knowledge.law_knowledge import _build_article_index, load_criminal_law_data
    from evaluation.run_full_eval import run_evaluation

    if not args.model.endswith(("-cloud", ":cloud")):
        raise ValueError("必须明确指定云tag，禁止下载或调用本地权重")
    if args.output.resolve() == args.raw_output.resolve():
        raise ValueError("summary与raw输出不能是相同路径")
    if args.output.exists() or args.raw_output.exists():
        raise ValueError("输出已存在，请用新路径保留历史证据")
    thinking: str | bool = args.thinking == "true" if args.thinking in {"true", "false"} else args.thinking
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("案例文件为空")
    selected = [case for case in cases if not args.case or case["id"] in args.case]
    if not selected:
        raise ValueError("未选择任何案例")
    if args.case and {case["id"] for case in selected} != set(args.case):
        raise ValueError("未知案例")
    async with httpx.AsyncClient(base_url=args.base_url, trust_env=False, timeout=15) as client:
        response = await client.post("/api/show", json={"model": args.model})
        response.raise_for_status()
        show = response.json()
        if thinking not in show.get("thinking", {}).get("values", []):
            raise ValueError("show未确认该模型支持指定思考级别")
        if "tools" not in show.get("capabilities", []):
            raise ValueError("show未确认该模型支持工具调用")
        response = await client.get("/api/tags")
        response.raise_for_status()
        tag = next((item for item in response.json()["models"] if item["name"] == args.model), None)
    corpus = load_criminal_law_data(profile="full")
    with CloudCapture(args.model, thinking, args.raw_output,
                      masked_case_facts(selected, args.query_mode), _build_article_index(corpus)) as capture:
        result = await run_evaluation("live-lawref", args.output, args.case, args.query_mode, cases_path=args.cases)
    result["actual_execution"] = {
        "requested_model": args.model,
        "returned_models": sorted({row["last_response"]["model"] for row in capture.records if row["last_response"]}),
        "provider": "Ollama cloud-tag proxy", "base_url": args.base_url, "thinking": thinking,
        "show_metadata": show, "catalog_entry": tag,
        "remote_weights_digest_verified": None,
        "proxy_manifest_digest": tag.get("digest") if tag else None,
        "remote_host_observed": tag.get("remote_host") if tag else None,
        "http_statuses": [row["http_status"] for row in capture.records],
        "protocol": "legacy; evaluation-only model copy; no native format/num_predict",
        "local_fallback": False, "raw_output": str(args.raw_output.resolve()),
        "raw_sha256": "sha256:" + hashlib.sha256(args.raw_output.read_bytes()).hexdigest(),
        "payload_check": "原案例现有mask_pii完整输出与公开快照逐字核对；gold不传模型",
    }
    result["llm_executed"] = any(row["last_response"] for row in capture.records)
    # 按真实工具执行轨迹记录检索，不继承mode所表达的运行意图。
    result["rag_executed"] = any(
        step["tool_name"] == "search_laws" and step["tool_status"] in {
            "success", "empty", "partial_dependency_failure", "dependency_failure", "timeout",
        }
        for case in result["cases"] for step in case["trajectory"]
    )
    result["actual_execution"]["rag_execution_basis"] = "search_laws实际执行轨迹；不等于所有召回均成功"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    """显式CLI只作用当前评测进程；账户或能力拒绝不会尝试其他模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", choices=["low", "medium", "high", "true", "false"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--case", action="append")
    parser.add_argument("--query-mode", choices=["article", "semantic"], default="article")
    args = parser.parse_args()
    os.environ.update({"LLM_TYPE": "OLLAMA", "OLLAMA_MODEL_NAME": args.model,
                       "OLLAMA_BASE_URL": args.base_url, "LAW_AGENT_FINAL_PROTOCOL": "legacy",
                       "LAW_KNOWLEDGE_PROFILE": "full"})
    result = asyncio.run(evaluate(args))
    print(json.dumps({"passed": result["passed"], "total": result["total"], "model": args.model}))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    # 支持从仓库根以脚本运行，同时保持测试中的evaluation包导入方式。
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
