from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from app.errors.codes import ErrorCode
from app.errors.exceptions import AppException


TokenCount = int | Literal["unknown"]
CostValue = float | Literal["unknown"]


def _new_id() -> str:
    return uuid.uuid4().hex


def _hash_value(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def build_safe_metadata(values: dict[str, object]) -> dict[str, object]:
    """仅保留字段名、规范化值的 hash 和长度，不保存原始值。"""
    safe: dict[str, object] = {"field_names": sorted(str(key) for key in values)}
    for key, value in values.items():
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        safe[str(key)] = {
            "sha256": _hash_value(value),
            "length": len(value) if isinstance(value, (str, list, tuple, dict)) else len(canonical),
            "field_names": sorted(str(item) for item in value) if isinstance(value, dict) else [],
        }
    return safe


def _is_safe_metadata(value: dict[str, object]) -> bool:
    field_names = value.get("field_names")
    if not isinstance(field_names, list) or not all(isinstance(item, str) for item in field_names):
        return False
    if set(value) != {"field_names", *field_names}:
        return False
    for name in field_names:
        summary = value.get(name)
        if not isinstance(summary, dict):
            return False
        if not isinstance(summary.get("sha256"), str) or not str(summary["sha256"]).startswith("sha256:"):
            return False
        if not isinstance(summary.get("length"), int):
            return False
        if not isinstance(summary.get("field_names"), list):
            return False
    return True


@dataclass
class TraceEvent:
    """统一的 span/event 契约；事件类型无关字段允许显式缺省。"""

    event_type: str
    name: str
    trace_id: str
    span_id: str
    request_id: str
    session_id: str
    parent_span_id: str | None = None
    node: str | None = None
    route_reason: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    input_tokens: TokenCount = "unknown"
    output_tokens: TokenCount = "unknown"
    duration_ms: float | None = None
    attempt: int | None = None
    outcome: str | None = None
    cache_hit: bool | None = None
    cost_usd: CostValue = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _started_ns: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """在事件边界强制摘要化 metadata，避免调用方误传原始值。"""
        if not _is_safe_metadata(self.metadata):
            self.metadata = build_safe_metadata(self.metadata)

    def to_dict(self) -> dict[str, object]:
        """返回可序列化事件，不暴露内部排序字段。"""
        result = asdict(self)
        result.pop("_started_ns", None)
        return result


class BoundedTraceStore:
    """线程安全的有界进程内事件存储。"""

    def __init__(self, max_events: int = 5000):
        if max_events <= 0:
            raise ValueError("观测事件上限必须大于 0")
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def append(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def export_tree(self, request_id: str) -> list[dict[str, object]]:
        """按 span 开始顺序导出请求调用树。"""
        events = [event for event in self.events() if event.request_id == request_id]
        events.sort(key=lambda event: event._started_ns)
        return [event.to_dict() for event in events]


@dataclass(frozen=True)
class _SpanContext:
    trace_id: str
    span_id: str
    request_id: str
    session_id: str
    node: str | None


_current_span: ContextVar[_SpanContext | None] = ContextVar("current_trace_span", default=None)


def current_trace_context() -> dict[str, str | None] | None:
    """返回当前异步上下文的关联标识，供网关与检索层继承。"""
    current = _current_span.get()
    if current is None:
        return None
    return {
        "trace_id": current.trace_id,
        "span_id": current.span_id,
        "request_id": current.request_id,
        "session_id": current.session_id,
        "node": current.node,
    }


@contextmanager
def trace_span(
    store: BoundedTraceStore,
    *,
    event_type: str,
    name: str,
    request_id: str | None = None,
    session_id: str | None = None,
    node: str | None = None,
    route_reason: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    input_tokens: TokenCount = "unknown",
    output_tokens: TokenCount = "unknown",
    attempt: int | None = None,
    cache_hit: bool | None = None,
    metadata: dict[str, object] | None = None,
) -> Iterator[TraceEvent]:
    """建立显式父子 span，并在退出时写入单个完成事件。"""
    parent = _current_span.get()
    started_ns = time.monotonic_ns()
    event = TraceEvent(
        event_type=event_type,
        name=name,
        trace_id=parent.trace_id if parent else _new_id(),
        span_id=_new_id(),
        parent_span_id=parent.span_id if parent else None,
        request_id=request_id or (parent.request_id if parent else _new_id()),
        session_id=session_id or (parent.session_id if parent else "unknown"),
        node=node or (parent.node if parent else None),
        route_reason=route_reason,
        model=model,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempt=attempt,
        cache_hit=cache_hit,
        metadata=metadata or {},
        outcome="success",
        _started_ns=started_ns,
    )
    token = _current_span.set(
        _SpanContext(
            trace_id=event.trace_id,
            span_id=event.span_id,
            request_id=event.request_id,
            session_id=event.session_id,
            node=event.node,
        )
    )
    try:
        yield event
    except BaseException:
        if event.outcome == "success":
            event.outcome = "error"
        raise
    finally:
        event.duration_ms = round((time.monotonic_ns() - started_ns) / 1_000_000, 3)
        _current_span.reset(token)
        store.append(event)


class SessionBudgetExceeded(AppException):
    """单会话调用数或 token 预算已耗尽。"""

    code = ErrorCode.SESSION_BUDGET_EXCEEDED
    status_code = 429
    message = "本会话自动调用预算已用尽，请转人工处理"

    def __init__(self, *, session_id: str, dimension: Literal["calls", "tokens", "sessions"]):
        self.session_id = session_id
        self.dimension = dimension
        super().__init__(detail=f"session_id={session_id}, dimension={dimension}")


class TokenCostEstimator:
    """仅依据显式模型单价和供应商 usage 估算美元成本。"""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None):
        self._prices = prices or {}

    @classmethod
    def from_env(cls) -> "TokenCostEstimator":
        """读取每百万 token 的输入、输出美元单价映射。"""
        raw = os.getenv("MODEL_PRICING_USD_PER_MILLION", "{}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        prices: dict[str, dict[str, float]] = {}
        for model, value in parsed.items():
            if not isinstance(model, str) or not isinstance(value, dict):
                continue
            input_price = value.get("input")
            output_price = value.get("output")
            if isinstance(input_price, (int, float)) and isinstance(output_price, (int, float)):
                prices[model] = {"input": float(input_price), "output": float(output_price)}
        return cls(prices)

    def estimate(self, model: str, input_tokens: TokenCount, output_tokens: TokenCount) -> CostValue:
        price = self._prices.get(model)
        if price is None or not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return "unknown"
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        return round(cost, 8)


class SessionBudget:
    """进程内、线程安全的单会话宽松调用与 token 预算。"""

    def __init__(self, *, max_calls: int, max_tokens: int, max_sessions: int = 10000):
        if max_calls <= 0 or max_tokens <= 0 or max_sessions <= 0:
            raise ValueError("session budget 必须大于 0")
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_sessions = max_sessions
        self._usage: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def _usage_for(self, session_id: str) -> dict[str, int]:
        usage = self._usage.get(session_id)
        if usage is not None:
            return usage
        if len(self._usage) >= self.max_sessions:
            raise SessionBudgetExceeded(session_id=session_id, dimension="sessions")
        usage = {"calls": 0, "tokens": 0}
        self._usage[session_id] = usage
        return usage

    def reserve_call(self, session_id: str) -> None:
        with self._lock:
            usage = self._usage_for(session_id)
            if usage["calls"] >= self.max_calls:
                raise SessionBudgetExceeded(session_id=session_id, dimension="calls")
            usage["calls"] += 1

    def record_tokens(self, session_id: str, *, input_tokens: TokenCount, output_tokens: TokenCount) -> None:
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return
        token_count = input_tokens + output_tokens
        with self._lock:
            usage = self._usage_for(session_id)
            usage["tokens"] += token_count
            if usage["tokens"] > self.max_tokens:
                raise SessionBudgetExceeded(session_id=session_id, dimension="tokens")

    def snapshot(self, session_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self._usage.get(session_id, {"calls": 0, "tokens": 0}))


trace_store = BoundedTraceStore(max_events=int(os.getenv("TRACE_MAX_EVENTS", "5000")))
session_budget = SessionBudget(
    max_calls=int(os.getenv("SESSION_MAX_CALLS", os.getenv("SESSION_MAX_MODEL_CALLS", "40"))),
    max_tokens=int(os.getenv("SESSION_MAX_TOKENS", "120000")),
    max_sessions=int(os.getenv("SESSION_BUDGET_MAX_SESSIONS", "10000")),
)
cost_estimator = TokenCostEstimator.from_env()
