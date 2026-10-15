"""识别模型已经持有的关闭协议，不触发模型构造或远程访问。"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CloseAction:
    """保留实际资源身份与释放方法，避免浅拷贝共享连接被重复关闭。"""

    target: Any
    call: Callable
    asynchronous: bool


def close_actions(resource: Any) -> tuple[CloseAction, ...]:
    """优先使用对象自身协议，否则查找供应商同步与异步客户端。"""

    def action(target):
        if target is None:
            return None
        aclose = getattr(target, "aclose", None)
        if callable(aclose):
            return CloseAction(target, aclose, True)
        close = getattr(target, "close", None)
        if callable(close):
            return CloseAction(target, close, inspect.iscoroutinefunction(close))
        return None

    own = action(resource)
    if own is not None:
        return (own,)
    result = []
    seen = set()
    for name in ("client", "_client", "_async_client"):
        item = action(getattr(resource, name, None))
        if item is not None and id(item.target) not in seen:
            seen.add(id(item.target))
            result.append(item)
    return tuple(result)
