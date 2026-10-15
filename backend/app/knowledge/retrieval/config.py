"""独立解析并校验检索配置，不导入旧流程或初始化模型。"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass


def _bounded(name: str, value: float, lower: float, upper: float) -> None:
    """拒绝非数值、非有限值及超出阶段预算的配置。"""
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"invalid bounded setting: {name}")


def _integer(name: str, value: int, lower: int, upper: int | None = None) -> None:
    """拒绝布尔值与隐式截断，仅允许范围内的整数。"""
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid integer setting: {name}")


def _env_integer(env: Mapping[str, str], name: str, default: int) -> int:
    """兼容整数形式的小数表示，但不截断非整数配置。"""
    value = float(env.get(name, str(default)))
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"invalid integer setting: {name}")
    return int(value)


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """沿用严格开关拼写，错误值不得静默开启或关闭阶段。"""
    value = env.get(name, str(default)).lower()
    if value not in {"true", "false", "1", "0", "on", "off"}:
        raise ValueError(f"invalid flag: {name}")
    return value in {"true", "1", "on"}


@dataclass(frozen=True)
class RetrievalConfig:
    """一次组装使用的不可变召回、扩展与重排配置。"""

    mode: str = "hybrid"
    recall_k: int = 20
    hyde_k: int = 10
    rerank: bool = True
    rerank_timeout: float = 30.0
    hyde: bool = False
    hyde_timeout: float = 10.0
    hyde_budget: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"vector", "hybrid"}:
            raise ValueError("invalid retrieval mode")
        _integer("recall_k", self.recall_k, 5, 50)
        _integer("hyde_k", self.hyde_k, 1, 20)
        _integer("hyde_budget", self.hyde_budget, 0, 1)
        _bounded("hyde_timeout", self.hyde_timeout, 0.01, 30)
        _bounded("rerank_timeout", self.rerank_timeout, 0.01, 120)
        if type(self.hyde) is not bool or type(self.rerank) is not bool:
            raise ValueError("retrieval stage switches must be bool")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RetrievalConfig:
        """读取现有部署变量；显式传入映射时不再读取进程环境。"""
        values = dict(os.environ if env is None else env)
        return cls(
            mode=values.get("LAW_FULL_RETRIEVAL_MODE", "hybrid"),
            recall_k=_env_integer(values, "LAW_FULL_RECALL_K", 20),
            hyde_k=_env_integer(values, "LAW_FULL_HYDE_K", 10),
            rerank=_flag(values, "LAW_FULL_RERANK_ENABLED", True),
            rerank_timeout=float(values.get("LAW_FULL_RERANK_TIMEOUT_SECONDS", "30")),
            hyde=_flag(values, "LAW_FULL_HYDE_ENABLED", False),
            hyde_timeout=float(values.get("LAW_FULL_HYDE_TIMEOUT_SECONDS", "10")),
            hyde_budget=_env_integer(values, "LAW_FULL_HYDE_CALL_BUDGET", 1),
        )
