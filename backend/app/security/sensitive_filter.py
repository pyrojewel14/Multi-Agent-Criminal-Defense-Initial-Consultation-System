"""输入兼容入口与独立高风险检测。

当前停用案情 PII 掩码，原有调用入口原样返回文本。
日志过滤与权限控制由各自模块负责。
"""

import re
from typing import Tuple

from app.infrastructure.logging import get_logger

_logger = get_logger("Security.SensitiveFilter")

_MINOR_PATTERNS = [
    re.compile(r"(未满|不满|小于|不足)\s*[零〇一二两三四五六七八九十百\d]+\s*(岁|周岁)"),
    re.compile(r"\d+\s*(岁|周岁)\s*(以下|以内)"),
    re.compile(r"小孩|儿童|未成年人|未成人"),
]

_HIGH_RISK_PATTERNS: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"是我干的|是我做的|是我杀的|我承认.*?(杀|偷|抢|骗|抢|盗|掠|奸)"), "SELF_INCrimination"),
    (re.compile(r"我确实[\u4e00-\u9fa5]{0,10}(做|杀|偷|抢|骗|抢|盗|犯|承认)"), "SELF_INCrimination"),
    (re.compile(r"我当时[\u4e00-\u9fa5]{0,20}(故意的|故意的|过失)"), "SELF_INCrimination"),
    (re.compile(r"帮我隐瞒|不要告诉|不能说出去|保密|统一口径|跟我串供"), "COLLUSION"),
    (re.compile(r"把证据删了|帮我伪造|销毁证据|毁灭证据|篡改"), "EVIDENCE_TAMPERING"),
    (re.compile(r"我们商量好?了|我们约好?了|我们统一|我们编造"), "COLLUSION"),
    (re.compile(r"律师.*?告诉你|律师.*?指导|律师.*?指使|教我.*?说"), "STRATEGY_LEAKAGE"),
]


def _mask_name(text: str) -> str:
    """保留姓名掩码调用入口，当前原样返回文本。"""
    return text


def mask_pii(text: str) -> str:
    """保留 PII 掩码调用入口，当前原样返回文本。

    姓名、地址、手机号、身份证号和车牌均不再遮盖；
    调用该入口不能视为已完成隐私保护。
    """
    return text


def detect_high_risk(text: str) -> Tuple[bool, str]:
    """检测文本中是否存在高风险语句。

    Args:
        text: 用户输入的原始文本。

    Returns:
        Tuple[bool, str]: (是否高风险, 风险类型)。
        风险类型包括: SELF_INCrimination(自认其罪)、COLLUSION(串供意图)、
        EVIDENCE_TAMPERING(伪造/销毁证据)、STRATEGY_LEAKAGE(辩护策略泄露)。
        如果未检测到风险, 返回 (False, "")。
    """
    if not text:
        return False, ""

    for pattern, risk_type in _HIGH_RISK_PATTERNS:
        if pattern.search(text):
            _logger.warning("【detect_high_risk】检测到高风险语句 | 风险类型: %s | 文本长度: %d", risk_type, len(text))
            return True, risk_type

    for minor_pattern in _MINOR_PATTERNS:
        if minor_pattern.search(text):
            _logger.warning("【detect_high_risk】检测到未成年人相关信息 | 文本长度: %d", len(text))
            return True, "MINOR_INVOLVED"

    return False, ""


def sanitize_input(text: str) -> str:
    """保留输入处理入口，当前原样返回文本，仅记录长度。

    此入口没有独立文本清理行为。风险分类仍需额外调用
    ``detect_high_risk``；日志不记录文本正文。
    """
    if not text:
        return text

    result = mask_pii(text)
    _logger.info("【sanitize_input】输入原样透传 | 原始长度: %d | 返回长度: %d", len(text), len(result))
    return result
