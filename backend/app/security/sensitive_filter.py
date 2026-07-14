"""敏感信息过滤器模块。

提供 PII 检测与掩码、高风险语句检测、输入清理等功能，
确保用户隐私保护和风险提示。
"""

import re
from typing import Tuple

from app.utils.logger import get_logger

_logger = get_logger("Security.SensitiveFilter")

CHINESE_SURNAMES: set[str] = {
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "马", "朱", "胡", "郭", "林", "何", "高", "梁",
    "郑", "罗", "宋", "谢", "唐", "韩", "曹", "许", "邓", "萧",
    "冯", "曾", "程", "蔡", "彭", "潘", "袁", "於", "董", "余",
    "苏", "叶", "吕", "魏", "蒋", "田", "杜", "丁", "沈", "姜",
    "范", "江", "傅", "钟", "汪", "廖", "章", "念", "万", "顾",
    "毛", "赖", "武", "康", "贺", "严", "尹", "钱", "施",
    "牛", "洪", "龚", "韦", "夹谷", "司马", "上官", "欧阳", "夏侯", "诸葛",
    "闻人", "东方", "赫连", "皇甫", "尉迟", "公羊", "澹台", "公冶", "宗政",
    "濮阳", "淳于", "单于", "太叔", "申屠", "公孙", "仲孙", "轩辕", "令狐",
}

_ID_PATTERN_15 = re.compile(r"(?<!\d)[1-9]\d{5}\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}(?!\d)")
_ID_PATTERN_18 = re.compile(r"(?<!\d)[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")
_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

_ADDRESS_KEYWORDS = ["省", "市", "区", "县", "路", "街", "道", "巷", "弄", "号", "栋", "楼", "室", "村", "镇", "乡"]
_ADDRESS_PATTERN = re.compile(r"[^\s]{2,6}(省|市|区|县)[^\s]{0,20}?(路|街|道|巷|弄|号|栋|楼|室)")

_VEHICLE_PROVINCE_CODES = [
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
    "琼", "渝", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "港", "澳", "台"
]
_VEHICLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[" + "".join(_VEHICLE_PROVINCE_CODES) + r"][A-Z][A-Z0-9]{5}(?![A-Za-z0-9])"
)

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


def _is_chinese_surname(char: str) -> bool:
    """判断字符是否为常见中文姓氏。

    Args:
        char: 待检测的字符。

    Returns:
        是否为常见中文姓氏。
    """
    return char in CHINESE_SURNAMES


def _mask_name(text: str) -> str:
    """掩码可能的中文姓名。

    Args:
        text: 原始文本。

    Returns:
        掩码后的文本。
    """
    result = text
    for surname in CHINESE_SURNAMES:
        if len(surname) == 1:
            pattern = re.compile(rf"{re.escape(surname)}[\u4e00-\u9fa5]{{1,2}}")
        else:
            pattern = re.compile(rf"{re.escape(surname)}[\u4e00-\u9fa5]?")
        result = pattern.sub("[NAME-MASKED]", result)
    return result


def mask_pii(text: str) -> str:
    """掩码文本中的个人身份信息（PII）。

    Args:
        text: 用户输入的原始文本。

    Returns:
        已掩码处理后的文本。
    """
    if not text:
        return text

    result = text

    result = _ID_PATTERN_15.sub("[ID-MASKED]", result)
    result = _ID_PATTERN_18.sub("[ID-MASKED]", result)

    result = _PHONE_PATTERN.sub("[PHONE-MASKED]", result)

    result = _mask_name(result)

    for keyword in _ADDRESS_KEYWORDS:
        if keyword in result and len(result) > 10:
            result = _ADDRESS_PATTERN.sub("[ADDR-MASKED]", result)
            break

    result = _VEHICLE_PATTERN.sub("[VEHICLE-MASKED]", result)

    return result


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
    """清理用户输入并返回 PII 掩码后的文本。

    高风险检测与清理相互独立；调用方需要风险分类时应额外调用
    ``detect_high_risk``。

    Args:
        text: 用户输入的原始文本。

    Returns:
        PII 已掩码处理后的文本。
    """
    if not text:
        return text

    sanitized = mask_pii(text)

    _logger.info("【sanitize_input】输入已清理 | 原始长度: %d | 清理后长度: %d", len(text), len(sanitized))

    return sanitized
