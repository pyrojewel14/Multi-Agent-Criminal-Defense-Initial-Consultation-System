"""提取并归一化刑法条号，保持母条与增补条的独立身份。"""

import re

_CN_NUM_MAP = {
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_CN_DIGITS = set("零一二三四五六七八九十百千")


def _cn_to_arabic(cn: str) -> str:
    """将中文数字转换为阿拉伯数字。"""
    if not cn or not all(char in _CN_DIGITS for char in cn):
        return cn
    result = 0
    current = 0
    for char in cn:
        if char in _CN_NUM_MAP:
            current = int(_CN_NUM_MAP[char])
        elif char == "十":
            result += (current or 1) * 10
            current = 0
        elif char == "百":
            result += current * 100
            current = 0
        elif char == "千":
            result += current * 1000
            current = 0
    return str(result + current)


def _normalize_article_number(article_number: str) -> str:
    """统一中文数字和阿拉伯数字法条编号的表示。"""
    if not article_number:
        return ""
    match = re.match(r"第(.+?)条(之[一二三四五六七八九十百千零\d]+)?", article_number)
    if not match:
        return article_number.strip()
    number, suffix = match.group(1), match.group(2) or ""
    if suffix:
        suffix = "之" + ({str(i): v for i, v in enumerate("零一二三四五六七八九十")}.get(suffix[1:], suffix[1:]))
    if number.isdigit():
        return f"第{number}条{suffix}"
    arabic = _cn_to_arabic(number)
    return f"第{arabic}条{suffix}" if arabic != number else article_number.strip()


def _extract_article_number_from_text(text: str | None) -> str:
    """从法条文本内容中提取法条编号。

    Args:
        text: 法条文本内容，允许为空

    Returns:
        提取到的法条编号，未找到则返回空字符串
    """
    if not text:
        return ""
    # 匹配 "第X条" 格式（X 为中文数字或阿拉伯数字）
    match = re.search(r"第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十百千零\d]+)?", text)
    return match.group(0) if match else ""


normalize_article_number = _normalize_article_number
extract_article_number = _extract_article_number_from_text
