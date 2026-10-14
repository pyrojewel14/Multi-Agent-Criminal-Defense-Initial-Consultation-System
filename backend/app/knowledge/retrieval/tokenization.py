"""供所有词法召回共用的中文切分协议。"""

import re

TOKENIZER_VERSION = "cjk-char-bigram-v1"


def chinese_tokens(text: str) -> list[str]:
    """连续中文拆成单字与相邻双字；英文数字按词，不使用案例词表。"""
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.extend(part)
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens
