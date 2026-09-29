import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_MODULE_LEVELS: dict[str, int] = {}
_raw_module_levels = os.getenv("LOG_LEVEL_MODULES", "")
if _raw_module_levels:
    for item in _raw_module_levels.split(","):
        item = item.strip()
        if "=" in item:
            name, level_str = item.split("=", 1)
            level_val = getattr(logging, level_str.strip().upper(), None)
            if level_val is not None:
                _MODULE_LEVELS[name.strip()] = level_val

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{15}(\d{2}[\dXx])?"), "[ID-MASKED]"),
    (re.compile(r"1[3-9]\d{9}"), "[PHONE-MASKED]"),
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL-MASKED]"),
    (re.compile(r"\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]"), "[ID-MASKED]"),
]


class SensitiveFilter(logging.Filter):
    """日志过滤器，对消息中的 PII（身份证、手机号、邮箱等）进行脱敏。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """对日志消息应用所有 PII 正则替换。

        Args:
            record: 日志记录对象。

        Returns:
            是否允许该日志记录通过。
        """
        msg = record.getMessage()
        for pattern, replacement in _PII_PATTERNS:
            msg = pattern.sub(replacement, msg)
        record.msg = msg
        record.args = None
        return True


_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "consultation.log",
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
_file_handler.suffix = "%Y%m%d"
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
_file_handler.addFilter(SensitiveFilter())

_console_handler = logging.StreamHandler()
_console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
_console_handler.addFilter(SensitiveFilter())


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """创建或获取一个配置好的 Logger（带文件和控制台处理器）。

    Logger 按名称缓存（标准库行为）。所有 Handler 共享，
    每次记录都会经过 PII 过滤器。

    Args:
        name: Logger 名称，通常使用模块名或类名。
        level: 可选的日志级别覆盖。
               优先级：level 参数 > LOG_LEVEL_MODULES 环境变量 > LOG_LEVEL 环境变量 > INFO。

    Returns:
        配置好的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    resolved = level or _MODULE_LEVELS.get(name) or getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(resolved)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)

    return logger
