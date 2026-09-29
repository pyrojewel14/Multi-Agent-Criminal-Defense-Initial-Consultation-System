from app.infrastructure.logging import get_logger


class DisclaimerService:
    """注入法律免责声明前缀的服务。"""

    DISCLAIMER_PREFIX = "本内容为智能辅助生成，仅供参考，待律师确认后生效。\n\n"

    def __init__(self):
        """初始化免责声明服务。"""
        self._logger = get_logger("Security.Disclaimer")

    def inject(self, content: str) -> str:
        """确保免责声明前缀存在于内容开头。

        如果前缀已存在，内容保持不变（幂等性）。
        否则，在内容前添加前缀。

        Args:
            content: 原始代理输出文本。

        Returns:
            确保免责声明前缀在开头的内容。
        """
        if content.startswith(self.DISCLAIMER_PREFIX):
            self._logger.debug("【inject】免责声明已存在，跳过注入")
            return content
        self._logger.debug("【inject】注入免责声明前缀")
        return self.DISCLAIMER_PREFIX + content


disclaimer = DisclaimerService()
DISCLAIMER_PREFIX = DisclaimerService.DISCLAIMER_PREFIX
