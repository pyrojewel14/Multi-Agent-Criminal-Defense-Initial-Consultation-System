import pytest

from app.security.disclaimer import DisclaimerService, DISCLAIMER_PREFIX, disclaimer


class TestDisclaimerInject:
    """Tests for DisclaimerService.inject."""

    def test_adds_disclaimer_to_content_without_one(self):
        content = "这是原始内容"
        result = disclaimer.inject(content)
        assert result.startswith(DISCLAIMER_PREFIX)
        assert "这是原始内容" in result

    def test_inject_prepends_prefix(self):
        content = "正文"
        result = disclaimer.inject(content)
        assert result == DISCLAIMER_PREFIX + "正文"

    def test_idempotency_calling_twice_does_not_duplicate(self):
        content = "原始内容"
        result1 = disclaimer.inject(content)
        result2 = disclaimer.inject(result1)
        assert result1 == result2
        # Verify only one prefix exists
        assert result2.count(DISCLAIMER_PREFIX) == 1

    def test_content_already_has_prefix_is_unchanged(self):
        pre_injected = DISCLAIMER_PREFIX + "已有声明的内容"
        result = disclaimer.inject(pre_injected)
        assert result == pre_injected

    def test_empty_content_gets_prefix(self):
        result = disclaimer.inject("")
        assert result == DISCLAIMER_PREFIX

    def test_disclaimer_service_instance_inject(self):
        """Test using DisclaimerService class directly."""
        service = DisclaimerService()
        content = "测试内容"
        result = service.inject(content)
        assert result.startswith(DisclaimerService.DISCLAIMER_PREFIX)
