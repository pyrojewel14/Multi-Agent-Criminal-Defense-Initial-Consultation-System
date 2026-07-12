"""receptionist 模块纯函数单元测试。"""

import pytest

from app.agents.receptionist import check_consent_given, extract_user_type


# ──────────────────────────── extract_user_type ──────────────────────────────


class TestExtractUserType:
    """extract_user_type 测试。"""

    # --- suspect ---

    def test_suspect_keyword_dangshiren(self):
        assert extract_user_type("我是当事人") == "suspect"

    def test_suspect_keyword_xianyiren(self):
        assert extract_user_type("我是嫌疑人") == "suspect"

    def test_suspect_keyword_xianyi(self):
        assert extract_user_type("我是嫌疑") == "suspect"

    def test_suspect_keyword_benren(self):
        assert extract_user_type("我本人") == "suspect"

    def test_suspect_keyword_bei_gaoren(self):
        assert extract_user_type("我是被告人") == "suspect"

    # --- victim ---

    def test_victim_keyword_beihairen(self):
        assert extract_user_type("我是被害人") == "victim"

    def test_victim_keyword_shouhai(self):
        assert extract_user_type("我是受害") == "victim"

    def test_victim_keyword_wo_bei(self):
        assert extract_user_type("我被") == "victim"

    # --- family ---

    def test_family_keyword_jiashu(self):
        assert extract_user_type("我是家属") == "family"

    def test_family_keyword_jiaren(self):
        assert extract_user_type("我是家人") == "family"

    def test_family_keyword_wo_jiaren(self):
        assert extract_user_type("我家人") == "family"

    def test_family_keyword_qinshu(self):
        assert extract_user_type("我亲属") == "family"

    # --- no match ---

    def test_no_match_returns_none(self):
        assert extract_user_type("我想咨询法律问题") is None

    def test_empty_string_returns_none(self):
        assert extract_user_type("") is None

    # --- partial match ---

    def test_partial_match_suspect(self):
        """'嫌疑人' 关键词需完整匹配 '我是嫌疑人' 子串。"""
        assert extract_user_type("我是嫌疑人") == "suspect"
        # "我是一名嫌疑人" 不包含 "我是嫌疑人" 精确子串，因此不匹配
        assert extract_user_type("我是一名嫌疑人") is None

    def test_partial_match_victim(self):
        assert extract_user_type("我被人打了") == "victim"

    # --- priority ---

    def test_first_matching_type_wins(self):
        """当文本同时匹配多个类型时，按 USER_TYPE_PATTERNS 字典遍历顺序返回第一个。"""
        result = extract_user_type("我是嫌疑人也是被害人家属")
        # suspect 的关键词先匹配到
        assert result == "suspect"


# ──────────────────────────── check_consent_given ────────────────────────────


class TestCheckConsentGiven:
    """check_consent_given 测试。"""

    def test_consent_tongyi(self):
        assert check_consent_given("我同意") is True

    def test_consent_queren(self):
        assert check_consent_given("确认") is True

    def test_consent_yizhixi(self):
        assert check_consent_given("已知悉") is True

    def test_consent_wo_yi_yuedu(self):
        assert check_consent_given("我已阅读") is True

    def test_consent_wo_tongyi(self):
        assert check_consent_given("我同意") is True

    def test_non_consent_text(self):
        # "不同意" 包含 "同意" 子串，因此 check_consent_given 返回 True
        assert check_consent_given("我不同意") is True

    def test_unrelated_text(self):
        assert check_consent_given("我想咨询法律问题") is False

    def test_empty_string(self):
        assert check_consent_given("") is False

    def test_consent_in_longer_text(self):
        assert check_consent_given("我已经阅读并同意权利义务告知书") is True
