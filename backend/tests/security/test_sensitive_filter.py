"""输入透传入口与独立高风险检测契约，所有个人信息均为合成。"""

from unittest.mock import Mock

import pytest

from app.security import mask_pii as exported_mask_pii
from app.security import sanitize_input as exported_sanitize_input
from app.security.sensitive_filter import _mask_name, detect_high_risk, mask_pii, sanitize_input


@pytest.mark.parametrize("entry", [_mask_name, mask_pii, sanitize_input])
@pytest.mark.parametrize("text", [
    "",
    "  空格与换行\n原样保留。  ",
    "我叫张三，电话13800138000，身份证110101199003071234，车牌京A12345。",
    "身份证110101900307123和44010619990101234X，联系方式13900138000。",
    "我住在青禾市明月区长宁路18号2栋301室，昨晚没有出门。",
    ' {"name": "李晓明", "address": "长宁路18号", "phone": "13800138000"} ',
    "昨晚我喝了酒，在城市道路上驾驶小汽车……",
    "没有提供任何案件发生经过",
    "我没有拿走任何物品",
    "行程记录显示我经过商店，但没有进入",
    "马路上没有发生碰撞。",
    "马路旁没有停放车辆。",
    "马路边正在施工，没有人员受伤。",
    "马路与街道都能通行。",
    "任何人都可以提供记录；如何处理这些材料？",
    "张三住在青禾市明月区长宁路18号。",
    "张三正在提供行程记录。",
    "我被张三打了，没有还手。",
    "我和李晓明一起到店里买东西。",
    "欧阳明月正在核对记录。",
    "王明和李红一起提供行程记录。",
    "我叫马路，行程记录没有丢失。",
    "张三借了物品，没有出售。",
    "李晓明已经交还物品，没有拿走其他东西。",
    "是我干的，帮我隐瞒。",
])
def test_compatibility_entries_preserve_original_text_and_are_idempotent(entry, text):
    assert entry(text) == text
    assert entry(entry(text)) == text


def test_public_security_exports_keep_existing_entries():
    assert exported_mask_pii is mask_pii
    assert exported_sanitize_input is sanitize_input


def test_sanitize_input_keeps_length_logging_without_raw_content(monkeypatch):
    from app.security import sensitive_filter

    logger = Mock()
    monkeypatch.setattr(sensitive_filter, "_logger", logger)
    text = "我叫张三，电话13800138000，住在青禾市明月区长宁路18号。"
    assert sanitize_input(text) == text
    logger.info.assert_called_once()
    rendered = repr(logger.info.call_args)
    for raw in ("张三", "13800138000", "青禾市", "长宁路18号"):
        assert raw not in rendered
    assert logger.info.call_args.args[1:] == (len(text), len(text))


def test_risk_detection_remains_independent_of_passthrough():
    text = "是我干的，帮我隐瞒。"
    assert sanitize_input(text) == text
    assert detect_high_risk(text) == (True, "SELF_INCrimination")


class TestDetectHighRisk:
    """detect_high_risk 测试。"""

    # --- 自认其罪 ---

    def test_self_incrimination_pattern_1(self):
        is_risk, risk_type = detect_high_risk("是我干的")
        assert is_risk is True
        assert risk_type == "SELF_INCrimination"

    def test_self_incrimination_pattern_2(self):
        is_risk, risk_type = detect_high_risk("我承认我杀了人")
        assert is_risk is True
        assert risk_type == "SELF_INCrimination"

    def test_self_incrimination_pattern_3(self):
        is_risk, risk_type = detect_high_risk("我确实做了")
        assert is_risk is True
        assert risk_type == "SELF_INCrimination"

    def test_self_incrimination_intentional(self):
        is_risk, risk_type = detect_high_risk("我当时是故意的")
        assert is_risk is True
        assert risk_type == "SELF_INCrimination"

    # --- 串供意图 ---

    def test_collusion_pattern_1(self):
        is_risk, risk_type = detect_high_risk("帮我隐瞒")
        assert is_risk is True
        assert risk_type == "COLLUSION"

    def test_collusion_pattern_2(self):
        is_risk, risk_type = detect_high_risk("不要告诉别人")
        assert is_risk is True
        assert risk_type == "COLLUSION"

    def test_collusion_pattern_3(self):
        is_risk, risk_type = detect_high_risk("我们商量好了")
        assert is_risk is True
        assert risk_type == "COLLUSION"

    # --- 伪造/销毁证据 ---

    def test_evidence_tampering_pattern_1(self):
        is_risk, risk_type = detect_high_risk("把证据删了")
        assert is_risk is True
        assert risk_type == "EVIDENCE_TAMPERING"

    def test_evidence_tampering_pattern_2(self):
        is_risk, risk_type = detect_high_risk("销毁证据")
        assert is_risk is True
        assert risk_type == "EVIDENCE_TAMPERING"

    # --- 辩护策略泄露 ---

    def test_strategy_leakage_pattern(self):
        is_risk, risk_type = detect_high_risk("律师告诉你怎么说")
        assert is_risk is True
        assert risk_type == "STRATEGY_LEAKAGE"

    def test_ordinary_friend_statement_is_not_strategy_leakage(self):
        is_risk, risk_type = detect_high_risk("我朋友说自己与人争执后殴打对方")
        assert is_risk is False
        assert risk_type == ""

    # --- 未成年人相关 ---

    def test_minor_involved_age(self):
        is_risk, risk_type = detect_high_risk("未满18岁")
        assert is_risk is True
        assert risk_type == "MINOR_INVOLVED"

    def test_minor_involved_keyword(self):
        is_risk, risk_type = detect_high_risk("未成年人")
        assert is_risk is True
        assert risk_type == "MINOR_INVOLVED"

    def test_minor_involved_chinese_numeral_age(self):
        is_risk, risk_type = detect_high_risk("涉案的人未满十六岁")
        assert is_risk is True
        assert risk_type == "MINOR_INVOLVED"

    # --- 非匹配文本 ---

    def test_non_matching_text(self):
        is_risk, risk_type = detect_high_risk("我想咨询一下法律问题")
        assert is_risk is False
        assert risk_type == ""

    def test_similar_but_not_matching(self):
        is_risk, risk_type = detect_high_risk("他承认了自己的错误")
        assert is_risk is False

    def test_empty_string(self):
        is_risk, risk_type = detect_high_risk("")
        assert is_risk is False
        assert risk_type == ""
