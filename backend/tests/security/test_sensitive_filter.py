"""sensitive_filter 模块纯函数单元测试。"""

import pytest

from app.security.sensitive_filter import (
    CHINESE_SURNAMES,
    _is_chinese_surname,
    _mask_name,
    detect_high_risk,
    mask_pii,
    sanitize_input,
)


# ──────────────────────────── _is_chinese_surname ────────────────────────────


class TestIsChineseSurname:
    """_is_chinese_surname 测试。"""

    def test_common_single_char_surnames(self):
        for name in ("王", "李", "张", "刘", "陈", "赵", "周", "吴"):
            assert _is_chinese_surname(name) is True

    def test_compound_surnames(self):
        for name in ("欧阳", "司马", "上官", "诸葛", "令狐"):
            assert _is_chinese_surname(name) is True

    def test_non_surname_chars(self):
        for char in ("大", "小", "中", "风", "雨", "山", "水"):
            assert _is_chinese_surname(char) is False

    def test_empty_string(self):
        assert _is_chinese_surname("") is False

    def test_non_chinese_char(self):
        assert _is_chinese_surname("A") is False
        assert _is_chinese_surname("1") is False


# ──────────────────────────── _mask_name ─────────────────────────────────────


class TestMaskName:
    """_mask_name 测试。"""

    def test_single_surname_with_one_char_name(self):
        result = _mask_name("我叫王明")
        assert "[NAME-MASKED]" in result
        assert "王明" not in result

    def test_single_surname_with_two_char_name(self):
        result = _mask_name("李晓明来了")
        assert "[NAME-MASKED]" in result
        assert "李晓明" not in result

    def test_compound_surname_masking(self):
        result = _mask_name("欧阳锋是高手")
        assert "[NAME-MASKED]" in result
        assert "欧阳锋" not in result

    def test_multiple_names_in_text(self):
        result = _mask_name("王明和李红一起")
        assert result.count("[NAME-MASKED]") == 2

    def test_no_name_in_text(self):
        text = "今天天气很好"
        assert _mask_name(text) == text

    def test_empty_string(self):
        assert _mask_name("") == ""


# ──────────────────────────── mask_pii ───────────────────────────────────────


class TestMaskPii:
    """mask_pii 测试。"""

    # --- 身份证号 ---

    def test_id_18_digits(self):
        text = "我的身份证号是110101199003071234"
        result = mask_pii(text)
        assert "[ID-MASKED]" in result
        assert "110101199003071234" not in result

    def test_id_15_digits(self):
        text = "身份证号110101900307123"
        result = mask_pii(text)
        assert "[ID-MASKED]" in result
        assert "110101900307123" not in result

    def test_id_18_with_x(self):
        text = "身份证号44010619990101234X"
        result = mask_pii(text)
        assert "[ID-MASKED]" in result
        assert "44010619990101234X" not in result

    # --- 手机号 ---

    def test_phone_11_digits(self):
        text = "我的手机号是13812345678"
        result = mask_pii(text)
        assert "[PHONE-MASKED]" in result
        assert "13812345678" not in result

    def test_phone_various_prefixes(self):
        for prefix in ("13", "15", "17", "18", "19"):
            phone = f"{prefix}012345678"
            result = mask_pii(f"电话{phone}")
            assert "[PHONE-MASKED]" in result

    # --- 中文姓名 ---

    def test_chinese_name_masked(self):
        text = "我叫张三"
        result = mask_pii(text)
        assert "[NAME-MASKED]" in result
        assert "张三" not in result

    # --- 地址 ---

    def test_address_with_lu(self):
        text = "我住在北京市朝阳区建国路100号"
        result = mask_pii(text)
        assert "[ADDR-MASKED]" in result

    def test_address_with_jie(self):
        text = "上海市浦东新区南京路88号"
        result = mask_pii(text)
        assert "[ADDR-MASKED]" in result

    def test_address_with_hao(self):
        text = "广州市天河区天河路12号"
        result = mask_pii(text)
        assert "[ADDR-MASKED]" in result

    def test_short_text_no_address_mask(self):
        """文本长度 <=10 时即使包含地址关键词也不做地址掩码。"""
        text = "北京路"
        result = mask_pii(text)
        assert "[ADDR-MASKED]" not in result

    # --- 车牌号 ---

    def test_vehicle_plate(self):
        text = "车牌号京A12345"
        result = mask_pii(text)
        assert "[VEHICLE-MASKED]" in result
        assert "京A12345" not in result

    def test_vehicle_plate_various_provinces(self):
        for plate in ("沪B67890", "粤C11111", "川D22222"):
            result = mask_pii(f"车牌{plate}")
            assert "[VEHICLE-MASKED]" in result

    # --- 边界情况 ---

    def test_empty_string(self):
        assert mask_pii("") == ""

    def test_no_pii(self):
        text = "今天天气很好"
        assert mask_pii(text) == text

    def test_multiple_pii_types(self):
        text = "我叫王明，手机号13812345678，身份证110101199003071234"
        result = mask_pii(text)
        assert "[NAME-MASKED]" in result
        assert "[PHONE-MASKED]" in result
        assert "[ID-MASKED]" in result

    def test_pii_at_start(self):
        result = mask_pii("13812345678是我的手机号")
        assert "[PHONE-MASKED]" in result

    def test_pii_at_middle(self):
        result = mask_pii("联系13812345678即可")
        assert "[PHONE-MASKED]" in result

    def test_pii_at_end(self):
        result = mask_pii("我的手机号是13812345678")
        assert "[PHONE-MASKED]" in result


# ──────────────────────────── detect_high_risk ───────────────────────────────


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

    # --- 未成年人相关 ---

    def test_minor_involved_age(self):
        is_risk, risk_type = detect_high_risk("未满18岁")
        assert is_risk is True
        assert risk_type == "MINOR_INVOLVED"

    def test_minor_involved_keyword(self):
        is_risk, risk_type = detect_high_risk("未成年人")
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


# ──────────────────────────── sanitize_input ─────────────────────────────────


class TestSanitizeInput:
    """sanitize_input 测试。"""

    def test_masks_pii(self):
        text = "我叫王明，手机号13812345678"
        result = sanitize_input(text)
        assert "[NAME-MASKED]" in result
        assert "[PHONE-MASKED]" in result

    def test_empty_string(self):
        assert sanitize_input("") == ""

    def test_no_pii_returns_same(self):
        text = "我想咨询法律问题"
        assert sanitize_input(text) == text

    def test_returns_masked_text(self):
        """sanitize_input 返回的是 PII 掩码后的文本。"""
        text = "身份证号110101199003071234"
        result = sanitize_input(text)
        assert "[ID-MASKED]" in result
        assert "110101199003071234" not in result
