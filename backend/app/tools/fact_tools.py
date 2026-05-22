from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


@tool
def extract_case_facts(
    incident_time: Optional[str] = None,
    incident_location: Optional[str] = None,
    parties: Optional[List[Dict[str, Any]]] = None,
    behavior_sequence: Optional[List[Dict[str, Any]]] = None,
    consequence: Optional[str] = None,
    evidence_mentioned: Optional[List[Dict[str, Any]]] = None,
    arrest_status: Optional[str] = None,
    surrender: Optional[bool] = None,
    victim_forgiveness: Optional[bool] = None,
    prior_record: Optional[bool] = None,
) -> Dict[str, Any]:
    """从咨询者描述中提取刑事案件关键事实要素。

    Args:
        incident_time: 事件发生时间，格式：YYYY-MM-DD 或 相对时间
        incident_location: 事件发生地点（已脱敏）
        parties: 当事人列表
        behavior_sequence: 行为时间序列
        consequence: 后果描述
        evidence_mentioned: 提到的证据线索
        arrest_status: 当前羁押状态
        surrender: 是否自首
        victim_forgiveness: 被害人是否谅解
        prior_record: 是否有前科劣迹

    Returns:
        包含所有提取字段的字典
    """
    return {
        "incident_time": incident_time,
        "incident_location": incident_location,
        "parties": parties or [],
        "behavior_sequence": behavior_sequence or [],
        "consequence": consequence,
        "evidence_mentioned": evidence_mentioned or [],
        "arrest_status": arrest_status,
        "surrender": surrender,
        "victim_forgiveness": victim_forgiveness,
        "prior_record": prior_record,
    }