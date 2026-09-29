"""结构化案件事实的字段读取规则。"""

from typing import Any, Dict

_FACT_KEY_MAPPING = {
    "time": "incident_time",
    "location": "incident_location",
    "parties": "parties",
    "behavior": "behavior_sequence",
    "consequence": "consequence",
    "evidence": "evidence_mentioned",
    "arrest": "arrest_status",
    "surrender": "surrender",
    "forgiveness": "victim_forgiveness",
    "record": "prior_record",
}


def get_fact_value(facts_structured: Dict[str, Any], key: str) -> Any:
    """按标准事实键或简写键读取结构化事实。"""
    return facts_structured.get(_FACT_KEY_MAPPING.get(key, key))


def is_mapped_fact_key(key: str) -> bool:
    """判断要件键是否属于已登记的结构化事实字段。"""
    return key in _FACT_KEY_MAPPING
