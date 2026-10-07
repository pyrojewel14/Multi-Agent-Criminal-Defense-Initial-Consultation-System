"""单案字段记忆：用户陈述与冲突留痕，不代表律师确认。"""

import json
from copy import deepcopy
from datetime import datetime, timezone

LIST_FIELDS = {"parties", "behavior_sequence", "evidence_mentioned"}
FIELDS = ("incident_time", "incident_location", "parties", "behavior_sequence", "consequence",
          "evidence_mentioned", "arrest_status", "surrender", "victim_forgiveness", "prior_record")
REDACTED_VALUES = {"[NAME-MASKED]", "[ID-MASKED]", "[PHONE-MASKED]", "[ADDR-MASKED]", "[VEHICLE-MASKED]"}


def _key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sources(entry, source_id):
    if source_id not in entry["source_ids"]:
        entry["source_ids"].append(source_id)


def _identity(field, value):
    if not isinstance(value, dict):
        return None
    keys = {"parties": ("id", "name"), "behavior_sequence": ("id", "event_id"),
            "evidence_mentioned": ("id", "evidence_id", "name")}[field]
    return next(((key, value[key]) for key in keys if value.get(key)), None)


def _merge_version(entry, value, source_id, now, status, correction):
    """同一字段或具名列表实体的更正和冲突共用留痕规则。"""
    if entry["value"] == value:
        _sources(entry, source_id)
        entry["updated_at"] = now
    elif correction:
        old = {key: deepcopy(entry[key]) for key in ("value", "source_ids", "updated_at", "status")}
        alternatives = [old] + [alt for alt in entry.get("alternatives", []) if alt["value"] != value]
        entry.update(value=value, source_ids=[source_id], updated_at=now,
                     status="user_corrected_unverified", alternatives=alternatives)
    else:
        alternatives = entry.setdefault("alternatives", [])
        alternative = next((alt for alt in alternatives if alt["value"] == value), None)
        if alternative:
            _sources(alternative, source_id)
            alternative["updated_at"] = now
        else:
            alternatives.append({"value": value, "source_ids": [source_id], "updated_at": now, "status": status})
        entry["status"] = "conflicted"


def merge_case_memory(memory, candidate, source_id, *, correction=False):
    """增量合并合法候选；空值不删除，显式更正仍保留旧版本与来源。"""
    result = deepcopy(memory or {"version": 1, "fields": {}})
    fields = result.setdefault("fields", {})
    now = datetime.now(timezone.utc).isoformat()
    status = "legacy_unverified" if source_id == "legacy" else "user_claim_unverified"
    for name in FIELDS:
        value = candidate.get(name)
        if value is None or value == "" or value == [] or value == {}:
            continue
        # 兼容历史脱敏数据：纯占位符没有可合并的值，不猜测原值或制造来源冲突。
        if isinstance(value, str) and value.strip() in REDACTED_VALUES:
            continue
        entry = fields.get(name)
        if name in LIST_FIELDS:
            entry = fields.setdefault(name, {"items": [], "status": status})
            for item in value:
                if not item:
                    continue
                existing = next((old for old in entry["items"] if _key(old["value"]) == _key(item)), None)
                identity = _identity(name, item)
                if existing is None and identity is not None:
                    existing = next((old for old in entry["items"] if _identity(name, old["value"]) == identity), None)
                if existing:
                    _merge_version(existing, item, source_id, now, status, correction)
                else:
                    entry["items"].append({"value": item, "source_ids": [source_id], "updated_at": now,
                                           "status": status})
            entry["updated_at"] = now
        elif entry is None:
            fields[name] = {"value": value, "source_ids": [source_id], "updated_at": now,
                            "status": status, "alternatives": []}
        else:
            _merge_version(entry, value, source_id, now, status, correction)
    return result


def project_facts(memory):
    """投影兼容事实字段；行为按最近陈述优先，保留原事件时间与来源。"""
    result = {name: [] if name in LIST_FIELDS else None for name in FIELDS}
    for name, entry in memory.get("fields", {}).items():
        if name in FIELDS:
            if name in LIST_FIELDS:
                items = entry["items"]
                if name == "behavior_sequence":
                    items = sorted(items, key=lambda item: item["updated_at"], reverse=True)
                result[name] = [item["value"] for item in items]
            else:
                result[name] = entry["value"]
    return result
