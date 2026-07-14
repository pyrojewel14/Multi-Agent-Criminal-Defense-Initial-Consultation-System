"""Test data factory functions.

Each factory returns a dict that matches the real application data structures,
with sensible defaults that can be selectively overridden via ``**overrides``.
"""

import uuid
from typing import Any, Dict

from app.state.consultation_state import ConsultationState, validate_consultation_state


def make_consultation_state(**overrides: Any) -> ConsultationState:
    """Build a ``ConsultationState`` dict with defaults, merged with overrides.

    The keys and types mirror ``app.state.consultation_state.ConsultationState``.
    """
    defaults = {
        "consultation_id": str(uuid.uuid4()),
        "user_id": "user-001",
        "session_id": "session-001",
        "user_type": "suspect",
        "consent_given": False,
        "facts_raw": [""],
        "facts_structured": {},
        "applied_laws": [],
        "current_agent": "Receptionist",
        "pending_questions": [],
        "alert_triggered": False,
        "risk_assessment": None,
        "lawyer_review_needed": False,
        "final_output": "",
        "conversation_history": [],
        "report_draft": None,
        "service_plan": None,
        "lawyer_id": None,
        "current_input": None,
        "facts_coverage_rate": None,
        "fact_law_loop_count": 0,
        "element_to_law_mapping": None,
        "identity_info": None,
        "user_role": None,
        "awaiting_lawyer_review": False,
        "lawyer_decision": None,
        "lawyer_feedback": None,
        "rag_only": False,
    }
    return validate_consultation_state({**defaults, **overrides})


def make_law_data() -> Dict[str, Any]:
    """Build sample criminal law JSON data with 3 articles.

    Each article contains: article_number, title, content, elements,
    base_sentence, charge_tags.
    """
    return {
        "articles": [
            {
                "article_number": "第二百六十四条",
                "title": "盗窃罪",
                "content": "盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。",
                "elements": [
                    "客体要件：公私财物所有权",
                    "客观要件：窃取数额较大的公私财物或多次盗窃",
                    "主体要件：一般主体，即达到法定刑事责任年龄的自然人",
                    "主观要件：直接故意，且具有非法占有的目的",
                ],
                "base_sentence": "三年以下有期徒刑、拘役或者管制",
                "charge_tags": ["财产犯罪", "盗窃"],
            },
            {
                "article_number": "第二百三十四条",
                "title": "故意伤害罪",
                "content": "故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。犯前款罪，致人重伤的，处三年以上十年以下有期徒刑；致人死亡或者以特别残忍手段致人重伤造成严重残疾的，处十年以上有期徒刑、无期徒刑或者死刑。",
                "elements": [
                    "客体要件：他人的身体健康权",
                    "客观要件：非法损害他人身体健康的行为",
                    "主体要件：一般主体",
                    "主观要件：故意",
                ],
                "base_sentence": "三年以下有期徒刑、拘役或者管制",
                "charge_tags": ["侵犯人身权利犯罪", "故意伤害"],
            },
            {
                "article_number": "第二百六十六条",
                "title": "诈骗罪",
                "content": "诈骗公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。",
                "elements": [
                    "客体要件：公私财物所有权",
                    "客观要件：使用欺骗方法骗取数额较大的公私财物",
                    "主体要件：一般主体",
                    "主观要件：直接故意，且具有非法占有的目的",
                ],
                "base_sentence": "三年以下有期徒刑、拘役或者管制",
                "charge_tags": ["财产犯罪", "诈骗"],
            },
        ]
    }


def make_risk_assessment(**overrides) -> Dict[str, Any]:
    """Build a risk assessment dict matching the actual schema.

    Keys: predicted_sentence_range, mitigating_factors, aggravating_factors,
    compulsory_measure_risk, evidence_risk_points, procedure_risks.
    """
    defaults = {
        "predicted_sentence_range": "三年以下有期徒刑、拘役或者管制",
        "mitigating_factors": [
            "初犯、偶犯",
            "自愿认罪认罚",
            "积极赔偿被害人损失并取得谅解",
        ],
        "aggravating_factors": [
            "涉案金额较大",
        ],
        "compulsory_measure_risk": {
            "arrest_likelihood": "中等",
            "bail_feasibility": "符合取保候审条件",
        },
        "evidence_risk_points": [
            "部分证据链尚不完整",
            "证人证言存在矛盾",
        ],
        "procedure_risks": [
            "侦查阶段可能延长羁押期限",
        ],
    }
    return {**defaults, **overrides}


def make_user_dict(**overrides) -> Dict[str, Any]:
    """Build a user info dict (user_id, role, username)."""
    defaults = {
        "user_id": "user-001",
        "role": "client",
        "username": "testuser",
    }
    return {**defaults, **overrides}


def make_applied_law(**overrides) -> Dict[str, Any]:
    """Build an applied_law dict matching the structure stored in ConsultationState.

    Keys: article_number, charge_name, charge_tags, elements, base_sentence,
    data_source.
    """
    defaults = {
        "article_number": "第二百六十四条",
        "charge_name": "盗窃罪",
        "charge_tags": ["财产犯罪", "盗窃"],
        "elements": [
            "客体要件：公私财物所有权",
            "客观要件：窃取数额较大的公私财物或多次盗窃",
            "主体要件：一般主体",
            "主观要件：直接故意，且具有非法占有的目的",
        ],
        "base_sentence": "三年以下有期徒刑、拘役或者管制",
        "data_source": "刑法",
    }
    return {**defaults, **overrides}
