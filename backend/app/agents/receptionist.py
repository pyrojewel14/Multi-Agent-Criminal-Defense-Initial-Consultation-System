from typing import Optional

from app.security.disclaimer import disclaimer
from app.state.consultation_state import ConsultationState
from app.utils.llm_gateway import llm_gateway
from app.utils.logger import get_logger
from app.utils.prompt_loader import prompt_loader

_logger = get_logger("Agent.Receptionist")

CONSENT_KEYWORDS = ["同意", "确认", "已知悉", "我已阅读", "我同意"]
USER_TYPE_PATTERNS = {
    "suspect": ["我是当事人", "我是嫌疑人", "我是嫌疑", "我本人", "我是被告人"],
    "victim": ["我是被害人", "我是受害", "我被"],
    "family": ["我是家属", "我是家人", "我家人", "我亲属"],
}


def extract_user_type(user_message: str) -> Optional[str]:
    """从用户消息中提取身份类型。

    Args:
        user_message: 用户输入的消息。

    Returns:
        身份类型字符串（suspect/victim/family）或 None。
    """
    for user_type, keywords in USER_TYPE_PATTERNS.items():
        for keyword in keywords:
            if keyword in user_message:
                _logger.debug("【extract_user_type】识别到用户身份类型: %s, 关键词: %s", user_type, keyword)
                return user_type
    return None


def check_consent_given(user_message: str) -> bool:
    """检查用户是否已同意权利义务告知书。

    Args:
        user_message: 用户输入的消息。

    Returns:
        是否已同意。
    """
    return any(keyword in user_message for keyword in CONSENT_KEYWORDS)


async def _generate_welcome(state: "ConsultationState") -> str:
    """生成欢迎语和权利义务告知。

    Args:
        state: 当前咨询状态。

    Returns:
        欢迎语回复文本。
    """
    prompt = prompt_loader.load("receptionist_prompt")
    user_message = (state.get("facts_raw") or [""])[-1]

    response = await llm_gateway.generate(
        system_prompt=prompt,
        user_message=user_message,
        temperature=0.1,
        is_legal=False,
    )

    return disclaimer.inject(response)


async def _process_consent(state: "ConsultationState", user_message: str) -> "ConsultationState":
    """处理用户同意流程。

    Args:
        state: 当前咨询状态。
        user_message: 用户消息。

    Returns:
        更新后的状态。
    """
    if check_consent_given(user_message):
        state["consent_given"] = True
        _logger.info("【_process_consent】用户已同意权利义务告知, session_id: %s", state.get("session_id", "unknown"))
        state = await _confirm_identity(state)
    else:
        response = await _generate_welcome(state)
        state["final_output"] = response
        state["current_agent"] = "Receptionist"

    return state


async def _confirm_identity(state: "ConsultationState") -> "ConsultationState":
    """引导用户确认身份类型。

    Args:
        state: 当前咨询状态。

    Returns:
        更新后的状态。
    """
    prompt = prompt_loader.load("receptionist_prompt")

    identity_prompt = (
        f"{prompt}\n\n当前状态：用户已同意权利义务告知，现在请引导用户选择身份类型。"
        f"\n用户当前输入：{(state.get('facts_raw') or [''])[-1]}"
    )

    response = await llm_gateway.generate(
        system_prompt=identity_prompt,
        user_message="请选择您的身份类型：",
        temperature=0.1,
        is_legal=False,
    )

    state["final_output"] = disclaimer.inject(response)
    state["current_agent"] = "Receptionist"

    return state


async def _extract_and_confirm_info(state: "ConsultationState", user_message: str) -> "ConsultationState":
    """提取身份信息并询问案件城市。

    Args:
        state: 当前咨询状态。
        user_message: 用户消息。

    Returns:
        更新后的状态。
    """
    user_type = extract_user_type(user_message)

    if user_type:
        state["user_type"] = user_type
        _logger.info("【_extract_and_confirm_info】确认用户身份类型: %s", user_type)

        prompt = prompt_loader.load("receptionist_prompt")
        city_prompt = f"{prompt}\n\n用户已确认身份类型为：{user_type}，\n现在请询问案件发生的城市信息。"

        response = await llm_gateway.generate(
            system_prompt=city_prompt,
            user_message=f"我是{user_type}，请告诉案件发生在哪个城市？",
            temperature=0.1,
            is_legal=False,
        )

        state["final_output"] = disclaimer.inject(response)
        state["current_agent"] = "Receptionist"
    else:
        state = await _confirm_identity(state)

    return state


async def _complete_reception(state: "ConsultationState", user_message: str) -> "ConsultationState":
    """完成接待流程，移交下一 Agent。

    Args:
        state: 当前咨询状态。
        user_message: 用户消息。

    Returns:
        更新后的状态。
    """
    city = user_message.strip()

    conversation_entry = {
        "agent": "Receptionist",
        "user_type": state.get("user_type"),
        "case_city": city,
        "consent_given": state.get("consent_given"),
    }

    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(conversation_entry)

    state["pending_questions"] = [f"案件发生在{city}，请继续收集案件详情"]
    state["current_agent"] = "FactDigger"

    completion_prompt = (
        f"用户身份：{state.get('user_type')}\n"
        f"案件城市：{city}\n"
        f"知情同意：已获得\n\n"
        f"接待流程已完成，正在移交至FactDigger进行事实收集。"
    )

    state["final_output"] = disclaimer.inject(completion_prompt)
    _logger.info(
        "【_complete_reception】接待完成，移交至FactDigger, session_id: %s", state.get("session_id", "unknown")
    )

    return state


async def receptionist_node(state: "ConsultationState") -> "ConsultationState":
    """Receptionist Agent 节点函数。

    处理首次咨询和身份确认流程。

    Args:
        state: 当前 ConsultationState。

    Returns:
        更新后的 ConsultationState。
    """
    _logger.info("【receptionist_node】Receptionist 节点被调用, session_id: %s", state.get("session_id", "unknown"))

    user_message = (state.get("facts_raw") or [""])[-1]

    if not state.get("consent_given"):
        _logger.info("【receptionist_node】用户尚未同意，开始接待流程")
        state = await _process_consent(state, user_message)
    else:
        _logger.info("【receptionist_node】用户已同意，处理身份信息")

        if not state.get("user_type"):
            user_type = extract_user_type(user_message)
            if user_type:
                state["user_type"] = user_type
                state = await _extract_and_confirm_info(state, user_message)
            else:
                state = await _confirm_identity(state)
        elif not state.get("conversation_history") or len(state.get("conversation_history", [])) == 0:
            state = await _extract_and_confirm_info(state, user_message)
        else:
            has_city_info = any(entry.get("case_city") for entry in state.get("conversation_history", []))

            if not has_city_info and user_message.strip():
                state = await _complete_reception(state, user_message)
            else:
                _logger.info("【receptionist_node】接待流程完成，移交至FactDigger")
                state["current_agent"] = "FactDigger"
                state["pending_questions"] = ["请继续收集案件详情"]

    return state


if __name__ == "__main__":
    import asyncio

    res = asyncio.run(receptionist_node(ConsultationState({"consent_given": False})))
    print(res)
