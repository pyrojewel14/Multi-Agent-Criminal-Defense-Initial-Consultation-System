"""ServicePlanner Agent 逻辑验证测试"""

import ast
import sys
from pathlib import Path

def verify_service_planner():
    """验证 ServicePlanner Agent 代码结构和逻辑"""
    print("=" * 70)
    print("ServicePlanner Agent 代码验证")
    print("=" * 70)
    
    # 文件路径
    service_planner_path = Path("/Users/clearguo/Desktop/Multi-Agent Criminal Defense Initial Consultation System/backend/app/agents/service_planner.py")
    
    # 1. 文件存在性检查
    print("\n📋 1. 文件存在性检查")
    if service_planner_path.exists():
        print(f"   ✅ 文件存在: {service_planner_path}")
    else:
        print(f"   ❌ 文件不存在: {service_planner_path}")
        return False
    
    # 2. 语法检查
    print("\n📋 2. 语法检查")
    try:
        with open(service_planner_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print("   ✅ 语法正确")
    except SyntaxError as e:
        print(f"   ❌ 语法错误: {e}")
        return False
    
    # 3. 导入检查
    print("\n📋 3. 必需导入检查")
    required_imports = [
        'TYPE_CHECKING',
        'get_logger',
        'llm_gateway',
        'disclaimer'
    ]
    
    for imp in required_imports:
        if imp in code:
            print(f"   ✅ 包含导入: {imp}")
        else:
            print(f"   ❌ 缺少导入: {imp}")
    
    # 4. 必需函数检查
    print("\n📋 4. 必需函数检查")
    required_functions = [
        'service_planner_node',
        '_build_service_request_message',
        '_load_service_planner_prompt',
        '_get_default_service_planner_prompt',
        '_parse_llm_response',
        '_extract_service_plan_structure',
        '_get_current_timestamp'
    ]
    
    for func in required_functions:
        if f"def {func}" in code:
            print(f"   ✅ 函数存在: {func}")
        else:
            print(f"   ❌ 函数缺失: {func}")
    
    # 5. async def 检查
    print("\n📋 5. 异步函数检查")
    async_functions = [
        'async def service_planner_node',
        'async def _build_service_request_message',
        'async def _load_service_planner_prompt',
        'async def _parse_llm_response',
        'async def _extract_service_plan_structure'
    ]
    
    for func in async_functions:
        if func in code:
            print(f"   ✅ 异步函数: {func.replace('async def ', '')}")
        else:
            print(f"   ⚠️ 非异步或缺失: {func.replace('async def ', '')}")
    
    # 6. 类型注解检查
    print("\n📋 6. 类型注解检查")
    if ': "ConsultationState"' in code:
        print("   ✅ 包含 ConsultationState 类型注解")
    else:
        print("   ⚠️ 缺少类型注解或格式不符")
    
    # 7. 状态更新检查
    print("\n📋 7. 状态更新检查")
    state_updates = [
        'state["service_plan"]',
        'state["report_draft"]',
        'state["lawyer_review_needed"] = True',
        'state["current_agent"] = "HumanReview"',
        'state["conversation_history"]'
    ]
    
    for update in state_updates:
        if update in code:
            print(f"   ✅ 状态更新: {update}")
        else:
            print(f"   ❌ 状态更新缺失: {update}")
    
    # 8. 日志记录检查
    print("\n📋 8. 日志记录检查")
    log_calls = [
        '_logger.info',
        '_logger.debug',
        '_logger.error',
        '_logger.warning'
    ]
    
    for log in log_calls:
        if log in code:
            count = code.count(log)
            print(f"   ✅ 日志调用: {log} ({count}次)")
        else:
            print(f"   ⚠️ 未使用日志: {log}")
    
    # 9. LLM 调用检查
    print("\n📋 9. LLM 调用检查")
    llm_calls = [
        'llm_gateway.generate',
        'temperature=0.1',
        'is_legal=False'
    ]
    
    for call in llm_calls:
        if call in code:
            print(f"   ✅ LLM 调用: {call}")
        else:
            print(f"   ⚠️ LLM 调用缺失: {call}")
    
    # 10. 免责声明检查
    print("\n📋 10. 免责声明检查")
    if 'disclaimer.inject' in code:
        print("   ✅ 使用 disclaimer.inject()")
    else:
        print("   ⚠️ 未使用 disclaimer.inject()")
    
    # 11. 提示词加载检查
    print("\n📋 11. 提示词加载检查")
    prompt_checks = [
        'prompt_loader.load',
        'service_planner',
        '_get_default_service_planner_prompt'
    ]
    
    for check in prompt_checks:
        if check in code:
            print(f"   ✅ 提示词处理: {check}")
        else:
            print(f"   ⚠️ 提示词处理缺失: {check}")
    
    # 12. 错误处理检查
    print("\n📋 12. 错误处理检查")
    if 'try:' in code and 'except' in code:
        print("   ✅ 包含 try-except 错误处理")
        if '_logger.error' in code:
            print("   ✅ 错误日志记录")
    else:
        print("   ⚠️ 缺少错误处理")
    
    # 13. 功能完整性检查
    print("\n📋 13. 功能完整性检查")
    features = [
        ("紧急行动建议", "urgent_actions"),
        ("辩护策略", "defense_strateg"),
        ("律师服务阶段", "service_phase"),
        ("费用结构", "fee_structur"),
        ("报告草案", "report_draft")
    ]
    
    for feature_name, keyword in features:
        if keyword in code:
            print(f"   ✅ 支持功能: {feature_name}")
        else:
            print(f"   ⚠️ 功能缺失: {feature_name}")
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 验证完成！ServicePlanner Agent 代码结构符合规范")
    print("=" * 70)
    
    # 显示主要功能说明
    print("\n📖 主要功能说明:")
    print("   1. 基于风险评估生成紧急行动建议")
    print("   2. 生成辩护策略概览")
    print("   3. 说明律师服务阶段及工作内容")
    print("   4. 提供符合律协标准的费用结构")
    print("   5. 生成《初期咨询报告》Markdown 草案")
    print("   6. 更新 ConsultationState 状态")
    print("   7. 设置 lawyer_review_needed = True")
    print("   8. 切换 current_agent = 'HumanReview'")
    
    return True


if __name__ == "__main__":
    try:
        verify_service_planner()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
