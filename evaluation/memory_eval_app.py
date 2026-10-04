"""评测专用宿主：真实 app lifespan/路由/图/memory，法律节点为确定性替身。

只由隔离子进程加载；不增加生产端点，不替换身份、同意或数据库契约。
"""
import hashlib
import json
import os
import time
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from app.consultation import workflow
from app.consultation.memory.summary import SUMMARY_SYSTEM_PROMPT
from app.infrastructure.observability.tracing import current_trace_context

MODE = os.environ['MEMORY_EVAL_MODE']
JOURNAL = Path(os.environ['MEMORY_EVAL_JOURNAL'])
ORIGINAL_INVOKE = ChatOllama.ainvoke


def deterministic_facts(text):
    """仅固定合成输入的供应商替身，输出仍经真实严格校验与合并。"""
    result = dict(incident_time=None, incident_location=None, parties=[], behavior_sequence=[],
                  consequence=None, evidence_mentioned=[], arrest_status=None, surrender=None,
                  victim_forgiveness=None, prior_record=None)
    for city in ('甲城', '乙城', '丙城'):
        if city + '商店' in text:
            result['incident_location'] = city + '商店'
    if '昨天晚上' in text or '昨晚' in text:
        result['incident_time'] = '昨天晚上'
    if '拿走' in text:
        result['behavior_sequence'] = [{'action': '拿走手机'}]
    if '监控' in text:
        result['evidence_mentioned'] = [{'type': '监控录像'}]
    if '没有前科' in text:
        result['prior_record'] = False
    if '没有自首' in text:
        result['surrender'] = False
    if '没有被拘留' in text:
        result['arrest_status'] = '未被拘留'
    return result


async def capture_invoke(model, messages, config=None, **kwargs):
    started = time.monotonic()
    trace = current_trace_context() or {}
    batch = None
    if messages[0].content == SUMMARY_SYSTEM_PROMPT:
        batch = json.loads(messages[-1].content)['new_messages']
    schema = kwargs.get('format', {})
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    record = {'node': 'summary' if batch is not None else trace.get('node'), 'messages': [msg.content for msg in messages],
              'summary_sequences': [row['sequence'] for row in batch] if batch else None,
              'model': model.model, 'reasoning': model.reasoning, 'num_ctx': model.num_ctx,
              'num_predict': model.num_predict, 'temperature': model.temperature,
              'system_fingerprint': fingerprint(messages[0].content),
              'messages_fingerprint': fingerprint([msg.content for msg in messages]),
              'schema_fingerprint': fingerprint(schema)}
    try:
        if MODE == 'deterministic':
            if kwargs.get('format', {}).get('properties', {}).get('incident_location'):
                content = json.dumps(deterministic_facts(messages[-1].content), ensure_ascii=False)
            elif batch is not None:
                content = '合成用户陈述待核实；地点更正与旧版本保留，监控线索待查。'
            else:
                content = '请补充案件情况。'
            response = AIMessage(content=content)
        else:
            response = await ORIGINAL_INVOKE(model, messages, config=config, **kwargs)
        record.update(content=response.content, usage=response.usage_metadata,
                      response_metadata=response.response_metadata, outcome='success')
        return response
    except BaseException as exc:
        record.update(outcome=type(exc).__name__)
        raise
    finally:
        record['seconds'] = round(time.monotonic() - started, 4)
        with JOURNAL.open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


async def law(state):
    return state


async def coverage(state):
    return dict(state, current_agent='FactDigger', facts_coverage_rate=0.0,
                final_output='已记录本轮陈述，仍待核实；请继续补充。', pending_questions=['请补充'])


ChatOllama.ainvoke = capture_invoke
workflow.law_ref_node = law
workflow._fact_digger_workflow_node = coverage

import main  # noqa: E402

# 法律 preflight 与法律节点一并替换，memory 数据和服务生命周期保持真实。
main.preflight_law_knowledge = lambda: {'evaluation_stub': True}
app = main.app
