"""固定合成案件 memory 验证；使用隔离服务进程，公开结果仅导出受控字段。"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

import aiosqlite  # noqa: E402
import httpx  # noqa: E402
import websockets  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

CODE_FINGERPRINT_PATHS = frozenset({
    'evaluation/memory_cases.json', 'evaluation/run_memory_eval.py', 'evaluation/memory_eval_app.py',
    'backend/app/consultation/agents/fact_digger.py', 'backend/app/prompts/extract_case_facts.txt',
    'backend/app/infrastructure/llm/gateway.py', 'backend/app/consultation/workflow.py',
    'backend/app/consultation/service.py', 'backend/main.py',
    'backend/app/consultation/memory/__init__.py', 'backend/app/consultation/memory/case.py',
    'backend/app/consultation/memory/context.py', 'backend/app/consultation/memory/coordinator.py',
    'backend/app/consultation/memory/schemas.py', 'backend/app/consultation/memory/summary.py',
    'backend/app/consultation/memory/transcript.py',
})


def fingerprint(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def load_cases(path=None):
    cases = json.loads(Path(path or ROOT / 'evaluation/memory_cases.json').read_text())
    seen = set()
    for case in cases:
        if case['id'] not in {'short', 'long'} or case['id'] in seen or case['recent'] not in {2, 4, 8}:
            raise ValueError('invalid memory cases')
        seen.add(case['id'])
        if not case['turns'] or any(not turn['content'].strip() for turn in case['turns']):
            raise ValueError('empty case')
    if seen != {'short', 'long'}:
        raise ValueError('missing case')
    return cases


def model_identity(tags, name):
    found = next((model for model in tags['models'] if model['name'] == name), None)
    if not found or not re.fullmatch('[0-9a-f]{64}', found.get('digest', '')):
        raise ValueError('model digest unavailable')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}', name):
        raise ValueError('invalid model name')
    quantization = found.get('details', {}).get('quantization_level')
    if quantization is not None and not re.fullmatch('[A-Z0-9_]{1,32}', quantization):
        raise ValueError('invalid quantization')
    return {'name': name, 'digest': found['digest'], 'quantization': quantization}


def summary_coverage(batches, cursor):
    flattened = [sequence for batch in batches for sequence in batch]
    return flattened == list(range(1, cursor + 1))


def recovery_assertions(before, after):
    return {'raw_restored': before['raw_digest'] == after['raw_digest'],
            'summary_restored': before['memory'].get('summary') == after['memory'].get('summary'),
            'case_restored': before['memory'].get('case') == after['memory'].get('case'),
            'recent_restored': before['memory'].get('recent') == after['memory'].get('recent'),
            'pending_restored': before['pending'] == after['pending'] == ['fact_intake']}


def recall_assertions(case_id, answer):
    """固定场景的粗粒度必要信息检查；不充当法律或泛化摘要质量指标。"""
    answer = answer or ''
    checks = {'unverified': bool(re.search('未.{0,3}核实|待核实|未经确认|未经验证', answer)),
              'prior_denial': bool(re.search('没有前科|无(犯罪)?前科|前科.{0,6}(无|否认|没有)|前科情况为用户自称无', answer))}
    if case_id == 'short':
        checks['location'] = '甲城' in answer
    else:
        checks.update(time=bool(re.search('昨天晚上|昨晚', answer)), old_locations='甲城' in answer and '乙城' in answer,
                      correction='丙城' in answer and bool(re.search('更正|修正', answer)), evidence='监控' in answer,
                      surrender_denial=bool(re.search('没有自首|未自首|无自首|自首.{0,6}(无|否认|没有)|自首情况为用户自称无', answer)))
    return checks


def numeric(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('metric must be nonnegative numeric')
    return value


def exit_code(value):
    """保留 OS 的带符号退出码；SIGTERM 后仍须核对真实 lifespan 清理日志。"""
    if type(value) is not int or not -128 <= value <= 255:
        raise ValueError('invalid process exit code')
    return value


def usage_metrics(value):
    if value is None:
        return None
    return {key: numeric(value[key]) for key in ('input_tokens', 'output_tokens', 'total_tokens')}


def digest(value):
    if not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid fingerprint')
    return value


def context_metrics(value):
    if value['kind'] not in {'full_history', 'active_context'}:
        raise ValueError('invalid context kind')
    return {'kind': value['kind'], **{key: numeric(value[key]) for key in (
        'estimated_input_tokens', 'utf8_bytes', 'seconds')}, 'actual_usage': usage_metrics(value['actual_usage']),
        **{key: digest(value[key]) for key in ('messages_fingerprint', 'system_fingerprint', 'question_fingerprint')},
        'answer_fingerprint': fingerprint(value.get('answer')),
        'recall_checks': boolean_checks(value.get('recall_checks', {}))}


def boolean_checks(value):
    if any(not re.fullmatch('[a-z_]{1,80}', key) or type(result) is not bool for key, result in value.items()):
        raise ValueError('checks must be boolean metrics')
    return dict(value)


def config_metrics(value):
    keys = ('recent_messages', 'context_token_budget', 'model_window', 'output_reserve',
            'summary_input_budget', 'summary_output_budget', 'summary_timeout')
    result = {key: numeric(value[key]) for key in keys if key in value}
    if 'llm_policy' in value:
        result['llm_policy'] = {key: numeric(value['llm_policy'][key]) for key in (
            'total_timeout_seconds', 'attempt_timeout_seconds', 'max_attempts', 'backoff_seconds')}
    return result


def call_metrics(value):
    node = value['node']
    if node not in {'fact_intake', 'summary', None} or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}', value['model']):
        raise ValueError('invalid provider metadata')
    if value['outcome'] not in {'success', 'TimeoutError', 'CancelledError', 'ConnectError', 'ResponseError'}:
        raise ValueError('unsupported outcome')
    sequences = value['summary_sequences']
    if sequences is not None and any(type(item) is not int or item < 1 for item in sequences):
        raise ValueError('invalid summary sequences')
    if value['reasoning'] not in (None, True, False):
        raise ValueError('invalid reasoning metadata')
    return {'node': node, 'model': value['model'], 'outcome': value['outcome'],
        'usage': usage_metrics(value['usage']), 'seconds': numeric(value['seconds']),
        'num_ctx': numeric(value['num_ctx']), 'num_predict': numeric(value['num_predict']),
        'reasoning': value['reasoning'], 'summary_sequences': sequences,
        **{key: digest(value[key]) for key in ('system_fingerprint', 'messages_fingerprint', 'schema_fingerprint')},
        'temperature': numeric(value['temperature'])}


def code_fingerprint_metrics(value):
    """只允许固定仓库相对路径和 SHA-256 摘要，拒绝未知条目及任意正文。"""
    if not isinstance(value, dict) or any(
        not isinstance(path, str) or path not in CODE_FINGERPRINT_PATHS
        or not isinstance(sha256, str) or not re.fullmatch('[0-9a-fA-F]{64}', sha256)
        for path, sha256 in value.items()
    ):
        raise ValueError('invalid code fingerprints')
    return dict(value)


def public_report(private):
    """只复制受控枚举、数值与指纹，不对任意正文做黑名单替换。"""
    if private['mode'] not in {'live', 'deterministic'}:
        raise ValueError('invalid evaluation mode')
    model = private['model']
    if model is not None:
        model = model_identity({'models': [{'name': model['name'], 'digest': model['digest'],
                                           'details': {'quantization_level': model['quantization']}}]}, model['name'])
    out = {'schema_version': 1, 'mode': private['mode'], 'model': model,
           'code_fingerprints': code_fingerprint_metrics(private['code_fingerprints']), 'scenarios': [],
           'limitations': ['synthetic_small_sample', 'legal_nodes_stubbed', 'single_worker',
                           'graceful_restart_only', 'cross_sqlite_not_atomic', 'summary_loss_possible']}
    for scenario in private['scenarios']:
        checks = boolean_checks(scenario['checks'])
        summary = scenario['summary']
        if summary['status'] not in {'empty', 'ready', 'failed', 'blocked_large_message'}:
            raise ValueError('invalid summary status')
        if scenario['id'] not in {'short', 'long'}:
            raise ValueError('invalid scenario id')
        row = {'id': scenario['id'], 'recent': numeric(scenario['recent']), 'raw_count': numeric(scenario['raw_count']),
               'checks': checks, 'context': [context_metrics(value) for value in scenario['context']],
               'calls': [call_metrics(value) for value in scenario['calls']]}
        row['summary'] = {'version': numeric(summary['version']),
                          'through_sequence': numeric(summary['through_sequence']), 'status': summary['status']}
        row['summary']['text_fingerprint'] = fingerprint(summary.get('text', ''))
        api = scenario['api']
        row['api'] = {'restart_checks': boolean_checks(api['restart_checks']),
            **boolean_checks({key: api[key] for key in ('replay_raw_unchanged', 'replay_graph_unchanged')}),
            **{key: numeric(api[key]) for key in ('continued_raw_delta', 'auth_status', 'permission_status', 'consent_status')},
            'shutdown_exit_codes': [exit_code(code) for code in api['shutdown_exit_codes']]}
        row['config'] = config_metrics(scenario.get('config', {}))
        out['scenarios'].append(row)
    return out


async def read_snapshot(directory, sid):
    from app.consultation.workflow import ConsultationOrchestrator
    async with aiosqlite.connect(directory / 'checkpoint.db') as conn:
        saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        graph = ConsultationOrchestrator(saver, persistent=True)
        snapshot = await graph.get_snapshot(sid)
    with sqlite3.connect(directory / 'audit.db') as conn:
        rows = conn.execute('SELECT id, sequence, sender_type, content, command_id, record_kind FROM consultation_messages ORDER BY sequence').fetchall()
    raw = [dict(zip(('id', 'sequence', 'sender_type', 'content', 'command_id', 'record_kind'), row)) for row in rows]
    return {'state': snapshot.values, 'memory': snapshot.values.get('memory', {}),
            'pending': list(snapshot.next), 'raw': raw, 'raw_digest': fingerprint(raw)}


def read_calls(directory):
    return [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]


class IsolatedServer:
    def __init__(self, mode, directory, recent):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self.env = dict(os.environ, DATABASE_PATH=str(directory / 'audit.db'),
            LANGGRAPH_CHECKPOINT_DB_PATH=str(directory / 'checkpoint.db'),
            MEMORY_EVAL_MODE=mode, MEMORY_EVAL_JOURNAL=str(directory / 'calls.jsonl'),
            MEMORY_RECENT_MESSAGES=str(recent), PYTHONPATH=os.pathsep.join((str(ROOT / 'backend'), str(ROOT))),
            LLM_TYPE='OLLAMA', JWT_SECRET_KEY='synthetic-eval-signing-key-only-32-bytes', REDIS_DB='15')
        self.process = None
        self.exit_codes = []
        self.log = None

    async def start(self):
        self.log = (self.directory / 'server.log').open('a')
        self.process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'evaluation.memory_eval_app:app',
            '--host', '127.0.0.1', '--port', str(self.port)], cwd=ROOT / 'backend', env=self.env,
            stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 40
        async with httpx.AsyncClient(trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('isolated server startup failed; inspect scratch server.log')
                try:
                    response = await client.get(self.url + '/health')
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.1)
        raise TimeoutError('isolated server startup timeout')

    async def stop(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            await asyncio.to_thread(self.process.wait, 30)
        if self.process:
            self.exit_codes.append(self.process.returncode)
        if self.log:
            self.log.close()
        self.process = None


async def run_api(mode, directory, recent=2, case=None):
    """只停止本函数创建的进程；不改真实业务库、不清空 Redis。"""
    directory = directory.resolve()
    case = case or load_cases()[1]
    server = IsolatedServer(mode, directory, recent)
    try:
        await server.start()
        async with httpx.AsyncClient(base_url=server.url, timeout=160, trust_env=False) as client:
            async def login(username):
                credentials = {'username': username, 'password': 'SyntheticOnly123!'}
                registered = await client.post('/api/v1/auth/register', json=credentials)
                assert registered.status_code == 201, registered.text
                response = await client.post('/api/v1/auth/login', json=credentials)
                assert response.status_code == 200, response.text
                data = response.json()['data']
                return data['access_token'], data['user']['id']
            token, owner = await login('synthetic-owner')
            other_token, _ = await login('synthetic-other')
            headers = {'Authorization': 'Bearer ' + token}
            response = await client.post('/api/v1/sessions', json={'user_type': 'suspect'}, headers=headers)
            assert response.status_code == 200, response.text
            sid = response.json()['session_id']
            prefix = '/api/v1/sessions/' + sid
            auth_status = (await client.get(prefix + '/state')).status_code
            permission_status = (await client.get(prefix + '/state', headers={'Authorization': 'Bearer ' + other_token})).status_code
            consent_status = (await client.post(prefix + '/message', json={'session_id': sid, 'content': '尚未同意'}, headers=headers)).status_code
            response = await client.post(prefix + '/confirm-consent', headers=headers, json={
                'session_id': sid, 'consent_given': True, 'consent_timestamp': '2026-10-04T00:00:00Z',
                'consent_version': '1.0', 'identity_info': {'user_type': 'suspect'}, 'idempotency_key': 'consent'})
            assert response.status_code == 200, response.text
            snapshots = []
            replies = []
            for index, turn in enumerate(case['turns']):
                payload = {'session_id': sid, 'content': turn['content'], 'idempotency_key': f'turn-{index}'}
                if index % 2 == 0:
                    response = await client.post(prefix + '/message', json=payload, headers=headers)
                    assert response.status_code == 200, response.text
                    replies.append(response.json()['message_id'])
                else:
                    async with websockets.connect(server.url.replace('http:', 'ws:') + f'/api/v1/sessions/{sid}/ws',
                            additional_headers=headers) as ws:
                        assert json.loads(await ws.recv())['type'] == 'ack'
                        await ws.send(json.dumps(dict(payload, type='message'), ensure_ascii=False))
                        while True:
                            item = json.loads(await asyncio.wait_for(ws.recv(), 160))
                            assert item['type'] != 'error', item
                            if item['type'] == 'message':
                                replies.append(item['message_id'])
                                break
                snapshots.append(await read_snapshot(directory, sid))
            before = await read_snapshot(directory, sid)
            calls_before = len(read_calls(directory))
            await server.stop()
            await server.start()
            after = await read_snapshot(directory, sid)
            response = await client.get(prefix + '/state', headers=headers)
            assert response.status_code == 200, response.text
            listing = await client.get('/api/v1/consultations/list', headers=headers)
            assert listing.status_code == 200 and sid in json.dumps(listing.json())
            # 成功末轮以另一 transport 和相同 key 重放，必须返回原 reply ID。
            last = len(case['turns']) - 1
            payload = {'session_id': sid, 'content': case['turns'][-1]['content'], 'idempotency_key': f'turn-{last}'}
            if last % 2:
                replay = await client.post(prefix + '/message', json=payload, headers=headers)
                assert replay.status_code == 200 and replay.json()['message_id'] == replies[-1]
            else:
                async with websockets.connect(server.url.replace('http:', 'ws:') + f'/api/v1/sessions/{sid}/ws', additional_headers=headers) as ws:
                    await ws.recv()
                    await ws.send(json.dumps(dict(payload, type='message'), ensure_ascii=False))
                    while True:
                        item = json.loads(await asyncio.wait_for(ws.recv(), 160))
                        assert item['type'] != 'error', item
                        if item['type'] == 'message':
                            assert item['message_id'] == replies[-1]
                            break
            replayed = await read_snapshot(directory, sid)
            calls_after = len(read_calls(directory))
            response = await client.post(prefix + '/message', headers=headers, json={
                'session_id': sid, 'content': '补充：目前没有被拘留。', 'idempotency_key': 'continued'})
            assert response.status_code == 200, response.text
            continued = await read_snapshot(directory, sid)
            await server.stop()
            report = {'restart_checks': recovery_assertions(before, after),
                'replay_raw_unchanged': replayed['raw_digest'] == before['raw_digest'],
                'replay_graph_unchanged': calls_before == calls_after and replayed['state'] == before['state'],
                'continued_raw_delta': len(continued['raw']) - len(before['raw']),
                'auth_status': auth_status, 'permission_status': permission_status, 'consent_status': consent_status,
                'shutdown_exit_codes': server.exit_codes, 'before': before, 'continued': continued, 'snapshots': snapshots,
                'owner': owner}
            (directory / 'api-private.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return report
    finally:
        await server.stop()


def context_pair(state, raw, question, settings=None):
    from app.consultation.memory.context import ContextBuilder, ContextLimitError
    from app.security.sensitive_filter import mask_pii
    system = '仅依据所提供的用户陈述回答回忆问题；所有陈述未经核实，不推断有罪或法律结论。'
    active = ContextBuilder(settings).build(system, question, state, task='recall')
    full = ContextBuilder(settings).build(system, question, history=[HumanMessage(content=mask_pii(json.dumps(
        {'role': row['sender_type'], 'content': row['content']}, ensure_ascii=False))) for row in raw])
    if full.dropped_groups:
        raise ContextLimitError('full-history baseline exceeds input budget')
    return full, active


async def compare_context(mode, state, raw, question, recent=8):
    from dataclasses import replace

    from app.consultation.memory.context import MemorySettings
    from app.infrastructure.llm.factory import chat_model_factory
    full, active = context_pair(state, raw, question, replace(MemorySettings.from_env(), recent_messages=recent))
    system = full.messages[0].content
    result = []
    for name, built in [('full_history', full), ('active_context', active)]:
        start = time.monotonic()
        usage = None
        answer = None
        if mode == 'live':
            from app.infrastructure.llm.gateway import _bounded_model
            model = _bounded_model(chat_model_factory.create_precise_model(0), 333, False)
            response = await model.ainvoke(built.messages)
            usage = response.usage_metadata
            answer = response.content
        result.append({'kind': name, 'estimated_input_tokens': built.estimated_input_tokens,
            'utf8_bytes': sum(len(msg.content.encode()) for msg in built.messages), 'actual_usage': usage,
            'seconds': round(time.monotonic() - start, 4), 'messages_fingerprint': fingerprint([m.content for m in built.messages]),
            'system_fingerprint': fingerprint(system), 'question_fingerprint': fingerprint(question), 'answer': answer})
    chat_model_factory.close()
    return result


def case_checks(case, api, calls):
    before = api['before']
    summary = before['memory'].get('summary', {})
    checks = {'raw_complete': len(before['raw']) == 3 + 2 * len(case['turns']),
        'raw_ordered': [row['sequence'] for row in before['raw']] == list(range(1, len(before['raw']) + 1)),
        'inputs_exact': [row['content'] for row in before['raw'] if row['sender_type'] == 'user'][1:] == [turn['content'] for turn in case['turns']],
        'sources_traceable': True, 'all_fact_artifacts_valid': all(snapshot['state'].get('artifact_results', {}).get('fact', {}).get('status') == 'success' for snapshot in api['snapshots'])}
    ids = {row['id'] for row in before['raw']}
    fields = before['memory'].get('case', {}).get('fields', {})
    for entry in fields.values():
        for item in entry.get('items', [entry]):
            for version in [item, *item.get('alternatives', [])]:
                checks['sources_traceable'] &= bool(version['source_ids']) and set(version['source_ids']) <= ids
                checks['sources_traceable'] &= version['status'] != 'confirmed'
    if case['id'] == 'short':
        checks['short_not_compressed'] = summary.get('version') == 0
        checks['location_recalled'] = '甲城' in str(fields.get('incident_location', {}).get('value'))
        checks['denial_retained'] = fields.get('prior_record', {}).get('value') is False
    else:
        snapshots = api['snapshots']
        location = fields.get('incident_location', {})
        checks['long_compressed'] = summary.get('version', 0) > 0 and summary.get('status') == 'ready'
        checks['incremental_summary_once'] = summary_coverage([call['summary_sequences'] for call in calls
            if call.get('summary_sequences') and max(call['summary_sequences']) <= summary.get('through_sequence', 0)
            and call['outcome'] == 'success'], summary.get('through_sequence', 0))
        checks['repeat_no_new_version'] = len(snapshots[2]['memory'].get('case', {}).get('fields', {}).get('incident_location', {}).get('alternatives', [])) == 0
        checks['conflict_retained'] = snapshots[3]['memory'].get('case', {}).get('fields', {}).get('incident_location', {}).get('status') == 'conflicted'
        checks['correction_with_history'] = location.get('status') == 'user_corrected_unverified' and '丙城' in str(location.get('value')) and len(location.get('alternatives', [])) >= 2
        checks['old_facts_retained'] = bool(fields.get('evidence_mentioned', {}).get('items')) and bool(fields.get('incident_time', {}).get('value'))
        checks['denials_retained'] = fields.get('prior_record', {}).get('value') is False and fields.get('surrender', {}).get('value') is False
    checks.update(api['restart_checks'], replay_raw_unchanged=api['replay_raw_unchanged'],
                  replay_graph_unchanged=api['replay_graph_unchanged'], continued=api['continued_raw_delta'] == 2)
    return checks


async def main(args):
    load_dotenv(ROOT / 'backend/.env')
    args.work_dir = args.work_dir.resolve()
    if not args.work_dir.is_relative_to(ROOT / '.scratch'):
        raise ValueError('raw evaluation evidence must stay in .scratch')
    args.work_dir.mkdir(parents=True, exist_ok=False)
    model = None
    if args.mode == 'live':
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(os.environ['OLLAMA_BASE_URL'].rstrip('/') + '/api/tags', timeout=15)
            response.raise_for_status()
            model = model_identity(response.json(), os.environ['OLLAMA_MODEL_NAME'])
    files = ['evaluation/memory_cases.json', 'evaluation/run_memory_eval.py', 'evaluation/memory_eval_app.py',
        'backend/app/consultation/agents/fact_digger.py', 'backend/app/prompts/extract_case_facts.txt',
        'backend/app/infrastructure/llm/gateway.py', 'backend/app/consultation/workflow.py',
        'backend/app/consultation/service.py', 'backend/main.py',
        *[str(path.relative_to(ROOT)) for path in (ROOT / 'backend/app/consultation/memory').glob('*.py')]]
    report = {'mode': args.mode, 'model': model,
        'code_fingerprints': {file: fingerprint((ROOT / file).read_bytes()) for file in files}, 'scenarios': []}
    for case in load_cases():
        directory = args.work_dir / case['id']
        api = await run_api(args.mode, directory, recent=case['recent'], case=case)
        calls = read_calls(directory)
        context = await compare_context(args.mode, api['before']['state'], api['before']['raw'], case['recall'], recent=case['recent'])
        from app.consultation.memory.context import MemorySettings
        from app.infrastructure.llm.gateway import LLMCallPolicy
        config = asdict(MemorySettings.from_env())
        config['recent_messages'] = case['recent']
        config['llm_policy'] = asdict(LLMCallPolicy.from_env())
        for value in context:
            value['recall_checks'] = recall_assertions(case['id'], value['answer']) if args.mode == 'live' else {}
        private = {'id': case['id'], 'recent': case['recent'], 'raw_count': len(api['before']['raw']),
            'checks': case_checks(case, api, calls), 'summary': api['before']['memory']['summary'],
            'api': api, 'calls': [], 'context': context, 'config': config}
        if args.mode == 'live':
            for value in context:
                private['checks']['recall_' + value['kind']] = all(value['recall_checks'].values())
        for call in calls:
            private['calls'].append({key: call[key] for key in ('node', 'model', 'reasoning', 'num_ctx', 'num_predict', 'outcome', 'usage', 'seconds', 'summary_sequences', 'system_fingerprint', 'messages_fingerprint', 'schema_fingerprint', 'temperature')})
        report['scenarios'].append(private)
        print(json.dumps({'id': case['id'], 'checks': private['checks']}, ensure_ascii=False), flush=True)
    (args.work_dir / 'report-private.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    args.report.write_text(json.dumps(public_report(report), ensure_ascii=False, indent=2))
    return 0 if all(all(row['checks'].values()) for row in report['scenarios']) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['deterministic', 'live'], default='deterministic')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parsed = parser.parse_args()
    sys.exit(asyncio.run(main(parsed)))
