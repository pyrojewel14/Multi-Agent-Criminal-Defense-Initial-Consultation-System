"""Memory 评测边界：拒绝错误证据、保护公开报告、真实服务停启。"""
import json

import pytest

from evaluation.run_memory_eval import (
    context_pair,
    load_cases,
    model_identity,
    public_report,
    recall_assertions,
    recovery_assertions,
    summary_coverage,
)


def test_cases_have_fixed_synthetic_inputs_and_no_gold_sent_to_extractor(tmp_path):
    cases = load_cases()
    assert [case['id'] for case in cases] == ['short', 'long']
    assert cases[0]['recent'] == 8 and cases[1]['recent'] == 2
    assert {turn['kind'] for case in cases for turn in case['turns']} >= {
        'initial', 'supplement', 'repeat', 'conflict', 'correction', 'denial'}
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps([dict(cases[0], recent=0)]))
    with pytest.raises(ValueError):
        load_cases(path)


def test_digest_must_match_actual_model_and_cannot_be_missing():
    tags = {'models': [{'name': 'qwen3.5:9b', 'digest': 'a' * 64, 'details': {'quantization_level': 'Q4_K_M'}}]}
    assert model_identity(tags, 'qwen3.5:9b')['digest'] == 'a' * 64
    with pytest.raises(ValueError):
        model_identity(tags, 'missing')
    with pytest.raises(ValueError):
        model_identity({'models': [{'name': 'qwen3.5:9b', 'digest': 'bad'}]}, 'qwen3.5:9b')


def test_summary_coverage_detects_repeat_gap_and_failed_candidate():
    assert summary_coverage([[1, 2], [3, 4]], 4) is True
    assert summary_coverage([[1, 2], [2, 3, 4]], 4) is False
    assert summary_coverage([[1, 2], [4]], 4) is False
    assert summary_coverage([[1, 2]], 0) is False


def test_restart_assertions_reject_changed_case_pending_raw_or_summary():
    before = {'raw_digest': 'a', 'memory': {'case': {'fields': {'x': 1}}, 'recent': [1],
                'summary': {'version': 2, 'through_sequence': 4, 'status': 'ready', 'text': '摘要'}},
              'pending': ['fact_intake']}
    assert all(recovery_assertions(before, before).values())
    after = json.loads(json.dumps(before))
    after['memory']['summary']['through_sequence'] = 5
    after['pending'] = []
    checks = recovery_assertions(before, after)
    assert not checks['summary_restored'] and not checks['pending_restored']
    after['raw_digest'] = 'b'
    assert not recovery_assertions(before, after)['raw_restored']


def test_public_report_allowlist_drops_bodies_addresses_tokens_and_freeform_errors():
    private = {'mode': 'deterministic', 'model': None, 'code_fingerprints': {},
        'scenarios': [{'id': 'short', 'recent': 8, 'checks': {'short_not_compressed': True},
            'raw_count': 5, 'summary': {'version': 0, 'through_sequence': 0, 'status': 'empty',
                                       'text': '/Users/private original contents'},
            'context': [], 'calls': [], 'content': '秘密', 'token': 'secret',
            'error': 'http://private.local', 'api': {'restart_checks': {'raw_restored': True},
                'replay_raw_unchanged': True, 'replay_graph_unchanged': True, 'continued_raw_delta': 2,
                'auth_status': 401, 'permission_status': 403, 'consent_status': 403, 'shutdown_exit_codes': [-15, 0]}}],
        'base_url': 'http://private.local', 'password': 'secret'}
    out = public_report(private)
    encoded = json.dumps(out, ensure_ascii=False)
    assert all(value not in encoded for value in ['private.local', '/Users/', '秘密', 'secret'])
    assert out['scenarios'][0]['summary']['version'] == 0
    assert out['scenarios'][0]['api']['shutdown_exit_codes'] == [-15, 0]
    private['scenarios'][0]['checks']['unsafe'] = 'secret'
    with pytest.raises(ValueError):
        public_report(private)


def test_public_nested_metrics_cannot_export_provider_bodies():
    private = {'mode': 'live', 'model': None, 'code_fingerprints': {}, 'scenarios': [
        {'id': 'short', 'recent': 8, 'raw_count': 5, 'checks': {},
         'summary': {'version': 0, 'through_sequence': 0, 'status': 'empty'},
         'calls': [], 'context': [{'kind': 'full_history', 'estimated_input_tokens': 10,
           'utf8_bytes': 9, 'actual_usage': {'input_tokens': 8, 'output_tokens': 1,
              'total_tokens': 9, 'body': 'secret original'}, 'seconds': 1,
           'messages_fingerprint': 'a' * 64, 'system_fingerprint': 'b' * 64,
           'question_fingerprint': 'c' * 64, 'answer': 'secret original'}],
         'api': {'restart_checks': {}, 'replay_raw_unchanged': True, 'replay_graph_unchanged': True,
                 'continued_raw_delta': 2, 'auth_status': 401, 'permission_status': 403,
                 'consent_status': 403, 'shutdown_exit_codes': [0, 0]}}]}
    assert 'secret original' not in json.dumps(public_report(private))


def test_public_code_fingerprints_accept_known_paths_and_copy_valid_hashes():
    fingerprints = {'evaluation/run_memory_eval.py': 'a' * 64,
                    'backend/app/consultation/memory/context.py': 'b' * 64}
    private = {'mode': 'deterministic', 'model': None,
               'code_fingerprints': fingerprints, 'scenarios': []}
    public = public_report(private)['code_fingerprints']
    assert public == fingerprints
    fingerprints['evaluation/run_memory_eval.py'] = 'secret body'
    assert public['evaluation/run_memory_eval.py'] == 'a' * 64


@pytest.mark.parametrize('fingerprints', [
    {'/Users/private/secret': 'secret body'},
    {'/Users/private/secret': 'a' * 64},
    {'../evaluation/run_memory_eval.py': 'a' * 64},
    {'evaluation/../evaluation/run_memory_eval.py': 'a' * 64},
    {'./evaluation/run_memory_eval.py': 'a' * 64},
    {'C:\\private\\secret': 'a' * 64},
    {'evaluation/unknown.py': 'a' * 64},
    {'backend/app/consultation/memory/unknown.py': 'a' * 64},
    {'evaluation/run_memory_eval.py': 'secret body'},
    {'evaluation/run_memory_eval.py': 'a' * 63},
    {'evaluation/run_memory_eval.py': 'g' * 64},
    {'evaluation/run_memory_eval.py': 'a' * 64 + '\n'},
    {'evaluation/run_memory_eval.py': {'body': 'secret body'}},
    {'evaluation/run_memory_eval.py': None},
    {1: 'a' * 64},
    ['secret body'],
    None,
])
def test_public_code_fingerprints_reject_untrusted_paths_and_bodies(fingerprints):
    private = {'mode': 'deterministic', 'model': None,
               'code_fingerprints': fingerprints, 'scenarios': []}
    with pytest.raises(ValueError, match='invalid code fingerprints'):
        public_report(private)


def test_recall_scoring_rejects_empty_template_and_missing_old_versions():
    assert not all(recall_assertions('short', '信息不明').values())
    assert all(recall_assertions('short', '甲城商店，没有前科，用户陈述未经核实。').values())
    assert not all(recall_assertions('long', '丙城，无前科，没有自首，未经核实。').values())
    assert all(recall_assertions('long', '昨天晚上，甲城、乙城更正为丙城，有监控，没有前科，没有自首，未经核实。').values())
    assert all(recall_assertions('short', '甲城商店且无犯罪前科，用户声称未经验证。').values())
    assert all(recall_assertions('long', '昨晚，甲城、乙城更正为丙城，监控待核实。前科情况为用户自称无，自首情况为用户自称无。').values())


@pytest.mark.parametrize('correction', [
    '当前修正后的地点为丙城商店',
    '丙城商店为用户修正后的说法',
])
def test_recall_scoring_accepts_explicit_correction_synonyms(correction):
    answer = f'昨天晚上，甲城、乙城存在冲突。{correction}。监控待核实，无前科，未自首。'
    assert all(recall_assertions('long', answer).values())


@pytest.mark.parametrize(('answer', 'failed_check'), [
    ('昨晚，甲城、乙城存在冲突。当前地点为丙城。监控待核实，无前科，未自首。', 'correction'),
    ('昨晚，甲城、乙城存在冲突。地点为用户修正后的说法。监控待核实，无前科，未自首。', 'correction'),
    ('昨晚，甲城，修正后的地点为丙城。监控待核实，无前科，未自首。', 'old_locations'),
    ('昨晚，乙城，修正后的地点为丙城。监控待核实，无前科，未自首。', 'old_locations'),
    ('昨晚，甲城、乙城，修正为丙城。监控待核实，前科未知，未自首。', 'prior_denial'),
    ('昨晚，甲城、乙城，修正为丙城。监控待核实，无前科，自首未知。', 'surrender_denial'),
])
def test_recall_scoring_synonyms_preserve_required_facts(answer, failed_check):
    checks = recall_assertions('long', answer)
    assert checks[failed_check] is False
    assert not all(checks.values())


def test_context_comparison_cannot_silently_drop_full_history():
    from app.consultation.memory.context import ContextLimitError, MemorySettings
    with pytest.raises(ContextLimitError):
        context_pair({}, [{'sender_type': 'user', 'content': '中' * 1000}], '问题',
                     MemorySettings(context_token_budget=500))
    full, active = context_pair({}, [{'sender_type': 'user', 'content': '甲城'}], '问题')
    assert full.messages[0].content == active.messages[0].content
    assert full.messages[1].content == active.messages[-1].content == '问题'
    assert full.dropped_groups == 0


@pytest.mark.asyncio
async def test_real_uvicorn_restart_http_ws_is_offline_repeatable(tmp_path):
    from evaluation.run_memory_eval import run_api
    report = await run_api('deterministic', tmp_path, recent=2)
    assert all(report['restart_checks'].values())
    assert report['replay_raw_unchanged'] and report['replay_graph_unchanged']
    assert report['continued_raw_delta'] == 2
    assert report['auth_status'] in (401, 403)
    assert report['permission_status'] == 403
    assert report['consent_status'] == 403
