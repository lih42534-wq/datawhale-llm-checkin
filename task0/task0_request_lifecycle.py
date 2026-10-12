"""DataWhale Task0 · 请求生命周期 · 旅行包 / lih42534-wq
Source: https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/INF-REQUEST-LIFECYCLE_Inference_Request_Lifecycle.ipynb
Four TODOs completed with AI assistance; original tests retained.
CPU event replay + PyTorch tensor cross-check. Not a real-model benchmark.
Run: ../work/venv/bin/python task0_request_lifecycle.py (from outputs/)
"""


from statistics import mean


TERMINAL_EVENTS = {'complete': 'completed', 'cancel': 'cancelled', 'timeout': 'timed_out', 'fail': 'failed'}
TRANSITIONS = {
    'not_arrived': {'arrive': 'queued'},
    'queued': {'admit': 'preprocessing'},
    'preprocessing': {'input_ready': 'prefill'},
    'prefill': {'first_token': 'decode'},
    'decode': {'token': 'decode', 'complete': 'completed'},
}
TERMINAL_STATES = {'completed', 'cancelled', 'timed_out', 'failed'}


def replay_request_events(events):
    """回放事件并返回请求状态、关键时间戳与输出词元时间。"""
    if not events:
        raise ValueError('events 不能为空')

    state = 'not_arrived'
    previous_time = None
    timestamps = {}
    token_times = []
    failure_stage = None

    for item in events:
        event = item['event']
        timestamp = float(item['timestamp'])
        if previous_time is not None and timestamp < previous_time:
            raise ValueError('事件时间必须单调')
        if state in TERMINAL_STATES:
            raise ValueError('终止状态后不能继续追加事件')

        if event in {'cancel', 'timeout', 'fail'}:
            if state == 'not_arrived':
                raise ValueError('请求到达前不能终止')
            failure_stage = state
            state = TERMINAL_EVENTS[event]
            timestamps['terminal'] = timestamp
        else:
            # ==========================================
            # TODO 1：推进合法请求状态
            # ==========================================
            next_state = TRANSITIONS.get(state, {}).get(event)
            if next_state is None:
                raise ValueError(f'状态 {state} 不接受事件 {event}')
            state = next_state
            timestamp_key = {
                'arrive': 'arrival', 'admit': 'admitted',
                'input_ready': 'input_ready', 'first_token': 'first_token',
                'complete': 'terminal',
            }.get(event)
            if timestamp_key is not None:
                timestamps[timestamp_key] = timestamp
            if event in {'first_token', 'token'}:
                token_times.append(timestamp)
        previous_time = timestamp

    if state not in TERMINAL_STATES:
        raise ValueError('轨迹缺少终止事件')
    return {
        'status': state,
        'failure_stage': failure_stage,
        'timestamps': timestamps,
        'token_times': token_times,
    }


def compute_request_metrics(report):
    """从状态报告计算当前轨迹已经具备的延迟指标。"""
    timestamps = report['timestamps']
    token_times = report['token_times']

    def duration(start, end):
        return timestamps[end] - timestamps[start] if start in timestamps and end in timestamps else None

    # ==========================================
    # TODO 2：计算相邻词元间隔
    # ==========================================
    token_gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    tpot = mean(token_gaps) if token_gaps else None
    return {
        'queue_wait': duration('arrival', 'admitted'),
        'preprocessing': duration('admitted', 'input_ready'),
        'prefill': duration('input_ready', 'first_token'),
        'ttft': duration('arrival', 'first_token'),
        'tpot': tpot,
        'decode_time': duration('first_token', 'terminal'),
        'token_span': token_times[-1] - token_times[0] if token_times else None,
        'e2e': duration('arrival', 'terminal'),
        'streamed_tokens': len(token_times),
    }


def classify_request_outcome(report, metrics, dominance_threshold=0.45):
    """返回异常终止阶段，或完成请求的阶段占比与瓶颈标签。"""
    if report['status'] != 'completed':
        return {
            'outcome': report['status'],
            'stage': report['failure_stage'],
            'dominant_share': None,
        }
    if not (0.0 < dominance_threshold <= 1.0):
        raise ValueError('dominance_threshold 必须位于 (0, 1]')
    phase_durations = {
        'queue': metrics['queue_wait'],
        'preprocessing': metrics['preprocessing'],
        'prefill': metrics['prefill'],
        'decode': metrics['decode_time'],
    }
    available = {name: value for name, value in phase_durations.items() if value is not None}
    # ==========================================
    # TODO 3：按阶段占比归因请求
    # ==========================================
    explained_ms = sum(available.values())
    if explained_ms <= 0:
        raise ValueError('可解释阶段时长必须为正')
    phase_shares = {stage: elapsed / explained_ms for stage, elapsed in available.items()}
    dominant_stage, dominant_share = max(phase_shares.items(), key=lambda pair: pair[1])
    return {
        'outcome': 'completed',
        'stage': dominant_stage if dominant_share >= dominance_threshold else 'mixed',
        'dominant_share': dominant_share,
    }


def summarize_workload(request_results):
    """汇总 ``(report, metrics)`` 列表并显式返回统计分母。"""
    total_requests = len(request_results)
    # ==========================================
    # TODO 4：固定工作负载统计分母
    # ==========================================
    completed_count = len([report for report, _ in request_results if report['status'] == 'completed'])
    ttft_values = [metrics['ttft'] for _, metrics in request_results if metrics['ttft'] is not None]
    return {
        'total_requests': total_requests,
        'completed_requests': completed_count,
        'success_rate': completed_count / total_requests if total_requests else None,
        'ttft_sample_count': len(ttft_values),
        'mean_ttft': mean(ttft_values) if ttft_values else None,
    }


def _completed_events():
    """构造阶段耗时接近、没有明显主导瓶颈的完整请求轨迹。"""
    return [
        {'event': 'arrive', 'timestamp': 0.0},
        {'event': 'admit', 'timestamp': 2.0},
        {'event': 'input_ready', 'timestamp': 3.0},
        {'event': 'first_token', 'timestamp': 7.0},
        {'event': 'token', 'timestamp': 9.0},
        {'event': 'token', 'timestamp': 11.0},
        {'event': 'complete', 'timestamp': 12.0},
    ]


def test_completed_trace_and_metrics():
    """完整轨迹应恢复状态、指标和 TTFT 阶段可加关系。"""
    report = replay_request_events(_completed_events())
    metrics = compute_request_metrics(report)
    assert report['status'] == 'completed'
    assert report['token_times'] == [7.0, 9.0, 11.0]
    assert metrics == {
        'queue_wait': 2.0, 'preprocessing': 1.0, 'prefill': 4.0,
        'ttft': 7.0, 'tpot': 2.0, 'decode_time': 5.0,
        'token_span': 4.0, 'e2e': 12.0, 'streamed_tokens': 3,
    }
    assert metrics['ttft'] == metrics['queue_wait'] + metrics['preprocessing'] + metrics['prefill']
    assert metrics['e2e'] == metrics['ttft'] + metrics['decode_time']


def test_stage_share_attribution():
    """最大阶段占比不足阈值时应标记混合瓶颈。"""
    report = replay_request_events(_completed_events())
    outcome = classify_request_outcome(report, compute_request_metrics(report))
    assert outcome['outcome'] == 'completed'
    assert outcome['stage'] == 'mixed'
    assert abs(outcome['dominant_share'] - 5.0 / 12.0) < 1e-9

    prefill_heavy = _completed_events()
    prefill_heavy[3] = {'event': 'first_token', 'timestamp': 9.0}
    prefill_heavy[4] = {'event': 'token', 'timestamp': 10.0}
    prefill_heavy[5] = {'event': 'token', 'timestamp': 11.0}
    heavy_report = replay_request_events(prefill_heavy)
    heavy_outcome = classify_request_outcome(heavy_report, compute_request_metrics(heavy_report))
    assert heavy_outcome['stage'] == 'prefill'


def test_partial_trace_preserves_missing_metrics():
    """首词元前取消时，TTFT 保持为空但 E2E 仍可计算。"""
    events = _completed_events()[:2] + [{'event': 'cancel', 'timestamp': 5.0}]
    report = replay_request_events(events)
    metrics = compute_request_metrics(report)
    outcome = classify_request_outcome(report, metrics)
    assert metrics['queue_wait'] == 2.0
    assert metrics['ttft'] is None and metrics['e2e'] == 5.0
    assert outcome == {'outcome': 'cancelled', 'stage': 'preprocessing', 'dominant_share': None}


def test_failed_decode_keeps_streamed_tokens():
    """解码中超时仍应保留已经输出的词元与失败阶段。"""
    events = _completed_events()[:5] + [{'event': 'timeout', 'timestamp': 10.0}]
    report = replay_request_events(events)
    metrics = compute_request_metrics(report)
    outcome = classify_request_outcome(report, metrics)
    assert metrics['streamed_tokens'] == 2
    assert outcome['outcome'] == 'timed_out' and outcome['stage'] == 'decode'


def test_workload_denominators():
    """成功率与 TTFT 均值必须报告各自的统计分母。"""
    completed_report = replay_request_events(_completed_events())
    cancelled_report = replay_request_events(
        _completed_events()[:2] + [{'event': 'cancel', 'timestamp': 5.0}]
    )
    results = [
        (completed_report, compute_request_metrics(completed_report)),
        (cancelled_report, compute_request_metrics(cancelled_report)),
    ]
    summary = summarize_workload(results)
    assert summary['total_requests'] == 2
    assert summary['completed_requests'] == 1
    assert summary['success_rate'] == 0.5
    assert summary['ttft_sample_count'] == 1 and summary['mean_ttft'] == 7.0


def test_invalid_transitions():
    """非法起点、逆序时间和终止后事件都应被拒绝。"""
    cases = [
        [{'event': 'admit', 'timestamp': 0.0}],
        [{'event': 'cancel', 'timestamp': 0.0}],
        [{'event': 'arrive', 'timestamp': 1.0}, {'event': 'admit', 'timestamp': 0.5}],
        _completed_events() + [{'event': 'token', 'timestamp': 13.0}],
    ]
    for events in cases:
        try:
            replay_request_events(events)
            raise AssertionError('非法事件序列应被拒绝')
        except ValueError:
            pass


test_completed_trace_and_metrics()
test_stage_share_attribution()
test_partial_trace_preserves_missing_metrics()
test_failed_decode_keeps_streamed_tokens()
test_workload_denominators()
test_invalid_transitions()
print('✅ 请求生命周期机制测试通过')


import platform
from datetime import datetime
from zoneinfo import ZoneInfo
print('INF-REQUEST-LIFECYCLE | Task0 | GitHub: lih42534-wq')
print('Executed:', datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'))
print('Environment:', platform.platform(), '| Python', platform.python_version(), '| CPU only')
print('Source: DataWhale / llm-algo-leetcode / INF-REQUEST-LIFECYCLE')
print()
tests = [test_completed_trace_and_metrics, test_stage_share_attribution,
         test_partial_trace_preserves_missing_metrics, test_failed_decode_keeps_streamed_tokens,
         test_workload_denominators, test_invalid_transitions]
for test in tests:
    test()
    print('[PASS]', test.__name__)
print('Original tests: 6 / 6 passed')
print()
report = replay_request_events(_completed_events())
metrics = compute_request_metrics(report)
print('Synthetic timestamps interpreted as milliseconds (not a model benchmark):')
print('Queue =', metrics['queue_wait'], 'ms; preprocessing =', metrics['preprocessing'], 'ms')
print('Prefill =', metrics['prefill'], 'ms; TTFT =', metrics['ttft'], 'ms')
print('TPOT =', metrics['tpot'], 'ms/token; E2E =', metrics['e2e'], 'ms')
print('Attribution:', classify_request_outcome(report, metrics))
cancelled = replay_request_events(_completed_events()[:2] + [{'event': 'cancel', 'timestamp': 5.0}])
print('Completed + cancelled workload:', summarize_workload([(report, metrics), (cancelled, compute_request_metrics(cancelled))]))
print()
# Additional meaningful boundary checks: one token, empty workload, zero-duration attribution.
one = replay_request_events(_completed_events()[:4] + [{'event': 'complete', 'timestamp': 8.0}])
assert compute_request_metrics(one)['tpot'] is None
assert summarize_workload([])['success_rate'] is None
assert summarize_workload([])['mean_ttft'] is None
zero = replay_request_events([dict(item, timestamp=0.0) for item in _completed_events()])
try:
    classify_request_outcome(zero, compute_request_metrics(zero))
except ValueError:
    pass
else:
    raise AssertionError('zero-duration attribution must be rejected')
print('Additional boundary checks: 3 / 3 passed')
print('ALL CHECKS PASSED')


import torch
print("微信群昵称：旅行包 | GitHub: lih42534-wq")
print("PyTorch", torch.__version__, "| device: CPU | macOS / Apple Silicon")
# 事件时间戳统一按 ms 解释；输出时间为 7、9、11 ms。
token_times = torch.tensor(report['token_times'], dtype=torch.float64)
phase_times = torch.tensor([metrics[k] for k in
    ['queue_wait', 'preprocessing', 'prefill', 'decode_time']], dtype=torch.float64)
ttft = token_times[0] - report['timestamps']['arrival']
tpot = torch.diff(token_times).mean()
shares = phase_times / phase_times.sum()
assert ttft.item() == metrics['ttft']
assert tpot.item() == metrics['tpot']
assert torch.isclose(shares.sum(), torch.tensor(1.0, dtype=torch.float64))
assert abs(shares.max().item() - 5 / 12) < 1e-9
print(f"TTFT = {ttft.item():.1f} ms | TPOT = {tpot.item():.1f} ms/token")
print("Phase shares [queue, preprocessing, prefill, decode]:")
print([round(x, 4) for x in shares.tolist()])
print("6 original tests + 3 boundary checks + PyTorch cross-check: ALL PASSED")
print("Synthetic event replay; not a measured LLM benchmark.")
