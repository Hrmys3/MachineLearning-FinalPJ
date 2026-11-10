"""从冻结配置和逐步日志复算H对照；不修改原始运行结果。"""
import argparse
import csv
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import yaml
from scipy.stats import binomtest
from mlpj.lift.eval.lift_protocol import FirstLiftMetric
from mlpj.lift.eval.run_act_validation import lift_step_summary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('outputs/lift/h_compare_v1'))
args = parser.parse_args()
fields = ['success_any', 'stable_native_success_10_steps', 'success_final', 'first_lift_hold_success']
configs, metadata, results, summary = [], [], {}, {}
for h in (1, 2, 5):
    root = args.input / f'h{h}'
    cfg = yaml.safe_load((root / 'parameters.yaml').read_text())
    assert cfg['eval']['execution_horizon'] == h
    normalized = deepcopy(cfg)
    normalized['eval'].pop('execution_horizon')
    configs.append(normalized)
    meta = json.loads((root / 'config.json').read_text())
    metadata.append({k: meta[k] for k in ['checkpoint_sha256', 'random_set_sha256', 'splits_sha256', 'episode_ids', 'evaluation_source_snapshot', 'prediction_horizon']})
    metrics = json.loads((root / 'metrics.json').read_text())
    assert metrics['completed'] == metrics['n'] == 100 and metrics['process_failures'] == 0
    results[h] = {}
    for i in meta['episode_ids']:
        ep = root / f'episode_{i:04d}'
        result = json.loads((ep / 'result.json').read_text())
        rows = list(csv.DictReader((ep / 'steps.csv').open()))
        assert len(rows) == cfg['eval']['steps'] == result['steps']
        native = [bool(int(row['native_success'])) for row in rows]
        metric = FirstLiftMetric(cfg['eval']['metric_confirmation_steps'], cfg['eval']['metric_hold_seconds'], cfg['eval']['control_freq'])
        for success in native:
            metric.update(success)
        recalculated = metric.result()
        assert all(result[k] == v for k, v in recalculated.items())
        streak = longest = 0
        for success in native:
            streak = streak + 1 if success else 0
            longest = max(longest, streak)
        assert result['success_any'] == any(native)
        assert result['success_final'] == native[-1]
        assert result['stable_native_success_10_steps'] == (longest >= 10)
        results[h][i] = result
    counts = {field: sum(r[field] for r in results[h].values()) for field in fields}
    assert all(counts[k] == metrics['counts'][k] for k in fields)
    intervals = {}
    for field, count in counts.items():
        # Wilson 95% interval; descriptive uncertainty under an iid-state approximation.
        n = 100; z = 1.95996398454; p = count / n
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
        intervals[field] = [center-half, center+half]
    perf = json.loads((root / 'performance.json').read_text())
    lifts = [r['first_lift_step'] for r in results[h].values() if r['first_lift_step'] is not None]
    summary[h] = dict(lift_latency={field: lift_step_summary(list(results[h].values()), field, 100, cfg['eval']['control_freq']) for field in ['first_success_step', 'first_lift_step', 'first_lift_confirmation_step']}, counts=counts, wilson95=intervals, hold_status=metrics['hold_status_counts'],
                      total_seconds=perf['script_wall_seconds'], batch_seconds=perf['inference_batch_seconds'],
                      grid_seconds=perf['grid_seconds'], mean_lift_step=float(np.mean(lifts)), median_lift_step=float(np.median(lifts)))
assert configs[0] == configs[1] == configs[2], 'H以外参数不同'
assert metadata[0] == metadata[1] == metadata[2], '资源、轮次或源码快照不同'
paired = {}
for h, baseline in [(1,5),(2,5),(2,1)]:
    paired[f'h{h}_vs_h{baseline}'] = {}
    for field in fields:
        gain = [i for i in results[h] if results[h][i][field] and not results[baseline][i][field]]
        loss = [i for i in results[h] if not results[h][i][field] and results[baseline][i][field]]
        discordant = len(gain)+len(loss)
        pvalue = binomtest(len(gain), discordant, .5).pvalue if discordant else 1.
        paired[f'h{h}_vs_h{baseline}'][field] = dict(gained=gain, lost=loss, delta_percentage_points=len(gain)-len(loss), mcnemar_exact_p=pvalue)
report = dict(input=str(args.input), configuration_and_snapshot_match=True, summaries=summary, paired=paired,
              note='每组单次100固定开发初态；配对精确McNemar检验未作多重比较校正。计时单次，不能推断稳定加速比。')
print(json.dumps(report, indent=2, ensure_ascii=False))
