"""顺序运行相同任务的性能对照；生成独立配置/输出并汇总速度与结果差异。"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT/'scripts'))
from project_config import load_config
import yaml

TRAIN = {'baseline', 'cache', 'workers', 'amp', 'compile', 'fast'}
EVAL = {'baseline', 'reuse', 'parallel', 'inference', 'amp', 'compile', 'no_video', 'fast'}


def variant_config(base, kind, name):
    config = copy.deepcopy(base)
    t, e = config['train'], config['eval']
    config['performance']['profiler_enabled'] = False
    if kind == 'train':
        t.update(num_workers=0, persistent_workers=False, pin_memory=False, cache_mode='none',
                 non_blocking_transfer=False, precision='fp32', compile_model=False)
        if name in {'cache', 'fast'}:
            t['cache_mode'] = 'processed_ram'
        if name == 'workers':
            t.update(num_workers=2, persistent_workers=True, pin_memory=True, non_blocking_transfer=True)
        if name in {'amp', 'fast'}:
            t['precision'] = 'amp_fp16'
        if name == 'compile':
            t['compile_model'] = True
    else:
        e.update(num_workers=1, episodes_per_worker=1, inference_mode=False, precision='fp32', compile_model=False)
        if name in {'reuse', 'fast', 'compile'}:
            e['episodes_per_worker'] = 5
        if name in {'parallel', 'fast'}:
            e['num_workers'] = 2
        if name in {'inference', 'fast'}:
            e['inference_mode'] = True
        if name == 'amp':
            e['precision'] = 'amp_fp16'
        if name == 'compile':
            e['compile_model'] = True
        if name == 'no_video':
            config['video'].update(save_rollout=False, auto_make_grids=False)
    return config


def compare_episodes(baseline, candidate):
    a = {p.parent.name: json.loads(p.read_text()) for p in Path(baseline).glob('episode_*/result.json')}
    b = {p.parent.name: json.loads(p.read_text()) for p in Path(candidate).glob('episode_*/result.json')}
    fields = ['success_any', 'stable_native_success_10_steps', 'success_final', 'first_lift_hold_success']
    changes = {key: [ep for ep in sorted(a.keys() & b.keys()) if a[ep][key] != b[ep][key]] for key in fields}
    action_delta = height_delta = 0.0
    for ep in a.keys() & b.keys():
        with (Path(baseline)/ep/'steps.csv').open() as f:
            rows_a = list(csv.DictReader(f))
        with (Path(candidate)/ep/'steps.csv').open() as f:
            rows_b = list(csv.DictReader(f))
        if len(rows_a) != len(rows_b):
            raise ValueError('对照步数不一致')
        for x, y in zip(rows_a, rows_b):
            action_delta = max(action_delta, *(abs(float(x[f'action_{i}'])-float(y[f'action_{i}'])) for i in range(7)))
            height_delta = max(height_delta, abs(float(x['cube_z_after'])-float(y['cube_z_after'])))
    return {'same_episode_set': a.keys() == b.keys(), 'paired_metric_changes': changes,
            'max_action_abs_difference': action_delta, 'max_cube_z_difference_m': height_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/lift.yaml'))
    parser.add_argument('--kind', choices=['train', 'eval'], required=True)
    parser.add_argument('--policy', choices=['act', 'act_hold'], help='所有评测方案共同采用的策略')
    parser.add_argument('--variants', help='逗号分隔；训练默认baseline,cache，评测默认baseline,reuse')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='所有方案共同关闭视频，隔离推理与编码开销')
    parser.add_argument('--diagnostics', action='store_true', help='所有方案共同启用CUDA事件和硬件采样')
    args = parser.parse_args()
    variants = (args.variants or ('baseline,cache' if args.kind == 'train' else 'baseline,reuse')).split(',')
    allowed = TRAIN if args.kind == 'train' else EVAL
    if 'baseline' not in variants or len(set(variants)) != len(variants) or not set(variants) <= allowed:
        parser.error(f'variants须含baseline且无重复，可选{sorted(allowed)}')
    if min(args.repeats, args.updates, args.episodes, args.torch_threads) < 1:
        parser.error('次数须为正整数')
    base = load_config(args.config)
    if args.policy is not None:
        base['eval']['policy'] = args.policy
    if args.no_video:
        base['video'].update(save_rollout=False, auto_make_grids=False)
    if args.diagnostics:
        base['performance'].update(cuda_events=True, hardware_monitor_enabled=True)
    args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(args.repeats):
        # 交替顺序减轻首次加载/文件缓存偏差，始终顺序执行避免互相抢资源。
        for name in (variants if repeat % 2 == 0 else list(reversed(variants))):
            config = variant_config(base, args.kind, name)
            config['performance']['torch_threads'] = args.torch_threads
            path = args.output/f'{name}_{repeat+1:02d}'
            if args.kind == 'train':
                config['train']['updates'] = args.updates
                config['train']['output'] = str(path)
            else:
                config['eval']['episode_limit'] = args.episodes
            config_path = path.with_suffix('.yaml')
            config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            load_config(config_path)
            command = [sys.executable, 'scripts/train/train_act_lift20.py' if args.kind == 'train'
                       else 'scripts/eval/run_act_validation.py', '--config', str(config_path)]
            if args.kind == 'eval':
                command += ['--output', str(path)]
            record = {'variant': name, 'repeat': repeat+1, 'output': str(path), 'command': command}
            if not args.prepare_only:
                start = time.perf_counter()
                with path.with_suffix('.log').open('w') as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                record.update(returncode=process.returncode, wall_seconds=time.perf_counter()-start)
                if (path/'performance.json').exists():
                    record['performance'] = json.loads((path/'performance.json').read_text())
                print(name, repeat+1, 'exit', process.returncode, 'seconds', round(record['wall_seconds'], 3), flush=True)
            runs.append(record)
            (args.output/'runs.json').write_text(json.dumps(runs, indent=2))
    report = {'kind': args.kind, 'runs': runs, 'comparisons': {},
              'note': 'wall_seconds包含启动、预处理/缓存、验证、保存/视频；模型计算收益另看阶段时间。'}
    if not args.prepare_only:
        for name in variants:
            selected = [r for r in runs if r['variant'] == name]
            baseline = [r for r in runs if r['variant'] == 'baseline']
            complete = all(r['returncode'] == 0 and 'performance' in r for r in selected+baseline)
            comparison = {'all_runs_completed': complete}
            if complete:
                comparison.update(median_wall_seconds=statistics.median(r['wall_seconds'] for r in selected),
                    wall_speedup=statistics.median(r['wall_seconds'] for r in baseline)/statistics.median(r['wall_seconds'] for r in selected))
                if args.kind == 'eval':
                    comparison['quality_by_repeat'] = [compare_episodes(next(b['output'] for b in baseline if b['repeat']==r['repeat']), r['output']) for r in selected]
                else:
                    comparison['validation_l1_by_repeat'] = [r['performance']['validation_l1'] for r in selected]
                    comparison['steady_samples_per_second'] = [r['performance']['steady_samples_per_second'] for r in selected]
            report['comparisons'][name] = comparison
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print('对照汇总:', (args.output/'comparison.json').resolve())
    if not args.prepare_only and any(r['returncode'] for r in runs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
