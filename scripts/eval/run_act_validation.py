"""固定初态批量评测；可选有限轮次模型复用/多worker，保留失败分母。"""
import os
from pathlib import Path
import json
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_STARTED = time.perf_counter()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_PROJECT_ROOT)
from lift_protocol import parse_args, experiment_config, evaluation_ids
from make_eval_grid import make_grids
from performance import HardwareMonitor


def write_metrics(root, settings, statuses, results, n):
    fields = ['success_any', 'stable_native_success_10_steps', 'success_final', 'first_lift_hold_success']
    if settings['metric_hold_seconds'] == 2.0:
        fields.append('first_lift_hold_2s_success')
    counts = {field: sum(bool(r[field]) for r in results) for field in fields}
    metrics = {'hold_required_seconds': settings['metric_hold_seconds'], 'n': n,
               'attempted': len(statuses), 'completed': len(results),
               'process_failures': sum(not s['completed'] for s in statuses),
               'worker_failures': len({s['worker_log'] for s in statuses if s['returncode'] != 0}),
               'counts': counts, 'rates': {k: v/n for k, v in counts.items()},
               'hold_status_counts': {state: sum(r['first_lift_hold_status'] == state for r in results)
                    for state in ['success', 'height_condition_lost', 'insufficient_window', 'no_confirmed_lift']}}
    (root/'metrics.json').write_text(json.dumps(metrics, indent=2))
    (root/'process_status.json').write_text(json.dumps(sorted(statuses, key=lambda s: s['episode_id']), indent=2))


def main():
    args = parse_args(batch=True)
    ids = evaluation_ids(args)
    settings, perf, video = (args.settings[k] for k in ['eval', 'performance', 'video'])
    if settings['episode_limit'] is not None:
        ids = ids[:settings['episode_limit']]
    config = experiment_config(args)
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    prediction_horizon = checkpoint['model_args']['num_queries']
    if settings['execution_horizon'] > prediction_horizon:
        raise ValueError('执行长度不能超过checkpoint预测长度')
    if checkpoint['camera_names'] != ['agentview', 'robot0_eye_in_hand']:
        raise ValueError('checkpoint相机顺序不匹配')
    config.update(prediction_horizon=prediction_horizon,
                  inference_image_size=checkpoint['image_size'],
                  training_seed=checkpoint.get('training_seed', config['training_seed']), episode_ids=ids)
    del checkpoint
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    frozen = args.output/'parameters.yaml'
    frozen.write_text(yaml.safe_dump(args.settings, sort_keys=False, allow_unicode=True))
    groups = [list(range(i, min(i+settings['episodes_per_worker'], len(ids))))
              for i in range(0, len(ids), settings['episodes_per_worker'])]
    statuses, results, worker_times = [], [], []
    env = os.environ.copy()
    env.update(MUJOCO_GL=env.get('MUJOCO_GL', 'glfw'), ACT_PRETRAINED='0', PYTHONUNBUFFERED='1')

    def launch(indices):
        command = [sys.executable, '-u', str(_PROJECT_ROOT/'scripts/eval/eval_act_one.py'),
                   '--worker-indices', ','.join(map(str, indices)), '--policy', args.policy,
                   '--output', str(args.output), '--checkpoint', str(args.checkpoint),
                   '--steps', str(args.steps), '--config', str(frozen), '--initial-states', args.initial_states]
        log_name = f'episode_{ids[indices[0]]:04d}.log' if len(indices) == 1 else f'worker_{indices[0]:04d}.log'
        start = time.perf_counter()
        with (args.output/log_name).open('w') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        return indices, process.returncode, time.perf_counter()-start, log_name

    batch_start = time.perf_counter()
    with HardwareMonitor(args.output/'hardware.csv', perf['hardware_monitor_enabled'], perf['hardware_sample_seconds']):
        with ThreadPoolExecutor(max_workers=settings['num_workers']) as pool:
            futures = [pool.submit(launch, indices) for indices in groups]
            for future in as_completed(futures):
                indices, returncode, elapsed, log_name = future.result()
                worker_times.append(elapsed)
                for index in indices:
                    episode_id = ids[index]
                    result_path = args.output/f'episode_{episode_id:04d}'/'result.json'
                    completed = False
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        # 某轮完成后同组后续轮崩溃，不丢弃先前已保存的有效结果。
                        if result['episode_id'] == episode_id and result['steps'] == args.steps:
                            results.append(result)
                            completed = True
                    status = {'episode_id': episode_id, 'returncode': returncode,
                              'result_exists': result_path.exists(), 'completed': completed,
                              'worker_log': log_name}
                    statuses.append(status)
                    print(status, flush=True)
                write_metrics(args.output, settings, statuses, results, len(ids))
        inference_seconds = time.perf_counter()-batch_start
        grid_start = time.perf_counter()
        grid_error = None
        if video['save_rollout'] and video['auto_make_grids']:
            try:
                make_grids(args.output)
            except Exception as exc:
                grid_error = str(exc)
                (args.output/'grid_error.txt').write_text(grid_error)
        grid_seconds = time.perf_counter()-grid_start
    (args.output/'performance.json').write_text(json.dumps({
        'kind': 'batch', 'parameters': args.settings, 'episode_ids': ids,
        'planned': len(ids), 'completed': len(results),
        'inference_batch_seconds': inference_seconds, 'grid_seconds': grid_seconds,
        'completed_episodes_per_minute': len(results)*60/inference_seconds,
        'worker_wall_seconds': worker_times, 'script_wall_seconds': time.perf_counter()-_SCRIPT_STARTED,
        'note': 'inference_batch包含子进程启动、模型加载、环境、逐轮GIF；grid另列。'}, indent=2))
    print('结果目录:', args.output.resolve())
    if grid_error or any(not s['completed'] or s['returncode'] != 0 for s in statuses):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
