"""性能优化正确性：缓存样本等价、配置约束和计时输出。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mlpj.common.paths import PROJECT_ROOT as ROOT
from mlpj.lift.config import load_config
from mlpj.common.performance import Timings, summary
import yaml


class SpeedTests(unittest.TestCase):
    def test_video_stride_preserves_timeline(self):
        from PIL import Image
        from mlpj.lift.eval.rollout_video import save_rollout_gif
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.gif'
            save_rollout_gif([Image.new('RGB', (8,8), c) for c in ['red','blue','green']], [0,2,4], path)
            with Image.open(path) as gif:
                total = 0
                for i in range(gif.n_frames):
                    gif.seek(i)
                    total += gif.info['duration']
                self.assertEqual(total, 250)

    def test_timer_and_percentile(self):
        timer = Timings()
        with timer.stage('work'):
            sum(range(100))
        self.assertEqual(timer.report()['wall_stages']['work']['count'], 1)
        self.assertAlmostEqual(summary([1., 2., 3.])['p95_seconds'], 2.9)

    def test_cache_worker_conflict(self):
        config = load_config(ROOT/'configs/lift/default.yaml')
        config['train'].update(cache_mode='raw_ram', num_workers=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_worker_crash_keeps_prior_results_and_failure_denominator(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from mlpj.lift.eval import run_act_validation as runner
        config = load_config(ROOT/'configs/lift/default.yaml')
        config['eval']['episodes_per_worker'] = 2
        config['video']['save_rollout'] = False
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'batch'
            args = SimpleNamespace(settings=config, output=output, checkpoint=Path('unused.pt'),
                                   policy='act', steps=100, initial_states='random')
            def launch(command, **kwargs):
                indices = [int(v) for v in command[command.index('--worker-indices')+1].split(',')]
                ep = 10 + indices[0]
                target = output/f'episode_{ep:04d}'
                target.mkdir()
                result = {k: True for k in ['success_any','stable_native_success_10_steps',
                          'success_final','first_lift_hold_success','first_lift_hold_2s_success']}
                result.update(episode_id=ep, steps=100, first_lift_hold_status='success')
                (target/'result.json').write_text(json.dumps(result))
                return SimpleNamespace(returncode=-6 if len(indices)==2 else 0)
            checkpoint = {'model_args': {'num_queries':20}, 'camera_names':['agentview','robot0_eye_in_hand'], 'image_size':[128,128]}
            with patch.object(runner, 'parse_args', return_value=args), \
                 patch.object(runner, 'evaluation_ids', return_value=[10,11,12]), \
                 patch.object(runner, 'experiment_config', return_value={'training_seed':42}), \
                 patch.object(torch, 'load', return_value=checkpoint), \
                 patch.object(runner.subprocess, 'run', side_effect=launch), patch('builtins.print'):
                with self.assertRaises(SystemExit):
                    runner.main()
            result = json.loads((output/'metrics.json').read_text())
            self.assertEqual(result['completed'], 2)
            self.assertEqual(result['process_failures'], 1)
            self.assertEqual(result['worker_failures'], 1)
            self.assertAlmostEqual(result['rates']['success_any'], 2/3)

    @unittest.skipUnless((ROOT/'data/lift/lift_act/episodes').exists(), 'requires local cache')
    def test_cached_samples_equal_original(self):
        import torch
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            path = ROOT/'src/mlpj/lift/train/train_act_lift20.py'
            spec = importlib.util.spec_from_file_location('train_speed_test', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            splits = json.loads((ROOT/'data/lift/lift_act/splits.json').read_text())
            stats = json.loads((ROOT/'data/lift/lift_act/stats_train_20.json').read_text())
            ids = splits['train_20'][:1]
            original = module.LiftDataset(ids, stats)
            for mode in ['raw_ram', 'processed_ram']:
                cached = module.LiftDataset(ids, stats, cache_mode=mode)
                for index in [0, len(original)//2, len(original)-1]:
                    for a, b in zip(original[index], cached[index]):
                        self.assertTrue(torch.equal(a, b), (mode, index))
        finally:
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
