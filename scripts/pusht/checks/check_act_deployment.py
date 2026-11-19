"""对照项目数据推理路径和固定XPolicyLab真实ACT部署适配器。"""
import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--gpu-uuid', required=True)
parser.add_argument('--episode', type=int, default=0)
parser.add_argument('--timesteps', type=int, nargs='+', default=[12, 179, 358])
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
args.checkpoint_dir = args.checkpoint_dir.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
os.environ['ACT_ACTION_DIM'] = '14'
UPSTREAM = ROOT / 'third_party/pusht/RoboDojo/XPolicyLab'
sys.path[:0] = [str(ROOT / 'src'), str(UPSTREAM.parent), str(UPSTREAM / 'policy/ACT'), str(UPSTREAM)]

import h5py
import numpy as np
import torch
import yaml
from detr.act_policy import ACTPolicy
from XPolicyLab.policy.ACT.model import Model
from XPolicyLab.utils.process_data import decode_image_bit, pack_robot_state
from mlpj.pusht.act_data import ACT_CAMERA_NAMES, compute_norm_stats, load_joint_episodes
from mlpj.pusht.act_dataset import PushTACTDataset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


result = {'ok': False, 'scope': 'offline ACT deployment equivalence; temporal aggregation disabled', 'frames': []}
started = time.perf_counter()
try:
    torch.set_num_threads(4)
    # 严格数值对照显式禁用TF32，避免不同卷积内存布局选择不同近似核。
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = yaml.safe_load((args.checkpoint_dir / 'parameters.yaml').read_text())
    old_result = json.loads((args.checkpoint_dir / 'result.json').read_text())
    checkpoint = args.checkpoint_dir / 'policy_last.ckpt'
    stats_path = args.checkpoint_dir / 'dataset_stats.pkl'
    assert old_result['ok'] and digest(checkpoint) == old_result['checkpoint_sha256']
    assert frozen['model']['camera_names'] == list(ACT_CAMERA_NAMES)
    for name, expected in frozen['source_sha256'].items():
        assert digest(ROOT / 'src/mlpj/pusht' / name) == expected, f'preprocessing source changed: {name}'
    if not (Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth').is_file():
        raise FileNotFoundError('ResNet18 cache missing')

    # 这里只读取本机本次训练产生的可信统计量pickle。
    with stats_path.open('rb') as file:
        stats = pickle.load(file)
    base = ROOT / 'data/pusht/robodojo_demos/1127bda94f17/data/RoboDojo/push_T/arx_x5/data'
    data_path = base / f'episode_{args.episode:07d}.hdf5'
    assert digest(data_path) == frozen['data_sha256'][args.episode]
    episode_ids = sorted(set(frozen['stats_episode_ids'] + [args.episode]))
    episodes = load_joint_episodes(base, episode_ids)
    recomputed = compute_norm_stats(episodes, frozen['stats_episode_ids'])
    assert all(np.array_equal(stats[k], recomputed[k]) for k in recomputed)
    assert all(0 <= t < len(episodes[args.episode][0]) for t in args.timesteps)
    dataset = PushTACTDataset(base, episodes, [args.episode], stats, horizon=frozen['model']['chunk_size'])

    config = dict(frozen['model'])
    config.update(ckpt_dir=str(args.checkpoint_dir), device='cuda:0', temporal_agg=False,
                  env_cfg_type='arx_x5', action_type='joint', action_dim=14)
    protocol = {'checkpoint_dir': str(args.checkpoint_dir), 'checkpoint_sha256': digest(checkpoint),
                'stats_sha256': digest(stats_path), 'parameters_sha256': digest(args.checkpoint_dir / 'parameters.yaml'),
                'episode': args.episode, 'timesteps': args.timesteps, 'model': config,
                'rtol': 1e-5, 'atol': 1e-6, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'torch': str(torch.__version__),
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'training_head': frozen['head'],
                'xpolicylab_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()}
    assert protocol['xpolicylab_head'] == frozen['xpolicylab_head']
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    source_dir = out / 'sources'
    source_dir.mkdir()
    shutil.copy2(__file__, source_dir / Path(__file__).name)
    for path in [ROOT / 'src/mlpj/pusht/act_data.py', ROOT / 'src/mlpj/pusht/act_dataset.py',
                 UPSTREAM / 'policy/ACT/model.py', UPSTREAM / 'policy/ACT/detr/act_policy.py',
                 UPSTREAM / 'utils/process_data.py']:
        shutil.copy2(path, source_dir / path.name)
    (out / 'xpolicylab.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=UPSTREAM, text=True))

    # 所有CUDA操作必须在实时UUID/空闲核验之后。
    assert not torch.cuda.is_initialized()
    before = subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    (out / 'gpu-before.txt').write_text(before)
    index, uuid, used, util = [v.strip() for v in before.strip().split(',')]
    if uuid != args.gpu_uuid or int(used) >= 10 or int(util) != 0:
        raise RuntimeError(f'GPU not idle: {before}')
    result.update(gpu_uuid=uuid, physical_index=int(index), cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    direct = ACTPolicy(config, model_cfg=Namespace(**config))
    direct.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    direct.eval()
    adapter = Model(dict(config))
    assert all(torch.equal(value, adapter.model.policy.state_dict()[key]) for key, value in direct.state_dict().items())
    result['weights_exact_equal'] = True
    assert adapter.camera_names == list(ACT_CAMERA_NAMES)
    assert adapter.robot_action_dim_info == {'arm_dim': [6, 6], 'ee_dim': [1, 1]}
    assert all(np.array_equal(stats[k], adapter.model.stats[k]) for k in stats)

    for timestep in args.timesteps:
        # 路径A：项目Dataset，包含标准化状态和CHW float32 RGB。
        images, qpos, _, _ = dataset.get_at(args.episode, timestep)
        with torch.no_grad():
            direct_chunk = direct(qpos.unsqueeze(0).cuda(), images.unsqueeze(0).cuda()).cpu().numpy()
        assert direct_chunk.shape == (1, config['chunk_size'], 14)
        direct_action = direct_chunk[0, 0] * stats['action_std'] + stats['action_mean']

        # 路径B：从原始HDF5独立构建与运行时相同的singular/color观测。
        with h5py.File(data_path, 'r') as file:
            observation = {'state': {}, 'vision': {}}
            for name in ['left_arm_joint_state', 'left_ee_joint_state',
                         'right_arm_joint_state', 'right_ee_joint_state']:
                observation['state'][name] = file[f'state/{name}s'][timestep]
            for camera in ACT_CAMERA_NAMES:
                observation['vision'][camera] = {'color': decode_image_bit(file[f'vision/{camera}/colors'][timestep])}
        adapter.reset()
        adapter.update_obs(observation)
        cached = adapter.model.obs_cache
        encoded_images = np.stack([cached[name] for name in ACT_CAMERA_NAMES]).astype(np.float32)
        encoded_qpos = adapter.model.pre_process(cached['qpos']).astype(np.float32)
        np.testing.assert_allclose(encoded_images, images.numpy(), rtol=0, atol=1e-7)
        np.testing.assert_allclose(encoded_qpos, qpos.numpy(), rtol=0, atol=1e-7)
        assert np.array_equal(encoded_images, images.numpy())
        assert np.array_equal(encoded_qpos, qpos.numpy())
        returned = adapter.get_action()
        assert len(returned) == 1
        action = returned[0]
        slices = {'left_arm_joint_state': (0, 6), 'left_ee_joint_state': (6, 7),
                  'right_arm_joint_state': (7, 13), 'right_ee_joint_state': (13, 14)}
        assert set(action) == set(slices)
        for key, (start, end) in slices.items():
            value = np.asarray(action[key])
            assert value.shape == (end - start,)
            assert np.isfinite(value).all()
        deployed = pack_robot_state({'state': action}, 'joint', adapter.robot_action_dim_info)
        cached_chunk = adapter.model.all_actions.detach().cpu().numpy()
        # 相同输入数值、相同权重，以适配器原生布局重算，定位CUDA布局差异。
        native_images = torch.from_numpy(np.stack([cached[name] for name in ACT_CAMERA_NAMES])).float().cuda().unsqueeze(0)
        native_qpos = torch.from_numpy(encoded_qpos).float().cuda().unsqueeze(0)
        with torch.no_grad():
            aligned_chunk = direct(native_qpos, native_images).cpu().numpy()
        np.testing.assert_allclose(cached_chunk, aligned_chunk, rtol=1e-5, atol=1e-6)
        native_chunk_pass = bool(np.allclose(cached_chunk, direct_chunk, rtol=1e-5, atol=1e-6))
        native_action_pass = bool(np.allclose(deployed, direct_action, rtol=1e-5, atol=1e-6))
        assert np.isfinite(deployed).all()
        grippers = deployed[[6, 13]]
        row = {'episode': args.episode, 'timestep': timestep,
               'image_input_bitwise_equal': bool(np.array_equal(encoded_images, images.numpy())),
               'qpos_input_bitwise_equal': bool(np.array_equal(encoded_qpos, qpos.numpy())),
               'project_image_strides': list(images.unsqueeze(0).stride()),
               'native_image_strides': list(native_images.stride()),
               'layout_aligned_chunk_max_abs_error': float(np.max(np.abs(cached_chunk - aligned_chunk))),
               'native_chunk_strict_pass': native_chunk_pass,
               'native_action_strict_pass': native_action_pass,
               'image_input_max_abs_error': float(np.max(np.abs(encoded_images - images.numpy()))),
               'qpos_input_max_abs_error': float(np.max(np.abs(encoded_qpos - qpos.numpy()))),
               'normalized_chunk_max_abs_error': float(np.max(np.abs(cached_chunk - direct_chunk))),
               'denormalized_first_action_max_abs_error': float(np.max(np.abs(deployed - direct_action))),
               'action': deployed.tolist(), 'arm_min': float(np.min(deployed[np.r_[0:6, 7:13]])),
               'arm_max': float(np.max(deployed[np.r_[0:6, 7:13]])),
               'gripper_raw_values': grippers.tolist(),
               'gripper_out_of_0_1': [bool(v < 0 or v > 1) for v in grippers]}
        result['frames'].append(row)
        np.savez(out / f'frame_{timestep:04d}.npz', direct_chunk=direct_chunk,
                 direct_action=direct_action, deployed_action=deployed, adapter_chunk=cached_chunk, layout_aligned_chunk=aligned_chunk)
        print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    strict_pass = all(r['native_chunk_strict_pass'] and r['native_action_strict_pass'] for r in result['frames'])
    result.update(ok=strict_pass, completed_frames=len(result['frames']),
                  checkpoint_sha256=protocol['checkpoint_sha256'], stats_sha256=protocol['stats_sha256'],
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2)
    if not strict_pass:
        raise AssertionError('Native layouts exceed strict tolerance; inspect saved layout-aligned diagnostics')
except BaseException as exc:
    result['error'] = repr(exc)
    raise
finally:
    result['elapsed_seconds'] = time.perf_counter() - started
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('EVIDENCE:', out, flush=True)
