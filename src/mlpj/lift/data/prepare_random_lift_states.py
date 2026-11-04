"""生成并逐条恢复校验固定随机开发初态集，不执行策略、不筛选成功样本。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from mlpj.common.paths import PROJECT_ROOT as ROOT
os.chdir(ROOT)
from mlpj.lift.config import load_config
import h5py
import numpy as np
import robosuite as suite

SOURCE = Path('data/lift/robomimic_source/v1.5/lift/ph/demo_v15.hdf5')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/lift/default.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config['random_set']
    output = Path(spec['path'])
    manifest = output.with_suffix('.json')
    if output.exists() or manifest.exists():
        raise FileExistsError(f'{output} 或清单已存在；修改路径建立新版本')
    with h5py.File(SOURCE) as f:
        metadata = json.loads(f['data'].attrs['env_args'])
    kwargs = metadata['env_kwargs'].copy()
    kwargs.update(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                  lite_physics=False, hard_reset=False)
    np.random.seed(spec['seed'])
    env = suite.make(metadata['env_name'], **kwargs)
    try:
        env.reset()
        xml = env.sim.model.get_xml()
        seeds = np.random.default_rng(spec['seed']).choice(2**31, size=spec['count'], replace=False)
        states, cube_positions, eef_positions = [], [], []
        for seed in seeds:
            np.random.seed(int(seed))
            obs = env.reset()
            state = env.sim.get_state().flatten().copy()
            assert np.isfinite(state).all() and not env._check_success()
            states.append(state)
            cube_positions.append(obs['cube_pos'].copy())
            eef_positions.append(obs['robot0_eef_pos'].copy())
        states = np.asarray(states)
        assert len(np.unique(states, axis=0)) == spec['count']
        # 在重建XML之后检查每一个初态；不是只确认文件可读。
        env.reset_from_xml_string(env.edit_model_xml(xml))
        for state, cube in zip(states, cube_positions):
            env.sim.reset()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.robots[0].reset(deterministic=True)
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env._update_observables(force=True)
            assert np.allclose(env.sim.get_state().flatten(), state, atol=1e-12, rtol=0)
            assert np.allclose(env._get_observations()['cube_pos'], cube, atol=1e-12, rtol=0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'x') as f:
            f.attrs['model_xml'] = xml
            f.attrs['env_args'] = json.dumps(metadata)
            f.attrs['purpose'] = 'development; not heldout; fixed geometry, native reset sampling'
            f.attrs['generation_seed'] = spec['seed']
            f.create_dataset('states', data=states)
            f.create_dataset('seeds', data=seeds)
            f.create_dataset('cube_positions', data=cube_positions)
            f.create_dataset('eef_positions', data=eef_positions)
        info = {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'count': len(states), 'seed': spec['seed'], 'episode_ids': list(range(len(states))),
                'purpose': 'development, not independent final test',
                'distribution': 'fixed model sampled once with master seed; native Lift reset: cube xy in [-0.03,0.03]m, yaw in [0,2pi); source robot initialization noise',
                'source': str(SOURCE), 'env_args': metadata,
                'code_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'config': config, 'generator_source': Path(__file__).read_text(),
                'versions': {k: importlib.metadata.version(k) for k in ['robosuite','mujoco','numpy']},
                'checks': 'all states finite, unique, initially unsuccessful; all XML/state restorations checked',
                'cube_xyz_min': np.min(cube_positions,axis=0).tolist(),
                'cube_xyz_max': np.max(cube_positions,axis=0).tolist()}
        manifest.write_text(json.dumps(info, indent=2))
        print(json.dumps({k:info[k] for k in ['path','sha256','count','seed','checks']}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
