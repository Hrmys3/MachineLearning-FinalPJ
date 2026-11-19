"""仅监听本机回环地址的真实ACT有界回合诊断服务。"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / 'third_party/pusht/RoboDojo/XPolicyLab'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    if not 1 <= args.max_actions <= 51:
        raise ValueError('max-actions must be between 1 and 51')
    args.output = args.output.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['ACT_ACTION_DIM'] = '14'
    for key in ('NO_PROXY', 'no_proxy'):
        os.environ[key] = ','.join(filter(None, [os.environ.get(key, ''), '127.0.0.1', 'localhost']))
    sys.path[:0] = [str(UPSTREAM.parent), str(UPSTREAM / 'policy/ACT'), str(UPSTREAM)]
    import numpy as np
    import torch
    import torchvision
    import websockets
    import yaml
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.policy.ACT.model import Model

    result = {'ok': False, 'scope': 'real ACT bounded WebSocket diagnostic', 'max_actions': args.max_actions,
              'forward_events': [],
              'resets': 0, 'observation_updates': 0, 'action_requests': 0}
    started = time.perf_counter()
    try:
        frozen = yaml.safe_load((args.checkpoint_dir / 'parameters.yaml').read_text())
        training_result = json.loads((args.checkpoint_dir / 'result.json').read_text())
        checkpoint = args.checkpoint_dir / 'policy_last.ckpt'
        assert training_result['ok'] and digest(checkpoint) == training_result['checkpoint_sha256']
        config = dict(frozen['model'])
        config.update(ckpt_dir=str(args.checkpoint_dir), device='cuda:0', temporal_agg=False,
                      env_cfg_type='arx_x5', action_type='joint', action_dim=14)
        protocol = {'checkpoint_sha256': digest(checkpoint),
                    'stats_sha256': digest(args.checkpoint_dir / 'dataset_stats.pkl'),
                    'max_actions': args.max_actions, 'model': config, 'gpu_uuid': args.gpu_uuid, 'cudnn_allow_tf32': False,
                    'matmul_allow_tf32': False, 'python': sys.version, 'torch': str(torch.__version__),
                    'torchvision': str(torchvision.__version__), 'websockets': websockets.__version__,
                    'xpolicylab_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()}
        assert protocol['xpolicylab_head'] == frozen['xpolicylab_head']
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        shutil.copy2(__file__, args.output / 'server_source.py')
        if not (Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth').is_file():
            raise FileNotFoundError('ResNet18 cache missing')
        torch.set_num_threads(4)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        assert not torch.cuda.is_initialized()
        before = subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        (args.output / 'gpu-before.txt').write_text(before)
        index, uuid, used, util = [v.strip() for v in before.strip().split(',')]
        if uuid != args.gpu_uuid or int(used) >= 10 or int(util) != 0:
            raise RuntimeError(f'GPU not idle: {before}')
        result.update(gpu_uuid=uuid, physical_index=int(index))
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        model = Model(config)

        def save_event(filename, row):
            with (args.output / filename).open('a') as file:
                file.write(json.dumps(row) + '\n')

        def on_forward(module, inputs):
            qpos, image = inputs[:2]
            row = {'request_index': result['action_requests'] + 1,
                   'model_t': model.model.t,
                   'observation_sequence': result['observation_updates'],
                   'qpos_sha256': hashlib.sha256(qpos.detach().cpu().numpy().tobytes()).hexdigest(),
                   'image_sha256': hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest(),
                   'qpos_shape': list(qpos.shape), 'image_shape': list(image.shape)}
            result['forward_events'].append(row)
            save_event('forward_events.jsonl', row)

        # ACTPolicy覆写了__call__，因此在其内部实际DETRVAE模块上记录前向。
        model.model.policy.model.register_forward_pre_hook(on_forward)

        class RecordedPolicy:
            def reset(self):
                result['resets'] += 1
                model.reset()

            def update_obs(self, obs):
                cameras = {}
                arrays = {}
                for name in config['camera_names']:
                    image = np.asarray(obs['vision'][name]['color'])
                    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
                    cameras[name] = {'shape': list(image.shape), 'dtype': str(image.dtype),
                                     'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
                    arrays[name] = image
                result['received_cameras'] = cameras
                result['received_state'] = {key: np.asarray(value).tolist() for key, value in obs['state'].items()}
                result['observation_updates'] += 1
                sequence = result['observation_updates']
                save_event('observation_events.jsonl', {
                    'sequence': sequence, 'cameras': cameras, 'state': result['received_state']})
                if sequence == 1:
                    np.savez_compressed(args.output / 'received_observation.npz', **arrays)
                elif (sequence - 1) % config['chunk_size'] == 0:
                    np.savez_compressed(args.output / f'received_observation_{sequence:04d}.npz', **arrays)
                model.update_obs(obs)

            def get_action(self):
                if result['action_requests'] >= args.max_actions:
                    raise RuntimeError('bounded diagnostic action limit reached')
                t_before = model.model.t
                forwards_before = len(result['forward_events'])
                start = time.perf_counter()
                actions = model.get_action()
                torch.cuda.synchronize()
                assert len(actions) == 1
                result['actions'] = [{key: np.asarray(value).tolist() for key, value in action.items()} for action in actions]
                slot = t_before % config['chunk_size']
                expected = model.model.all_actions[0, slot].detach().cpu().numpy()
                expected = expected * model.model.stats['action_std'] + model.model.stats['action_mean']
                packed = np.concatenate([np.asarray(actions[0][key]) for key in (
                    'left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state')])
                assert np.array_equal(expected, packed), 'returned action does not match cached slot'
                fresh = len(result['forward_events']) - forwards_before
                assert fresh == int(slot == 0), 'unexpected actual model forward count'
                result['action_requests'] += 1
                result['inference_seconds'] = time.perf_counter() - start
                row = {'request_index': result['action_requests'], 'model_t_before': t_before,
                       'cached_slot': slot, 'actual_forwards_this_request': fresh,
                       'observation_sequence': result['observation_updates'],
                       'actions': result['actions'], 'cache_action_max_abs_error': float(np.max(np.abs(expected-packed))),
                       'seconds': result['inference_seconds']}
                save_event('action_events.jsonl', row)
                (args.output / 'prediction.json').write_text(json.dumps(result['actions'], indent=2) + '\n')
                print('ACTION_SERVED', result['action_requests'], 'slot', slot, 'forwards', fresh, flush=True)
                return actions

        async def serve():
            server = PolicyServer(RecordedPolicy(), PolicyServerConfig(host='127.0.0.1', port=0))
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signum, stop.set)
            await server.start()
            (args.output / 'port.json').write_text(json.dumps({'url': server.url, 'pid': os.getpid()}) + '\n')
            print('READY', server.url, flush=True)
            try:
                await stop.wait()
            finally:
                await server.stop()

        asyncio.run(serve())
        result['ok'] = (result['resets'] == 1 and result['observation_updates'] == result['action_requests'] == args.max_actions
                        and [e['request_index'] for e in result['forward_events']] == list(range(1, args.max_actions + 1, config['chunk_size'])))
        result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-actions', type=int, default=1)
    raise SystemExit(main(parser.parse_args()))
