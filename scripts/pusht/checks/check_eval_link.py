"""固定布局0：真实 EvalEnv 观测 -> 本机假策略 -> 一次原生关节动作。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party/pusht/RoboDojo"
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "XPolicyLab"), str(ROOT / "src")]


def check_gpu(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("unset CUDA_VISIBLE_DEVICES: physical CUDA/Vulkan indices must match")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    (args.output / "gpu-before.txt").write_text(rows)
    selected = next((row.split(",") for row in rows.splitlines()
                     if int(row.split(",")[0]) == args.device_id), None)
    if selected is None or selected[1].strip() != args.gpu_uuid or int(selected[2]) >= 10 or int(selected[3]) != 0:
        raise ValueError("selected GPU UUID mismatch or card not idle")


def main(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    check_gpu(args)
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), "127.0.0.1", "localhost"]))
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.chdir(UPSTREAM)
    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    launcher = AppLauncher({"headless": True, "enable_cameras": True,
                            "device": f"cuda:{args.device_id}", "multi_gpu": False,
                            "kit_args": zero_delay_kit_args()})
    app = launcher.app
    env = None
    result = {"ok": False, "scope": "one real EvalEnv action using CPU dummy policy; no ACT model"}
    try:
        import numpy as np
        import torch
        from cuda.core import Device
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH
        from src.eval_client.eval_env import create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        torch.cuda.set_device(args.device_id)
        Device(args.device_id).set_current()
        cfgpath = Path(ENV_CONFIG_PATH)
        ev = load_yaml(str(cfgpath / "arx_x5.yml"))
        ev.update(task_name="push_T", num_envs=1, device_id=args.device_id, seed=0,
                  policy_name="dummy", additional_info="eval_link_check", eval_batch=False,
                  physx_monitor_enabled=False)
        cfg = OmegaConf.create({
            **{key: load_yaml(str(cfgpath / key / (ev["config"][key] + ".yml")))
               for key in ("sim", "scene", "robot", "camera")},
            "task_env": load_yaml(str(UPSTREAM / "task/RoboDojo/config/push_T.yml")),
            "eval_cfg": ev,
            "deploy_cfg": {"policy_name": "dummy", "host": "127.0.0.1", "port": args.port,
                           "policy_server_url": f"ws://127.0.0.1:{args.port}"},
        })
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [0]
        cfg, _ = process_config(process_randomization(cfg), "push_T")
        cfg.camera.default_frequency = ev["observation"]["collect_freq"]
        (args.output / "effective-config.yaml").write_text(OmegaConf.to_yaml(cfg))
        env = create_eval_env(cfg, app)
        env.reset([0])
        observation = env.get_obs()
        expected_cameras = {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        if set(observation["vision"]) != expected_cameras:
            raise AssertionError(f"unexpected cameras: {sorted(observation['vision'])}")
        for name in expected_cameras:
            image = np.asarray(observation["vision"][name]["color"])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise AssertionError(f"invalid {name} image: {image.shape}, {image.dtype}")
        env.model_client.call(func_name="update_obs", obs=observation)
        actions = env.model_client.call(func_name="get_action")
        if len(actions) != 1:
            raise ValueError(f"expected one action, got {len(actions)}")
        action = actions[0]
        expected_dims = {"left_arm_joint_state": 6, "left_ee_joint_state": 1,
                         "right_arm_joint_state": 6, "right_ee_joint_state": 1}
        if {key: len(value) for key, value in action.items()} != expected_dims:
            raise AssertionError(f"unexpected action fields: {action}")
        before = np.asarray(observation["state"]["left_arm_joint_state"], dtype=float)
        env.take_action(action)
        after_obs = env.get_obs()
        after = np.asarray(after_obs["state"]["left_arm_joint_state"], dtype=float)
        delta = float(after[0] - before[0])
        result.update({"ok": bool(np.isfinite(delta) and abs(delta - .01) < .003),
                       "camera_keys": sorted(observation["vision"]),
                       "state_keys": sorted(observation["state"]),
                       "action_dims": {key: len(value) for key, value in action.items()},
                       "left_joint0_before": float(before[0]),
                       "left_joint0_after": float(after[0]),
                       "left_joint0_delta": delta,
                       "take_action_count": env.take_action_cnt[0],
                       "collect_interval": env.obs_manager.collect_interval})
        if not result["ok"]:
            raise AssertionError(f"joint response mismatch: {delta}")
    except BaseException as exc:
        result["error"] = repr(exc)
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.model_client.close()
                env.close()
            except BaseException as exc:
                result["close_error"] = repr(exc)
                result["ok"] = False
                traceback.print_exc()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        app.close()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.device_id < 0 or not 1 <= args.port <= 65535:
        parser.error("invalid GPU index or port")
    raise SystemExit(main(args))
