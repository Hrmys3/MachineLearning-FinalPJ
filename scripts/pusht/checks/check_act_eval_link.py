"""固定布局0：真实三路观测 -> ACT WebSocket -> 一次原生EvalEnv动作。"""
import argparse
from copy import deepcopy
import hashlib
import shutil
import time
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
    shutil.copy2(__file__, args.output / "client_source.py")
    layout_path = UPSTREAM / "Assets/Eval_Layout/RoboDojo/arx_x5/0/push_T_0.json"
    protocol = {"task": "push_T", "layout": 0, "physical_gpu": args.device_id,
                "gpu_uuid": args.gpu_uuid, "port": args.port, "actions": 1,
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "upstream_diff": subprocess.check_output(["git", "diff"], cwd=UPSTREAM, text=True),
                "layout_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest()}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
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
    result = {"ok": False, "scope": "one real ACT WebSocket action in EvalEnv, no task success evaluation"}
    try:
        import numpy as np
        import torch
        from cuda.core import Device
        from omegaconf import OmegaConf
        from PIL import Image
        from env.global_configs import ENV_CONFIG_PATH
        from src.eval_client.eval_env import create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.device_id)
        Device(args.device_id).set_current()
        cfgpath = Path(ENV_CONFIG_PATH)
        ev = load_yaml(str(cfgpath / "arx_x5.yml"))
        ev.update(task_name="push_T", num_envs=1, device_id=args.device_id, seed=0,
                  policy_name="ACT", additional_info="act_eval_link_check", eval_batch=False,
                  physx_monitor_enabled=False)
        cfg = OmegaConf.create({
            **{key: load_yaml(str(cfgpath / key / (ev["config"][key] + ".yml")))
               for key in ("sim", "scene", "robot", "camera")},
            "task_env": load_yaml(str(UPSTREAM / "task/RoboDojo/config/push_T.yml")),
            "eval_cfg": ev,
            "deploy_cfg": {"policy_name": "ACT", "host": "127.0.0.1", "port": args.port,
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
        manager = env.robot_manager

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=float)
            if not np.isfinite(value).all():
                raise ValueError("non-finite physical state")
            return value

        def snapshot():
            robots = {}
            for robot in manager.robot_list:
                if robot.type != "target":
                    continue
                robots[robot.arm_name] = {
                    "arm": array(manager.get_joint(robot)[0]).tolist(),
                    "gripper": array(manager.get_end_effector_real_val(robot)[0]).tolist(),
                }
            objects = {}
            for label in ("t", "target_t"):
                pos, ori = env.scene_manager.layout_manager.get_instance_pose(0, label=label)
                objects[label] = np.concatenate([array(pos).reshape(-1), array(ori).reshape(-1)]).tolist()
            return {"robots": robots, "objects": objects}

        def capture(obs, tag):
            cameras = {}
            for name in sorted(expected_cameras):
                image = np.asarray(obs["vision"][name]["color"])
                assert image.shape == (480, 640, 3) and image.dtype == np.uint8
                Image.fromarray(image).save(args.output / f"{tag}_{name}.png")
                cameras[name] = {"shape": list(image.shape), "dtype": str(image.dtype),
                                 "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            return cameras

        result["before"] = snapshot()
        result["sent_cameras"] = capture(observation, "before")
        result["sent_state"] = {key: array(value).tolist() for key, value in observation["state"].items()}
        env.model_client.call(func_name="update_obs", obs=observation)
        start = time.perf_counter()
        actions = env.model_client.call(func_name="get_action")
        result["request_seconds"] = time.perf_counter() - start
        if len(actions) != 1:
            raise ValueError(f"expected one action, got {len(actions)}")
        action = actions[0]
        expected_dims = {"left_arm_joint_state": 6, "left_ee_joint_state": 1,
                         "right_arm_joint_state": 6, "right_ee_joint_state": 1}
        if {key: len(value) for key, value in action.items()} != expected_dims:
            raise AssertionError(f"unexpected action fields: {action}")
        result["action"] = {key: array(value).tolist() for key, value in action.items()}
        (args.output / "prediction-before-execution.json").write_text(json.dumps(result, indent=2) + "\n")

        clipping = {}
        arm_targets = {}
        for i, robot in enumerate(manager.robot_list):
            if robot.type != "target":
                continue
            arm_key = manager.process_name(robot.arm_name)
            ee_key = manager.process_name(robot.gripper_name)
            target = array(action[arm_key])
            current = array(manager.get_joint(robot)[0])
            limits = getattr(manager.robot_key[i].data, "joint_pos_limits", None)
            bounds = None if limits is None else array(limits[0, robot.arm_joint_indices]).tolist()
            arm_targets[arm_key] = {"target": target.tolist(), "delta": (target-current).tolist(),
                                    "max_abs_delta_rad": float(np.max(np.abs(target-current))),
                                    "joint_limits": bounds}
            raw = float(action[ee_key][0])
            clipped = float(np.clip(raw, 0, 1))
            lo, hi = [float(v) for v in robot.gripper_scale]
            primary = lo + clipped * (hi-lo) if robot.gripper_move["sign"] == 1 else lo + (1-clipped)*(hi-lo)
            clipping[ee_key] = {"raw": raw, "clipped_0_1": clipped,
                                "was_clipped": raw != clipped, "expected_primary_m": primary}
        result["arm_targets"] = arm_targets
        result["gripper_mapping"] = clipping
        print("MODEL_ACTION_BEFORE_EXECUTION", json.dumps(result["action"]), flush=True)

        def serializable(value):
            if isinstance(value, dict):
                return {key: serializable(v) for key, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(v) for v in value]
            if hasattr(value, "tolist"):
                return value.tolist()
            return value

        original_control = env.process_control_info
        def recorded_control(control_info, env_idx):
            sequence = original_control(control_info, env_idx)
            result["native_control_targets"] = serializable(deepcopy(control_info))
            result["native_sequence_length"] = len(sequence)
            result["native_first_control"] = serializable(sequence[0])
            result["native_last_control"] = serializable(sequence[-1])
            return sequence
        env.process_control_info = recorded_control
        physics_steps = [0]
        original_step = env.sim_step
        def counted_step(*a, **kw):
            physics_steps[0] += 1
            return original_step(*a, **kw)
        env.sim_step = counted_step
        try:
            env.take_action(action)
        finally:
            env.sim_step = original_step
            env.process_control_info = original_control
        result["physical_steps"] = physics_steps[0]
        result["after"] = snapshot()
        after_obs = env.get_obs()
        result["after_cameras"] = capture(after_obs, "after")
        for key, mapping in clipping.items():
            commanded = float(result["native_last_control"][key]["position"][0])
            mapping["actual_native_primary_target_m"] = commanded
            assert abs(commanded-mapping["expected_primary_m"]) < 1e-7
        changes = {}
        for name, before_robot in result["before"]["robots"].items():
            delta = array(result["after"]["robots"][name]["arm"])-array(before_robot["arm"])
            changes[name] = {"delta_rad": delta.tolist(), "max_abs_delta_rad": float(np.max(np.abs(delta)))}
        result["measured_arm_changes"] = changes
        result["t_translation_m"] = float(np.linalg.norm(
            array(result["after"]["objects"]["t"][:3])-array(result["before"]["objects"]["t"][:3])))
        result["take_action_count"] = env.take_action_cnt[0]
        result["collect_interval"] = float(env.obs_manager.collect_interval)
        result["dt"] = float(env.dt)
        result["ok"] = (physics_steps[0] == 10 and result["native_sequence_length"] == 10
                         and env.take_action_cnt[0] == 1 and any(v["max_abs_delta_rad"] > 1e-7 for v in changes.values()))
        if not result["ok"]:
            raise AssertionError("single-action execution checks failed")
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
