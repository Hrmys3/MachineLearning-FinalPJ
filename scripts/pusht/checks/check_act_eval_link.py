"""固定布局0：真实三路观测 -> ACT WebSocket -> 有界原生EvalEnv短回合。"""
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
    if not 1 <= args.steps <= 51:
        raise ValueError("steps must be between 1 and 51")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    check_gpu(args)
    shutil.copy2(__file__, args.output / "client_source.py")
    layout_path = UPSTREAM / "Assets/Eval_Layout/RoboDojo/arx_x5/0/push_T_0.json"
    protocol = {"task": "push_T", "layout": 0, "physical_gpu": args.device_id,
                "gpu_uuid": args.gpu_uuid, "port": args.port, "actions": args.steps,
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
    result = {"ok": False, "scope": "bounded real ACT WebSocket rollout; not task success evaluation",
              "planned_actions": args.steps, "steps": []}
    video = None
    video_log = None
    video_frames = 0
    run_started = time.perf_counter()
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
        # 原生run_eval在reset后先注册奖励条件；直接驱动take_action也须保留这一步。
        env.run_reward()
        result["reward_conditions"] = {
            "checks": len(env.reward_manager.check_list[0]),
            "queries": len(env.reward_manager.query_list[0]),
        }
        assert result["reward_conditions"]["checks"] > 0
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

        def capture(obs, tag, save_images=True):
            cameras = {}
            for name in sorted(expected_cameras):
                image = np.asarray(obs["vision"][name]["color"])
                assert image.shape == (480, 640, 3) and image.dtype == np.uint8
                if save_images:
                    Image.fromarray(image).save(args.output / f"{tag}_{name}.png")
                cameras[name] = {"shape": list(image.shape), "dtype": str(image.dtype),
                                 "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            return cameras

        video_log = (args.output / "ffmpeg.log").open("w")
        video = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-video_size", "640x480", "-framerate", "25",
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", str(args.output / "head.mp4")
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=video_log)
        video.stdin.write(np.asarray(observation["vision"]["cam_head"]["color"]).tobytes())
        video_frames = 1
        total_steps = 0
        result["initial"] = snapshot()
        result["termination_reason"] = "action_budget"
        for step_index in range(1, args.steps + 1):
            if env.is_episode_end():
                result["termination_reason"] = "native_episode_end"
                break
            row = {"action_index": step_index, "completed": False}
            result["steps"].append(row)
            row["before"] = snapshot()
            row["sent_cameras"] = capture(observation, "before" if args.steps == 1 else f"step_{step_index:04d}_before", save_images=step_index in {1, 49, 50, 51})
            row["sent_state"] = {key: array(value).tolist() for key, value in observation["state"].items()}
            env.model_client.call(func_name="update_obs", obs=observation)
            start = time.perf_counter()
            actions = env.model_client.call(func_name="get_action")
            row["request_seconds"] = time.perf_counter() - start
            if len(actions) != 1:
                raise ValueError(f"expected one action, got {len(actions)}")
            action = actions[0]
            expected_dims = {"left_arm_joint_state": 6, "left_ee_joint_state": 1,
                             "right_arm_joint_state": 6, "right_ee_joint_state": 1}
            if {key: len(value) for key, value in action.items()} != expected_dims:
                raise AssertionError(f"unexpected action fields: {action}")
            row["action"] = {key: array(value).tolist() for key, value in action.items()}
            (args.output / f"prediction_{step_index:04d}_before_execution.json").write_text(json.dumps(row, indent=2) + "\n")

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
            row["arm_targets"] = arm_targets
            row["gripper_mapping"] = clipping
            with (args.output / "step_events.jsonl").open("a") as file:
                file.write(json.dumps({"phase": "before_execution", **row}) + "\n")
            print("MODEL_ACTION_BEFORE_EXECUTION", step_index, json.dumps(row["action"]), flush=True)

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
                row["native_control_targets"] = serializable(deepcopy(control_info))
                row["native_sequence_length"] = len(sequence)
                row["native_first_control"] = serializable(sequence[0])
                row["native_last_control"] = serializable(sequence[-1])
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
            row["physical_steps"] = physics_steps[0]
            row["after"] = snapshot()
            after_obs = env.get_obs()
            row["after_cameras"] = capture(after_obs, "after" if args.steps == 1 else f"step_{step_index:04d}_after", save_images=step_index in {1, 49, 50, 51})
            for key, mapping in clipping.items():
                commanded = float(row["native_last_control"][key]["position"][0])
                mapping["actual_native_primary_target_m"] = commanded
                assert abs(commanded-mapping["expected_primary_m"]) < 1e-7
            changes = {}
            for name, before_robot in row["before"]["robots"].items():
                delta = array(row["after"]["robots"][name]["arm"])-array(before_robot["arm"])
                changes[name] = {"delta_rad": delta.tolist(), "max_abs_delta_rad": float(np.max(np.abs(delta)))}
            row["measured_arm_changes"] = changes
            row["t_translation_m"] = float(np.linalg.norm(
                array(row["after"]["objects"]["t"][:3])-array(row["before"]["objects"]["t"][:3])))
            row["take_action_count"] = env.take_action_cnt[0]
            row["collect_interval"] = float(env.obs_manager.collect_interval)
            row["dt"] = float(env.dt)
            row["ok"] = (physics_steps[0] == 10 and row["native_sequence_length"] == 10
                             and env.take_action_cnt[0] == step_index)
            if not row["ok"]:
                raise AssertionError("per-action execution checks failed")
            row["completed"] = True
            total_steps += row["physical_steps"]
            with (args.output / "step_events.jsonl").open("a") as file:
                file.write(json.dumps({"phase": "after_execution", **row}) + "\n")
            video.stdin.write(np.asarray(after_obs["vision"]["cam_head"]["color"]).tobytes())
            video_frames += 1
            observation = after_obs
            if env.is_episode_end():
                result["termination_reason"] = "native_episode_end"
                break
        result["completed_actions"] = sum(row["completed"] for row in result["steps"])
        result["total_physical_steps"] = total_steps
        result["simulated_seconds"] = total_steps * float(env.dt)
        result["final"] = snapshot()
        result["final_cameras"] = capture(observation, "final")
        result["take_action_count"] = env.take_action_cnt[0]
        result["dt"] = float(env.dt)
        result["collect_interval"] = float(env.obs_manager.collect_interval)
        result["native_end_flag"] = bool(env.end_flag[0])
        result["native_success_flag"] = bool(env.success[0])
        # success在运行中初始为True，必须结合end_flag才能解释为任务成功。
        result["native_task_success"] = bool(env.end_flag[0] and env.success[0])
        result["t_translation_m"] = float(np.linalg.norm(
            array(result["final"]["objects"]["t"][:3])-array(result["initial"]["objects"]["t"][:3])))
        result["ok"] = (result["completed_actions"] == args.steps
                         and total_steps == args.steps * 10 and env.take_action_cnt[0] == args.steps)
        if args.steps == 1 and result["steps"]:
            result.update(result["steps"][0])
        if not result["ok"]:
            raise AssertionError("rollout ended before the planned action budget")
    except BaseException as exc:
        result["error"] = repr(exc)
        traceback.print_exc()
    finally:
        result["completed_actions"] = sum(row.get("completed", False) for row in result["steps"])
        result["total_physical_steps"] = sum(row.get("physical_steps", 0) for row in result["steps"])
        result["video_frames"] = video_frames
        result["elapsed_seconds"] = time.perf_counter() - run_started
        if video is not None:
            try:
                video.stdin.close()
                rc = video.wait(timeout=20)
                result["video_returncode"] = rc
                if rc != 0:
                    raise RuntimeError(f"ffmpeg exited {rc}")
            except BaseException as exc:
                if video.poll() is None:
                    video.kill()
                    video.wait()
                result["video_error"] = repr(exc)
                result["ok"] = False
        if video_log is not None:
            video_log.close()
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
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.device_id < 0 or not 1 <= args.port <= 65535:
        parser.error("invalid GPU index or port")
    raise SystemExit(main(args))
