"""固定RoboDojo布局的重置、稳定性与原生关节伺服检查；不连接策略服务。"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party/pusht/RoboDojo"


def json_value(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot record {type(value).__name__}")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_value) + "\n")


def check_idle_gpu(args, filename):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("unset CUDA_VISIBLE_DEVICES: matching CUDA/Vulkan physical indices required")
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    (args.output / filename).write_text(rows)
    selected = next((r.split(",") for r in rows.splitlines()
                     if int(r.split(",")[0]) == args.device_id), None)
    if selected is None or selected[1].strip() != args.gpu_uuid or int(selected[2]) >= 10 or int(selected[3]) != 0:
        raise ValueError("selected GPU UUID does not match or GPU is not idle")


def worker(args):
    # 仿真相关导入仅在已通过GPU检查的独立worker中执行。
    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    launcher = AppLauncher({"headless": True, "enable_cameras": True,
                            "device": f"cuda:{args.device_id}", "multi_gpu": False,
                            "kit_args": zero_delay_kit_args()})
    app = launcher.app
    env = None
    out = args.output
    result = {"ok": False, "planned_resets": args.repeats, "resets": [], "actions": []}
    started = time.perf_counter()
    try:
        from copy import deepcopy
        import numpy as np
        import torch
        from cuda.core import Device
        from omegaconf import OmegaConf
        from PIL import Image
        from env.global_configs import ENV_CONFIG_PATH
        from env.robot_manager.control_manager import MetaControl
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras
        from task.RoboDojo.task_registry import load_task_class
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        torch.cuda.set_device(args.device_id)
        Device(args.device_id).set_current()
        cfgpath = Path(ENV_CONFIG_PATH)
        ev = load_yaml(str(cfgpath / "arx_x5.yml"))
        ev.update(task_name="push_T", num_envs=1, device_id=args.device_id, seed=0)
        cfg = OmegaConf.create({
            **{key: load_yaml(str(cfgpath / key / (ev["config"][key] + ".yml")))
               for key in ["sim", "scene", "robot", "camera"]},
            "task_env": load_yaml(str(UPSTREAM / "task/RoboDojo/config/push_T.yml")),
            "eval_cfg": ev,
        })
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [0]
        cfg, _ = process_config(process_randomization(cfg), "push_T")
        cfg.camera.default_frequency = ev["observation"]["collect_freq"]
        (out / "effective-config.yaml").write_text(OmegaConf.to_yaml(cfg))
        _, cls = load_task_class("push_T")
        env = cls(cfg, app)
        manager = env.robot_manager
        layout_manager = env.scene_manager.layout_manager
        layout_path = UPSTREAM / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/push_T_{args.layout_id}.json"
        layout = json.loads(layout_path.read_text())
        result["layout_sha256"] = hashlib.sha256(layout_path.read_bytes()).hexdigest()
        result["saved_layout"] = layout
        result["robot_order"] = [
            {"arm": r.arm_name, "gripper": r.gripper_name,
             "arm_joints": r.arm_joints_name, "gripper_joints": r.gripper_joints_name,
             "root_pose": r.entity_origin_pose, "gripper_scale": r.gripper_scale}
            for r in manager.robot_list]

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            a = np.asarray(value, dtype=float).reshape(-1)
            if not np.isfinite(a).all():
                raise ValueError("Non-finite state")
            return a

        def state():
            objects = {}
            for label in ["t", "target_t", "camera_stand"]:
                pos, ori = layout_manager.get_instance_pose(0, label=label)
                objects[label] = np.concatenate([array(pos), array(ori)]).tolist()
            robots = {}
            for r in manager.robot_list:
                robots[r.arm_name] = {
                    "arm": array(manager.get_joint(r)[0]).tolist(),
                    "gripper": array(manager.get_end_effector_real_val(r)[0]).tolist(),
                    "end_pose": array(manager.get_real_endpose(r)[0]).tolist(),
                }
            return {"objects": objects, "robots": robots}

        def pose_error(a, b):
            a, b = array(a), array(b)
            qa, qb = a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:])
            return {"position_m": float(np.linalg.norm(a[:3] - b[:3])),
                    "angle_deg": float(np.degrees(2 * np.arccos(np.clip(abs(qa @ qb), 0, 1))))}

        def capture(name):
            wait_for_latest_cameras(env)
            frames = env.capture_manager.step()
            paths = []
            for i, cam in enumerate(frames):
                rgba = np.asarray(cam["rgb"][0]["data"])
                assert rgba.shape == (480, 640, 4) and np.isfinite(rgba).all()
                assert rgba[..., :3].std() > 1
                assert str(env.capture_manager._output_buffers[i]["rgb"].device) == f"cuda:{args.device_id}"
                path = f"{name}_{env.capture_manager.camera_names[0][i]}.png"
                Image.fromarray(rgba[..., :3].astype(np.uint8)).save(out / path)
                paths.append(path)
            return paths

        def persist():
            result["wall_seconds"] = time.perf_counter() - started
            save_json(out / "result.json", result)

        for repeat in range(args.repeats):
            record = {"index": repeat, "native_stable": False}
            result["resets"].append(record)
            env.success, env.end_flag = [True], [False]
            layout_manager.set_saved_layout(0, deepcopy(layout))
            env.reset(seed=[0])
            record["after_reset"] = state()
            env.scene_manager.apply_saved_poses([0])
            record["after_apply"] = state()
            record["native_stability_before"] = state()
            native_stable, unstable = layout_manager.check_layout_stability(env, render=False)
            record.update(native_stable=bool(native_stable), unstable_envs=list(unstable),
                          native_success=list(env.success), native_end_flag=list(env.end_flag))
            record["after_native_stability"] = state()
            manager.set_origin_endpose()
            manager.set_robot_init_state()
            env.reward_manager.init_state()
            for _ in range(200):
                env.sim_step(render=False)
            record["before_hold"] = state()
            samples = []
            for step in range(args.hold_steps):
                env.sim_step(render=False)
                if (step + 1) % 10 == 0 or step == args.hold_steps - 1:
                    samples.append(state())
            record["hold_samples"] = samples
            record["after_hold"] = state()
            errors = [pose_error(record["before_hold"]["objects"]["t"], s["objects"]["t"])
                      for s in samples]
            record["hold_max_drift"] = {key: max(e[key] for e in errors) for key in errors[0]}
            record["strict_hold_passed"] = (record["hold_max_drift"]["position_m"] <= .001
                                             and record["hold_max_drift"]["angle_deg"] <= 1.)
            record["frames"] = capture(f"reset_{repeat:02d}")
            persist()
            print("RESET_CHECK", repeat, native_stable, record["hold_max_drift"], flush=True)

        baseline = result["resets"][0]["after_hold"]
        comparisons = []
        for record in result["resets"][1:]:
            current = record["after_hold"]
            errors = {label: pose_error(baseline["objects"][label], current["objects"][label])
                      for label in baseline["objects"]}
            joint_max = max(float(np.max(np.abs(array(baseline["robots"][name][key]) -
                                                array(current["robots"][name][key]))))
                            for name in baseline["robots"] for key in ["arm", "gripper"])
            comparisons.append({"index": record["index"], "object_errors": errors,
                                "robot_max_joint_difference": joint_max,
                                "passed": all(v["position_m"] <= .001 and v["angle_deg"] <= 1.
                                              for v in errors.values()) and joint_max <= .001})
        result["reset_comparisons"] = comparisons
        stable = all(r["native_stable"] and r["strict_hold_passed"] for r in result["resets"])

        # 与TaskEnv相同的低层MetaControl伺服路径。这里的夹爪值是米，
        # EvalEnv部署接口的0..1归一化值必须另经原生映射，不能混用。
        def command_for(snapshot):
            command = {}
            for r in manager.robot_list:
                rstate = snapshot["robots"][r.arm_name]
                command[manager.process_name(r.arm_name)] = {"position": rstate["arm"], "velocity": [0.] * 6}
                command[manager.process_name(r.gripper_name)] = {"position": rstate["gripper"], "velocity": [0.] * 2}
            return command

        def send(command, steps=100):
            for i in range(steps):
                env.step([MetaControl(command)])
                env.sim_step(render=(i % 10 == 0))

        if stable:
            initial = state()
            for robot in manager.robot_list:
                for kind, delta in [("arm", .02), ("gripper", -.002)]:
                    before = state()
                    command = command_for(initial)
                    key = manager.process_name(robot.arm_name if kind == "arm" else robot.gripper_name)
                    target = array(command[key]["position"])
                    target[0] += delta
                    if kind == "gripper":
                        target[1] = target[0] * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]
                    command[key]["position"] = target.tolist()
                    send(command)
                    moved = state()
                    motion = array(moved["robots"][robot.arm_name][kind]) - array(before["robots"][robot.arm_name][kind])
                    other_motion = {r.arm_name: (array(moved["robots"][r.arm_name]["arm"]) -
                                               array(before["robots"][r.arm_name]["arm"])).tolist()
                                    for r in manager.robot_list if r is not robot}
                    error = float(np.max(np.abs(array(moved["robots"][robot.arm_name][kind]) - target)))
                    action_record = {"arm": robot.arm_name, "kind": kind, "requested_delta": delta,
                                     "units": "rad" if kind == "arm" else "m", "command": command,
                                     "before": before, "after": moved, "actual_delta": motion.tolist(),
                                     "max_target_error": error, "other_arm_motion": other_motion,
                                     "frames": capture(f"action_{robot.arm_name}_{kind}")}
                    threshold = .003 if kind == "arm" else .0005
                    action_record["response_passed"] = (float(motion[0]) * delta > 0
                                                         and abs(float(motion[0])) >= abs(delta) * .5
                                                         and error <= threshold
                                                         and all(max(abs(v) for v in values) <= .001
                                                                 for values in other_motion.values()))
                    send(command_for(initial))
                    returned = state()
                    action_record["returned"] = returned
                    action_record["return_error"] = float(np.max(np.abs(array(returned["robots"][robot.arm_name][kind]) -
                                                                                   array(initial["robots"][robot.arm_name][kind]))))
                    action_record["return_passed"] = action_record["return_error"] <= threshold
                    result["actions"].append(action_record)
                    persist()
                    print("ACTION_CHECK", robot.arm_name, kind, action_record["response_passed"],
                          action_record["return_passed"], flush=True)
        else:
            result["actions_skipped_reason"] = "Initial layout failed native stability or strict static hold"
        result["ok"] = (stable and all(c["passed"] for c in comparisons)
                        and len(result["actions"]) == 4
                        and all(a["response_passed"] and a["return_passed"] for a in result["actions"]))
        result["scope"] = "layout 0 diagnostic; native and strict stability, repeated reset, low-level servo; no policy or task success"
        persist()
    except BaseException as exc:
        result["error"] = repr(exc)
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        save_json(out / "result.json", result)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["close_error"] = repr(exc)
                result["ok"] = False
                save_json(out / "result.json", result)
                traceback.print_exc()
        app.close()
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hold-steps", type=int, default=250)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.device_id, args.layout_id) < 0 or args.repeats < 2 or args.hold_steps < 10 or args.timeout <= 0:
        parser.error("invalid device/layout/repeat/hold/timeout")
    args.output = args.output.resolve()
    if args.worker:
        if os.environ.get("MLPJ_CHECK_PARENT_PID") != str(os.getppid()):
            parser.error("internal worker requires its launcher parent")
        if not (args.output / "protocol.json").is_file() or (args.output / "result.json").exists():
            parser.error("worker output is unprepared or already contains results")
        with (args.output / ".worker-started").open("x") as claim:
            claim.write(str(os.getpid()))
        check_idle_gpu(args, "gpu-worker-before.txt")
        raise SystemExit(worker(args))
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("unset CUDA_VISIBLE_DEVICES: this check uses matching CUDA/Vulkan physical indices")
    args.output.mkdir(parents=True, exist_ok=False)
    check_idle_gpu(args, "gpu-before.txt")
    save_json(args.output / "protocol.json", {
        "argv": sys.argv, "layout_id": args.layout_id, "repeats": args.repeats, "hold_steps": args.hold_steps,
        "device_id": args.device_id, "gpu_uuid": args.gpu_uuid, "dt": .004,
        "strict_hold_position_m": .001, "strict_hold_angle_deg": 1.,
        "repeat_position_m": .001, "repeat_angle_deg": 1., "repeat_joint_tolerance": .001,
        "servo_arm_delta_rad": .02, "servo_gripper_delta_m": -.002,
        "arm_error_rad": .003, "gripper_error_m": .0005,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip(),
        "upstream_diff": subprocess.check_output(["git", "diff"], cwd=UPSTREAM, text=True),
    })
    (args.output / "task_checks_source.py").write_bytes(Path(__file__).read_bytes())
    envvars = dict(os.environ)
    envvars["OMNI_KIT_ACCEPT_EULA"] = "YES"
    envvars["MLPJ_CHECK_PARENT_PID"] = str(os.getpid())
    envvars["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    envvars["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(UPSTREAM), str(UPSTREAM / "XPolicyLab"),
                                           envvars.get("PYTHONPATH", "")])
    cmd = [sys.executable, "-u", "-m", "mlpj.pusht.task_checks", "--worker", "--output", str(args.output),
           "--device-id", str(args.device_id), "--gpu-uuid", args.gpu_uuid, "--layout-id", str(args.layout_id),
           "--repeats", str(args.repeats), "--hold-steps", str(args.hold_steps)]
    timed_out = False
    with (args.output / "runtime.log").open("w") as log:
        child = subprocess.Popen(cmd, cwd=UPSTREAM, env=envvars, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            rc = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            rc = child.wait()
            timed_out = True
    save_json(args.output / "process-status.json", {"returncode": rc, "timed_out": timed_out})
    after = subprocess.check_output(["nvidia-smi", "-i", str(args.device_id),
                                    "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv"], text=True)
    (args.output / "gpu-after.txt").write_text(after)
    print(f"Check exited {rc}; evidence: {args.output}")
    raise SystemExit(1 if timed_out else rc)


if __name__ == "__main__":
    main()
