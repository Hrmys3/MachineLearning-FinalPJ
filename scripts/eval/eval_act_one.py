
# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)
"""单个验证初态的 ACT / ACT＋规则保持闭环评测。"""

import json
import csv
import sys
from argparse import Namespace
from lift_protocol import FirstLiftMetric, HoldController, parse_args, experiment_config
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import robosuite as suite

root = Path("third_party/act_lift").resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "detr"))
from detr.models import build_ACT_model

BASE = Path("data/lift_act")
SOURCE = Path("data/robomimic_source/v1.5/lift/ph/demo_v15.hdf5")
CAMERAS = ["agentview", "robot0_eye_in_hand"]


def decode(value):
    return value.decode() if isinstance(value, bytes) else value


def main(args=None):
    args = args or parse_args()
    splits = json.loads((BASE / "splits.json").read_text())
    episode_id = splits["valid"][args.index]
    output = args.output / f"episode_{episode_id:04d}"
    # 在模型加载和仿真之前拒绝复用目录，也保留失败运行证据。
    output.mkdir(parents=True, exist_ok=False)
    config = experiment_config(args)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # checkpoint 是我们自己生成的可信本地文件。
    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = build_ACT_model(Namespace(**ckpt["model_args"])).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = ckpt["stats"]

    mapping = [
        json.loads(line)
        for line in Path(
            "data/robomimic_mt4_ph/meta/robomimic_source_episodes.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    entry = next(r for r in mapping if r["episode_index"] == episode_id)

    env = None
    try:
        with h5py.File(SOURCE, "r") as f:
            metadata = json.loads(decode(f["data"].attrs["env_args"]))
            demo = f["data"][entry["source_demo_key"]]
            initial_state = demo["states"][0]
            xml = decode(demo.attrs["model_file"])

        kwargs = metadata["env_kwargs"].copy()
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=CAMERAS,
            camera_heights=256,
            camera_widths=256,
            lite_physics=False,
            horizon=max(250, args.steps + 1),
            control_freq=20,
        )
        env = suite.make(metadata["env_name"], **kwargs)
        env.reset()
        env.reset_from_xml_string(env.edit_model_xml(xml))
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env.robots[0].reset(deterministic=True)
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        env._update_observables(force=True)
        obs = env._get_observations()

        # 与同一示范的缓存首帧比较，确认每路相机的翻转约定。
        flips = {}
        with h5py.File(
            BASE / "episodes" / f"episode_{episode_id:04d}.hdf5", "r"
        ) as f:
            for camera in CAMERAS:
                recorded = f[f"images/{camera}"][0].astype(np.float32)
                raw = obs[f"{camera}_image"]
                errors = [
                    np.mean((raw.astype(np.float32) - recorded) ** 2),
                    np.mean(
                        (np.flipud(raw).astype(np.float32) - recorded) ** 2
                    ),
                ]
                flips[camera] = bool(errors[1] < errors[0])
                print(camera, "raw/flipped MSE:", errors,
                      "flip:", flips[camera], flush=True)

            expected_state = f["state"][0]

        def state_vector(observation):
            return np.concatenate([
                observation["robot0_eef_pos"],
                observation["robot0_eef_quat"],
                observation["robot0_gripper_qpos"],
            ])

        print("Initial state max difference:",
              np.max(np.abs(state_vector(obs) - expected_state)))

        def image_array(observation, camera):
            image = observation[f"{camera}_image"]
            return np.flipud(image).copy() if flips[camera] else image.copy()

        def predict(observation):
            state = (
                state_vector(observation) - np.asarray(stats["state_mean"])
            ) / np.asarray(stats["state_std"])
            images = np.stack([
                image_array(observation, c) for c in CAMERAS
            ])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
            images = F.interpolate(
                images, (128, 128), mode="bilinear", align_corners=False
            )
            mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
            images = ((images - mean) / std).unsqueeze(0).cuda()
            state = torch.tensor(
                state, dtype=torch.float32, device="cuda"
            ).unsqueeze(0)

            with torch.no_grad():
                prediction, _, _ = model(state, images, None)
            actions = prediction[0].cpu().numpy()
            actions = (
                actions * np.asarray(stats["action_std"])
                + np.asarray(stats["action_mean"])
            )
            assert np.isfinite(actions).all()
            return np.clip(actions, -1, 1)

        metric = FirstLiftMetric()
        hold = HoldController() if args.policy == "act_hold" else None
        hold_start_step = None
        frames = [Image.fromarray(image_array(obs, "agentview"))]
        trace = []
        success_streak = 0
        max_success_streak = 0
        first_stable_success = None
        first_loss_after_success = None
        chunk_id = 0

        success_any = False
        first_success = None
        step = 0

        while step < args.steps:
            chunk = predict(obs) if hold_start_step is None else np.zeros((5, 7))
            chunk_id += 1
            for chunk_offset, action in enumerate(chunk[:min(5, args.steps - step)]):
                action_source = "act"
                if hold_start_step is not None:
                    action = hold.action(obs)
                    action_source = "hold_rule"

                before_eef = obs["robot0_eef_pos"].copy()
                before_cube = obs["cube_pos"].copy()
                before_gripper = obs["robot0_gripper_qpos"].copy()
                obs, reward, done, info = env.step(action)
                step += 1
                success = bool(env._check_success())
                metric.update(success)
                if hold is not None and hold_start_step is None and metric.start is not None:
                    hold.activate(obs, env.robots[0].part_controllers["right"])
                    hold_start_step = step + 1  # 第一条规则动作在下一步执行

                success_streak = success_streak + 1 if success else 0
                max_success_streak = max(
                    max_success_streak, success_streak
                )

                # 原生成功连续 10 步，即在 20 Hz 下保持约 0.5 秒。
                if success_streak >= 10 and first_stable_success is None:
                    first_stable_success = step

                if (
                    success_any and not success
                    and first_loss_after_success is None
                ):
                    first_loss_after_success = step

                gripper = obs["robot0_gripper_qpos"].copy()
                eef = obs["robot0_eef_pos"]
                cube = obs["cube_pos"]

                row = {
                    "step": step,
                    "action_source": action_source,
                    "time_s": step / 20.0,
                    "chunk_id": chunk_id,
                    "chunk_offset": chunk_offset,
                    "reward": float(reward),
                    "native_success": int(success),
                    "success_streak_steps": success_streak,
                    # action 是从前一时刻观测预测、在本步执行的动作。
                    "gripper_command": float(action[-1]),
                    "gripper_q0_before": float(before_gripper[0]),
                    "gripper_q1_before": float(before_gripper[1]),
                    "gripper_q0_after": float(gripper[0]),
                    "gripper_q1_after": float(gripper[1]),
                    # 两个滑动关节坐标之差，仅作为开度代理量。
                    "gripper_aperture_proxy": float(
                        abs(gripper[0] - gripper[1])
                    ),
                    "cube_z_before": float(before_cube[2]),
                    "cube_z_after": float(cube[2]),
                    "cube_dz": float(cube[2] - before_cube[2]),
                    "eef_z_before": float(before_eef[2]),
                    "eef_z_after": float(eef[2]),
                    "eef_cube_distance": float(np.linalg.norm(eef - cube)),
                }
                for i, value in enumerate(action):
                    row[f"action_{i}"] = float(value)
                for label, position in (
                    ("eef", eef), ("cube", cube)
                ):
                    for axis, value in zip("xyz", position):
                        row[f"{label}_{axis}"] = float(value)
                trace.append(row)

                if step % 5 == 0:
                    print(
                        f"step={step:03d} "
                        f"grip_cmd={action[-1]:+.3f} "
                        f"aperture={row['gripper_aperture_proxy']:.4f} "
                        f"cube_z={cube[2]:.4f} "
                        f"success={success} streak={success_streak}",
                        flush=True,
                    )

                success_any |= success
                if success and first_success is None:
                    first_success = step
                frames.append(Image.fromarray(
                    image_array(obs, "agentview")
                ))

            if step % 20 == 0:
                print("step:", step, "success:", success,
                      "cube_z:", float(obs["cube_pos"][2]), flush=True)

        with (output / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)

        peak_index = max(
            range(len(trace)), key=lambda i: trace[i]["cube_z_after"]
        )
        peak_row = trace[peak_index]

        # 仅为日志诊断阈值，不改变策略或任务成功标准。
        negative_after_peak = next(
            (
                r["step"] for r in trace[peak_index + 1:]
                if r["gripper_command"] < -0.2
            ),
            None,
        )

        frames[0].save(
            output / "rollout.gif",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        result = {
            **metric.result(),
            "policy": args.policy,
            "hold_rule_start_step": hold_start_step if hold_start_step is not None and hold_start_step <= step else None,
            "hold_target_position": hold.position.tolist() if hold is not None and hold.position is not None else None,
            "hold_target_quaternion_xyzw": hold.quaternion.tolist() if hold is not None and hold.position is not None else None,
            "validation_episode": episode_id,
            "steps": step,
            "success_any": bool(success_any),
            "success_final": bool(env._check_success()),
            "first_success_step": first_success,
            "image_flips": flips,
            "execution_horizon": 5,
            "max_native_success_streak_steps": max_success_streak,
            "max_native_success_streak_seconds": max_success_streak / 20,
            "stable_native_success_10_steps": max_success_streak >= 10,
            "first_stable_success_confirmation_step": first_stable_success,
            "first_native_failure_after_success": first_loss_after_success,
            "peak_cube_height": peak_row["cube_z_after"],
            "peak_cube_height_step": peak_row["step"],
            "first_negative_gripper_command_after_peak": negative_after_peak,

        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print("GIF:", (output / "rollout.gif").resolve())

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
