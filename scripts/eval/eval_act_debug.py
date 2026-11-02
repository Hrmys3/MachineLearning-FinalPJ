"""单个验证初态的 ACT 闭环调试；不作为正式成功率结果。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)

import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
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


def main():
    # checkpoint 是我们自己生成的可信本地文件。
    ckpt = torch.load(
        "outputs/act_lift20_candidate/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = build_ACT_model(Namespace(**ckpt["model_args"])).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = ckpt["stats"]

    splits = json.loads((BASE / "splits.json").read_text())
    episode_id = splits["valid"][0]
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
            horizon=250,
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

        frames = [Image.fromarray(image_array(obs, "agentview"))]
        success_any = False
        first_success = None
        step = 0

        while step < 200:
            chunk = predict(obs)
            for action in chunk[:5]:
                obs, reward, done, info = env.step(action)
                step += 1
                success = bool(env._check_success())
                success_any |= success
                if success and first_success is None:
                    first_success = step
                frames.append(Image.fromarray(
                    image_array(obs, "agentview")
                ))

            if step % 20 == 0:
                print("step:", step, "success:", success,
                      "cube_z:", float(obs["cube_pos"][2]), flush=True)

        output = Path("outputs/act_debug")
        output.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output / "rollout.gif",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        result = {
            "validation_episode": episode_id,
            "steps": step,
            "success_any": bool(success_any),
            "success_final": bool(env._check_success()),
            "first_success_step": first_success,
            "image_flips": flips,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print("GIF:", (output / "rollout.gif").resolve())

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
