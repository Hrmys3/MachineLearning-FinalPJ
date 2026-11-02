
# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)
from pathlib import Path

import numpy as np
from PIL import Image
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

env = None

try:
    controller = load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        horizon=100,
    )

    obs = env.reset()
    low, _ = env.action_spec

    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros_like(low))

    frame = obs["agentview_image"]
    print("Image shape:", frame.shape)
    print("Image dtype:", frame.dtype)
    print("Pixel range:", frame.min(), frame.max())
    print("Pixel standard deviation:", float(frame.std()))

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    # 同时保存原图与上下翻转版本，方便核对相机图像方向。
    Image.fromarray(frame).save(output / "lift_camera_raw.png")
    Image.fromarray(np.flipud(frame)).save(output / "lift_camera_flipped.png")

    print("PASS: camera frames saved to", output.resolve())

finally:
    if env is not None:
        env.close()
