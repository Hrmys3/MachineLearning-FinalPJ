
# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)
import numpy as np
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
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=100,
    )

    obs = env.reset()
    low, high = env.action_spec

    print("Observation keys:", list(obs.keys()))
    print("Action dimension:", len(low))
    print("Action range:", low, high)
    print("Initial cube position:", obs["cube_pos"])

    action = np.zeros_like(low)

    for step in range(20):
        obs, reward, done, info = env.step(action)
        if step % 5 == 0:
            print(
                f"step={step:02d}, reward={reward:.3f}, "
                f"done={done}, cube_pos={obs['cube_pos']}"
            )

    print("PASS: Lift reset and 20 simulation steps completed.")

finally:
    if env is not None:
        env.close()
