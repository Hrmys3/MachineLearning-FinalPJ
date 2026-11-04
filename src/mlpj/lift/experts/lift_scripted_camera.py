"""使用物体真值执行 Lift 脚本基线，保存 GIF、指标及姿态诊断。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
from mlpj.common.paths import PROJECT_ROOT as _PROJECT_ROOT
_project_os.chdir(_PROJECT_ROOT)

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def print_pose_diagnostics(obs, controller):
    """输出姿态；robosuite 的这些四元数观测采用 xyzw 顺序。"""
    cube_rot = Rotation.from_quat(obs["cube_quat"]).as_matrix()
    eef_rot = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()

    print("cube quaternion:", obs["cube_quat"])
    print("eef quaternion:", obs["robot0_eef_quat"])
    # 旋转矩阵的每一列表示一个局部坐标轴在世界坐标系中的方向。
    print("cube axes in world:\n", np.round(cube_rot, 3))
    print("eef axes in world:\n", np.round(eef_rot, 3))
    print(
        "OSC rotation frame:",
        controller["body_parts"]["right"].get(
            "input_ref_frame", "未显式设置"
        ),
    )


def run_episode(env, seed, controller):
    """运行一次固定阶段、位置反馈控制的抓取测试。"""
    np.random.seed(seed)
    obs = env.reset()
    print_pose_diagnostics(obs, controller)

    frames = []
    initial_z = float(obs["cube_pos"][2])
    step = 0
    native_success = False
    stable_steps = 0
    max_stable_steps = 0
    first_success_step = None

    def capture():
        # 当前相机输出上下倒置；复制后保存，避免共享可变图像内存。
        image = np.flipud(obs["agentview_image"]).copy()
        frames.append(Image.fromarray(image))

    def move(target, gripper, steps):
        """用位置误差生成归一化 OSC 输入，执行固定数量控制步。"""
        nonlocal obs, step, native_success
        nonlocal stable_steps, max_stable_steps, first_success_step

        for _ in range(steps):
            error = target - obs["robot0_eef_pos"]

            # 7 维 OSC_POSE 输入：
            # 前三维：归一化位置增量，经控制器缩放，不直接等于米。
            # 中间三维：旋转增量置零，不主动改变目标朝向。
            # 最后一维：夹爪，-1 张开，+1 闭合。
            action = np.zeros(7)
            action[:3] = np.clip(6.0 * error, -0.6, 0.6)
            action[-1] = gripper

            obs, reward, done, info = env.step(action)
            step += 1

            # 20 Hz 控制，每两步采一帧，对应约 10 FPS。
            if step % 2 == 0:
                capture()

            success = bool(env._check_success())
            native_success = native_success or success
            if success and first_success_step is None:
                first_success_step = step

            # 自定义标准：中心相对重置时升高 10 cm，连续保持 10 步。
            # 这不是方块底面到桌面的距离。
            lifted = float(obs["cube_pos"][2]) >= initial_z + 0.10
            stable_steps = stable_steps + 1 if lifted else 0
            max_stable_steps = max(max_stable_steps, stable_steps)

            if done:
                raise RuntimeError("脚本结束前已达到 episode 步数上限。")

    capture()
    cube = obs["cube_pos"].copy()

    # 1. 张开夹爪，追踪初始方块中心上方 15 cm 的位置。
    move(cube + [0, 0, 0.15], gripper=-1, steps=100)
    print("  approach:", obs["robot0_eef_pos"], "cube:", obs["cube_pos"])

    # 2. 下降；这是目标位置，不代表实际跟踪误差为零。
    move(cube + [0, 0, 0.005], gripper=-1, steps=100)
    print("  descend: ", obs["robot0_eef_pos"], "cube:", obs["cube_pos"])

    # 3. 保持当前末端位置，闭合夹爪。
    grasp_target = obs["robot0_eef_pos"].copy()
    move(grasp_target, gripper=1, steps=30)

    # 4. 将末端目标提高 20 cm，方块是否随之抬起由实际接触决定。
    move(grasp_target + [0, 0, 0.20], gripper=1, steps=120)

    # 5. 保持当前末端位置与夹爪闭合。
    hold_target = obs["robot0_eef_pos"].copy()
    move(hold_target, gripper=1, steps=30)

    output = Path("outputs/lift")
    output.mkdir(exist_ok=True)
    gif_path = output / f"lift_seed_{seed}.gif"

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    with Image.open(gif_path) as saved:
        gif_frames = saved.n_frames

    print(
        f"  GIF: captured={len(frames)}, "
        f"saved={gif_frames}, path={gif_path.resolve()}"
    )
    if gif_frames <= 1:
        raise RuntimeError("GIF 只有一帧，需要检查相机观测。")

    return {
        "seed": seed,
        "native_success_any": native_success,
        "native_success_final": bool(env._check_success()),
        "stable_lift_success": max_stable_steps >= 10,
        "first_native_success_step": first_success_step,
        "max_stable_steps": max_stable_steps,
        "initial_cube_z": initial_z,
        "final_cube_z": float(obs["cube_pos"][2]),
        "steps": step,
        "gif_frames": gif_frames,
    }


def main():
    controller = load_composite_controller_config(
        controller="BASIC", robot="Panda"
    )
    controller["body_parts"]["right"]["type"] = "OSC_POSE"

    env = None
    try:
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
            use_object_obs=True,
            control_freq=20,
            horizon=500,
        )

        results = []
        for seed in range(3):
            print(f"\nEpisode seed={seed}")
            result = run_episode(env, seed, controller)
            results.append(result)
            print(json.dumps(result, indent=2))

        output = Path("outputs/lift")
        output.mkdir(exist_ok=True)
        path = output / "lift_scripted_camera_results.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        print(
            "\nNative final successes:",
            sum(r["native_success_final"] for r in results),
            "/", len(results),
        )
        print(
            "Stable lift successes:",
            sum(r["stable_lift_success"] for r in results),
            "/", len(results),
        )
        print("Results saved:", path.resolve())

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
