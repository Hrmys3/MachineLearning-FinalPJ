"""使用物体真值执行 Lift 脚本基线，保存 GIF、指标及姿态诊断。"""

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
    # 批量评测不打印完整姿态矩阵。

    # 使用 OSC 对应的实际末端 site 姿态，而不是假设观测四元数
    # 与控制器参考点完全一致。
    site_id = env.robots[0].eef_site_id["right"]

    def current_rotation():
        matrix = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
        return Rotation.from_matrix(matrix)

    cube_rotation = Rotation.from_quat(obs["cube_quat"])
    current = current_rotation()

    # 末端 z 轴向下，水平轴对齐方块边。
    # 正方形具有四重对称性，选择相对当前姿态旋转最小的候选。
    downward = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))
    candidates = [
        cube_rotation
        * Rotation.from_euler("z", k * np.pi / 2)
        * downward
        for k in range(4)
    ]
    target_rotation = min(
        candidates,
        key=lambda candidate: (candidate * current.inv()).magnitude(),
    )
    print("Initial orientation error (deg):",
          np.degrees((target_rotation * current.inv()).magnitude()))


    frames = []
    initial_z = float(obs["cube_pos"][2])
    step = 0
    native_success = False
    stable_steps = 0
    max_stable_steps = 0
    first_success_step = None

    def capture():
        pass

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

            # 世界坐标系旋转误差；保持同一个目标姿态，避免持续追踪
            # 已被碰动的方块而在抓取过程中反复转腕。
            rotation_error = (
                target_rotation * current_rotation().inv()
            ).as_rotvec()
            action[3:6] = np.clip(0.8 * rotation_error, -0.15, 0.15)
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


    angle_error = np.degrees(
        (target_rotation * current_rotation().inv()).magnitude()
    )
    print("Orientation error before descent (deg):", angle_error)
    if angle_error > 10:
        raise RuntimeError("朝向尚未对齐，停止下降；请检查诊断输出。")

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

    gif_frames = 0
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
    controller["body_parts"]["right"]["input_ref_frame"] = "world"

    env = None
    try:
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_names="agentview",
            camera_heights=128,
            camera_widths=128,
            use_object_obs=True,
            control_freq=20,
            horizon=500,
        )

        results = []
        for seed in range(1000, 1050):
            print(f"\nEpisode seed={seed}")
            result = run_episode(env, seed, controller)
            results.append(result)
            checkpoint = Path("outputs/lift_aligned_eval50_results.json")
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

            print(json.dumps(result, indent=2))

        output = Path("outputs")
        output.mkdir(exist_ok=True)
        path = output / "lift_aligned_eval50_results.json"
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
