# ============================================================
# lift_scripted.py — robosuite Lift 任务脚本化策略测试
# 用预设轨迹让 Panda 完成 pick-and-lift，跑 3 个 episode 并记录成功率
# ============================================================
import json
from pathlib import Path
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
def run_episode(env, seed):
    """跑一个 episode，返回 metrics dict"""
    np.random.seed(seed)
    obs = env.reset()
    initial_z = float(obs["cube_pos"][2])          # 方块初始高度（用于稳定判定）
    step = 0
    native_success = False                         # robosuite 原生成功（任意时刻）
    stable_steps = 0                               # 当前连续稳定抬起步数
    max_stable_steps = 0                           # 历史最大连续稳定步数
    first_success_step = None                      # 首次原生成功步数
    def move(target, gripper, steps):
        """P 控制器：末端追踪 target，gripper 控制夹爪，执行 steps 步"""
        nonlocal obs, step, native_success
        nonlocal stable_steps, max_stable_steps, first_success_step
        for _ in range(steps):
            # OSC_POSE 动作空间 7 维：
            #   [0:3] 末端位置增量 (dx,dy,dz)
            #   [3:6] 末端朝向增量 (本脚本置 0，保持当前朝向)
            #   [6]   夹爪 (-1=开, 1=关)
            error = target - obs["robot0_eef_pos"]
            action = np.zeros(7)
            action[:3] = np.clip(6.0 * error, -0.6, 0.6)   # P 增益 6.0，限幅 0.6
            action[-1] = gripper
            obs, reward, done, info = env.step(action)
            step += 1
            success = bool(env._check_success())
            native_success = native_success or success
            if success and first_success_step is None:
                first_success_step = step
            # 自定义稳定标准：方块比初始高 >= 10cm，连续计数
            lifted = float(obs["cube_pos"][2]) >= initial_z + 0.10
            stable_steps = stable_steps + 1 if lifted else 0
            max_stable_steps = max(max_stable_steps, stable_steps)
            if done:
                raise RuntimeError("Episode horizon reached before script ended.")
    cube = obs["cube_pos"].copy()
    # 阶段 1：approach — 末端移动到方块上方 15cm，夹爪打开
    move(cube + [0, 0, 0.15], gripper=-1, steps=100)
    print("  approach:", obs["robot0_eef_pos"], "cube:", obs["cube_pos"])
    # 阶段 2：descend — 下降到方块上方 5mm
    move(cube + [0, 0, 0.005], gripper=-1, steps=100)
    print("  descend: ", obs["robot0_eef_pos"], "cube:", obs["cube_pos"])
    # 阶段 3：grasp — 闭合夹爪抓取方块
    grasp_target = obs["robot0_eef_pos"].copy()
    move(grasp_target, gripper=1, steps=30)
    # 阶段 4：lift — 抬起 20cm
    move(grasp_target + [0, 0, 0.20], gripper=1, steps=120)
    hold_target = obs["robot0_eef_pos"].copy()
    # 阶段 5：hold — 保持位置 30 步，验证稳定性
    move(hold_target, gripper=1, steps=30)
    return {
        "seed": seed,
        "native_success_any": native_success,              # 任意时刻原生成功
        "native_success_final": bool(env._check_success()), # 最终状态原生成功
        "stable_lift_success": max_stable_steps >= 10,     # 稳定抬起 >= 10 步
        "first_native_success_step": first_success_step,
        "max_stable_steps": max_stable_steps,
        "initial_cube_z": initial_z,
        "final_cube_z": float(obs["cube_pos"][2]),
        "steps": step,
    }
env = None
try:
    # 加载 BASIC composite controller，强制使用 OSC_POSE（末端位姿增量控制）
    controller = load_composite_controller_config(
        controller="BASIC", robot="Panda"
    )
    controller["body_parts"]["right"]["type"] = "OSC_POSE"
    env = suite.make(
        env_name="Lift",                # Lift 任务：机械臂抓起并抬起方块
        robots="Panda",                 # Franka Panda 机械臂
        controller_configs=controller,
        has_renderer=False,             # 无头环境，不打开 viewer
        has_offscreen_renderer=False,   # 不启用离屏渲染（省资源）
        use_camera_obs=False,           # 不返回相机图像（纯状态模式）
        use_object_obs=True,            # 返回方块位姿等物体观测
        control_freq=20,                # 20 Hz 控制频率
        horizon=500,                    # 单 episode 最多 500 步
    )
    results = []
    for seed in range(3):               # 跑 3 个 episode
        print(f"\nEpisode seed={seed}")
        result = run_episode(env, seed)
        results.append(result)
        print(json.dumps(result, indent=2))
    # 保存结果到 outputs/lift_scripted_results.json
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    path = output / "lift_scripted_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # 汇总打印
    print("\nNative final successes:",
          sum(r["native_success_final"] for r in results), "/", len(results))
    print("Stable lift successes:",
          sum(r["stable_lift_success"] for r in results), "/", len(results))
    print("Results saved:", path.resolve())
finally:
    if env is not None:
        env.close()                     # 保证释放 MuJoCo 资源
