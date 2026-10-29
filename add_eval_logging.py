from pathlib import Path

source = Path("eval_act_debug.py").read_text(encoding="utf-8")


def replace_once(old, new):
    global source
    assert source.count(old) == 1, \
        f"未找到唯一替换位置，请勿继续：{old}"
    source = source.replace(old, new, 1)


replace_once("import json", "import json\nimport csv")

replace_once(
    '        frames = [Image.fromarray(image_array(obs, "agentview"))]',
    '''        frames = [Image.fromarray(image_array(obs, "agentview"))]
        trace = []
        success_streak = 0
        max_success_streak = 0
        first_stable_success = None
        first_loss_after_success = None
        chunk_id = 0
'''
)

replace_once(
    "            chunk = predict(obs)",
    """            chunk = predict(obs)
            chunk_id += 1"""
)

replace_once(
    "            for action in chunk[:5]:",
    "            for chunk_offset, action in enumerate(chunk[:5]):"
)

replace_once(
    "                obs, reward, done, info = env.step(action)",
    """                before_eef = obs["robot0_eef_pos"].copy()
                before_cube = obs["cube_pos"].copy()
                before_gripper = obs["robot0_gripper_qpos"].copy()
                obs, reward, done, info = env.step(action)"""
)

replace_once(
    "                success = bool(env._check_success())",
    '''                success = bool(env._check_success())
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
'''
)

replace_once(
    '        output = Path("outputs/act_debug")',
    '        output = Path("outputs/act_logged")'
)

replace_once(
    '        output.mkdir(parents=True, exist_ok=True)',
    '''        output.mkdir(parents=True, exist_ok=True)

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
'''
)

replace_once(
    '            "image_flips": flips,',
    '''            "image_flips": flips,
            "execution_horizon": 5,
            "max_native_success_streak_steps": max_success_streak,
            "max_native_success_streak_seconds": max_success_streak / 20,
            "stable_native_success_10_steps": max_success_streak >= 10,
            "first_stable_success_confirmation_step": first_stable_success,
            "first_native_failure_after_success": first_loss_after_success,
            "peak_cube_height": peak_row["cube_z_after"],
            "peak_cube_height_step": peak_row["step"],
            "first_negative_gripper_command_after_peak": negative_after_peak,
'''
)

compile(source, "eval_act_logged.py", "exec")
Path("eval_act_logged.py").write_text(source, encoding="utf-8")
print("已生成 eval_act_logged.py")
