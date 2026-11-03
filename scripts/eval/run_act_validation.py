"""独立子进程验证；保留异常记录，全部初态计入成功率分母。"""
import os
from pathlib import Path
import json
import subprocess
import sys
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(_PROJECT_ROOT)
from lift_protocol import parse_args, experiment_config, evaluation_ids
from make_eval_grid import make_grids


def main():
    args = parse_args(batch=True)
    episode_ids = evaluation_ids(args)
    config = experiment_config(args)
    # 在创建输出目录和启动100个子进程之前核对模型与执行长度。
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prediction_horizon = checkpoint["model_args"]["num_queries"]
    if args.settings["eval"]["execution_horizon"] > prediction_horizon:
        raise ValueError("执行长度不能超过checkpoint预测长度")
    if checkpoint["camera_names"] != ["agentview", "robot0_eye_in_hand"]:
        raise ValueError("checkpoint相机顺序不匹配")
    config.update(prediction_horizon=prediction_horizon,
                  inference_image_size=checkpoint["image_size"],
                  training_seed=checkpoint.get("training_seed", config["training_seed"]))
    del checkpoint
    config["episode_ids"] = episode_ids
    # 整个实验目录必须是新的；中断后不自动混合或覆盖。
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    frozen_config = args.output / "parameters.yaml"
    frozen_config.write_text(yaml.safe_dump(args.settings, sort_keys=False, allow_unicode=True))
    statuses, results = [], []
    for index, episode_id in enumerate(episode_ids):
        directory = args.output / f"episode_{episode_id:04d}"
        env = os.environ.copy()
        env.update(MUJOCO_GL=env.get("MUJOCO_GL", "glfw"), ACT_PRETRAINED="0", PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(_PROJECT_ROOT / "scripts/eval/eval_act_one.py"),
                   "--index", str(index), "--policy", args.policy,
                   "--output", str(args.output), "--checkpoint", str(args.checkpoint),
                   "--steps", str(args.steps), "--config", str(frozen_config), "--initial-states", args.initial_states]
        # 外层日志在子进程建立逐轮目录前即可保存加载/初始化错误。
        log_path = args.output / f"episode_{episode_id:04d}.log"
        with log_path.open("w") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_path = directory / "result.json"
        completed = process.returncode == 0 and result_path.exists()
        statuses.append({"episode_id": episode_id, "returncode": process.returncode,
                         "result_exists": result_path.exists(), "completed": completed})
        if completed:
            results.append(json.loads(result_path.read_text()))
        (args.output / "process_status.json").write_text(json.dumps(statuses, indent=2))
        fields = ["success_any", "stable_native_success_10_steps", "success_final", "first_lift_hold_success"]
        if args.settings["eval"]["metric_hold_seconds"] == 2.0:
            fields.append("first_lift_hold_2s_success")
        n = len(episode_ids)
        counts = {field: sum(bool(r[field]) for r in results) for field in fields}
        metrics = {"hold_required_seconds": args.settings["eval"]["metric_hold_seconds"], "n": n, "attempted": len(statuses), "completed": len(results),
                   "process_failures": sum(not s["completed"] for s in statuses),
                   "counts": counts, "rates": {k: v / n for k, v in counts.items()},
                   "hold_status_counts": {state: sum(r["first_lift_hold_status"] == state for r in results)
                       for state in ["success", "height_condition_lost", "insufficient_window", "no_confirmed_lift"]}}
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(statuses[-1], flush=True)
    print("结果目录:", args.output.resolve())
    # 即使部分进程失败也生成总览，缺失样本标记在原位置。
    try:
        make_grids(args.output)
    except Exception as exc:
        (args.output / "grid_error.txt").write_text(str(exc))
        raise RuntimeError("推理记录已保存，但GIF拼接失败；参见 grid_error.txt") from exc
    if any(not s["completed"] for s in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
