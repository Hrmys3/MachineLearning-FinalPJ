"""逐进程评测全部验证初态；失败进程单独记录，不静默忽略。"""

import json
import os
import subprocess
import sys
from pathlib import Path

splits = json.loads(Path("data/lift_act/splits.json").read_text())
root = Path("outputs/act_valid100")
root.mkdir(parents=True, exist_ok=True)
statuses = []

for index, episode_id in enumerate(splits["valid"]):
    directory = root / f"episode_{episode_id:04d}"
    directory.mkdir(exist_ok=True)

    # 每次运行保留已有结果，避免旧结果被误当成本次结果。
    result_path = directory / "result.json"
    if result_path.exists():
        raise SystemExit(
            f"{result_path} 已存在。请先检查已有结果，避免重复或混合运行。"
        )

    env = os.environ.copy()
    env.update(
        EVAL_INDEX=str(index),
        MUJOCO_GL="glfw",
        ACT_PRETRAINED="0",
        PYTHONUNBUFFERED="1",
    )

    print(
        f"[{index + 1}/{len(splits['valid'])}] episode={episode_id}",
        flush=True,
    )
    with (directory / "run.log").open("w") as log:
        process = subprocess.run(
            [sys.executable, "-u", "eval_act_one.py"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    status = {
        "episode_id": episode_id,
        "returncode": process.returncode,
        "result_exists": result_path.exists(),
    }
    statuses.append(status)
    (root / "process_status.json").write_text(
        json.dumps(statuses, indent=2), encoding="utf-8"
    )
    print(status, flush=True)

print("全部进程已结束，结果目录:", root.resolve())
