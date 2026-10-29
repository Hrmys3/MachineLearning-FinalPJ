"""固定 episode 划分，分别计算各标注预算下的训练集统计量。"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path("data/robomimic_mt4_ph")
output = Path("data/lift_act")
output.mkdir(parents=True, exist_ok=True)

mapping = [
    json.loads(line)
    for line in (root / "meta/robomimic_source_episodes.jsonl")
    .read_text().splitlines()
    if line.strip()
]
ids = sorted(
    row["episode_index"]
    for row in mapping
    if row["source_task_key"] == "lift"
)
assert len(ids) == 200

rng = np.random.default_rng(42)
shuffled = rng.permutation(ids).tolist()

train_pool = shuffled[:160]
valid = shuffled[160:180]
heldout = shuffled[180:]

splits = {
    "seed": 42,
    "train_pool": train_pool,
    "train_20": train_pool[:20],
    "train_50": train_pool[:50],
    "valid": valid,
    "heldout": heldout,
}

assert set(splits["train_20"]).issubset(splits["train_50"])
assert not set(train_pool) & set(valid)
assert not set(train_pool) & set(heldout)
assert not set(valid) & set(heldout)

# 防止以后意外覆盖成另一组划分。
split_path = output / "splits.json"
if split_path.exists():
    assert json.loads(split_path.read_text()) == splits, \
        "已有划分不同，请先检查，不自动覆盖。"
else:
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

episodes = {}
for path in sorted((root / "data").rglob("*.parquet")):
    rows = pq.read_table(
        path,
        columns=[
            "episode_index", "frame_index", "task_index",
            "observation.state", "action",
        ],
    ).to_pylist()
    for row in rows:
        if row["episode_index"] in ids:
            assert row["task_index"] == 0
            episodes.setdefault(row["episode_index"], []).append(row)

for episode_id in ids:
    episodes[episode_id].sort(key=lambda row: row["frame_index"])

for budget in (20, 50):
    selected = splits[f"train_{budget}"]
    rows = [row for i in selected for row in episodes[i]]

    states = np.asarray(
        [r["observation.state"] for r in rows], dtype=np.float64
    )
    actions = np.asarray(
        [r["action"] for r in rows], dtype=np.float64
    )
    assert np.isfinite(states).all() and np.isfinite(actions).all()

    # 小标准差设下限，避免归一化时放大数值噪声。
    stats = {
        "episode_ids": selected,
        "num_frames": len(rows),
        "state_mean": states.mean(axis=0).tolist(),
        "state_std": np.maximum(states.std(axis=0), 1e-2).tolist(),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": np.maximum(actions.std(axis=0), 1e-2).tolist(),
    }
    path = output / f"stats_train_{budget}.json"
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"train_{budget}: {budget} episodes, {len(rows)} frames")

print("valid:", len(valid))
print("heldout:", len(heldout))
print("PASS: splits and training-only statistics saved.")
print("Output:", output.resolve())
