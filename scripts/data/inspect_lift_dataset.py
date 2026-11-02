"""检查社区数据与官方源动作的一致性，并导出首条示范预览。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("data/robomimic_mt4_ph")
SOURCE = Path("data/robomimic_source/v1.5/lift/ph/demo_v15.hdf5")
OUTPUT = Path("outputs/dataset_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

episode_id = 0
info = json.loads((ROOT / "meta/info.json").read_text())

metadata = []
for path in sorted((ROOT / "meta/episodes").rglob("*.parquet")):
    metadata.extend(pq.read_table(path).to_pylist())

episode = next(
    row for row in metadata if row["episode_index"] == episode_id
)

data_path = ROOT / info["data_path"].format(
    chunk_index=episode["data/chunk_index"],
    file_index=episode["data/file_index"],
)
rows = [
    row for row in pq.read_table(data_path).to_pylist()
    if row["episode_index"] == episode_id
]
rows.sort(key=lambda row: row["frame_index"])

assert len(rows) == episode["length"]
assert [r["frame_index"] for r in rows] == list(range(len(rows)))
assert all(r["task_index"] == 0 for r in rows)

states = np.asarray([r["observation.state"] for r in rows])
actions = np.asarray([r["action"] for r in rows])
assert states.shape == (len(rows), 9)
assert actions.shape == (len(rows), 7)
assert np.isfinite(states).all() and np.isfinite(actions).all()

mapping = [
    json.loads(line)
    for line in (ROOT / "meta/robomimic_source_episodes.jsonl")
    .read_text().splitlines()
    if line.strip()
]
source_row = next(r for r in mapping if r["episode_index"] == episode_id)

with h5py.File(SOURCE, "r") as f:
    original = f["data"][source_row["source_demo_key"]]["actions"][:]

assert original.shape == actions.shape
difference = float(np.max(np.abs(original - actions)))
print("Episode:", episode_id, "frames:", len(rows))
print("Maximum action difference from official source:", difference)
assert np.allclose(original, actions, atol=1e-6, rtol=0)

for camera in ("agentview", "robot0_eye_in_hand"):
    key = f"observation.images.{camera}"
    prefix = f"videos/{key}"
    video_path = ROOT / info["video_path"].format(
        video_key=key,
        chunk_index=episode[f"{prefix}/chunk_index"],
        file_index=episode[f"{prefix}/file_index"],
    )
    start_time = episode[f"{prefix}/from_timestamp"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - info["fps"]) < 0.01, fps
    start_frame = round(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    try:
        for _ in range(len(rows)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Video ended before episode was decoded.")
            # OpenCV 返回 BGR；数据集视频先不额外上下翻转。
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        cap.release()

    frames[0].save(OUTPUT / f"episode_0_{camera}_first.png")
    frames[-1].save(OUTPUT / f"episode_0_{camera}_last.png")
    frames[0].save(
        OUTPUT / f"episode_0_{camera}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    print(camera, "decoded frames:", len(frames), "start frame:", start_frame)

print("PASS: action consistency and video decoding checks completed.")
print("Preview directory:", OUTPUT.resolve())
