"""将选定 Lift 示范转换为逐轨迹 HDF5；保留原始 RGB 分辨率。"""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("data/robomimic_mt4_ph")
OUTPUT = Path("data/lift_act/episodes")
OUTPUT.mkdir(parents=True, exist_ok=True)

info = json.loads((ROOT / "meta/info.json").read_text())
splits = json.loads(Path("data/lift_act/splits.json").read_text())
selected = sorted(set(splits["train_50"] + splits["valid"]))
cameras = ["agentview", "robot0_eye_in_hand"]

metadata = {}
for path in sorted((ROOT / "meta/episodes").rglob("*.parquet")):
    for row in pq.read_table(path).to_pylist():
        metadata[row["episode_index"]] = row

frame_rows = {}
for path in sorted((ROOT / "data").rglob("*.parquet")):
    for row in pq.read_table(path).to_pylist():
        if row["episode_index"] in selected:
            frame_rows.setdefault(row["episode_index"], []).append(row)


def check_file(path, episode_id, length):
    with h5py.File(path, "r") as f:
        assert f.attrs["episode_index"] == episode_id
        assert f["state"].shape == (length, 9)
        assert f["action"].shape == (length, 7)
        assert np.isfinite(f["state"][:]).all()
        assert np.isfinite(f["action"][:]).all()
        for camera in cameras:
            images = f[f"images/{camera}"]
            assert images.shape == (length, 256, 256, 3)
            assert images.dtype == np.uint8
            # 实际读取全部图像，检查压缩数据是否可解码。
            for image in images:
                assert image.shape == (256, 256, 3)


for episode_id in selected:
    meta = metadata[episode_id]
    rows = sorted(
        frame_rows[episode_id], key=lambda row: row["frame_index"]
    )
    length = len(rows)

    assert length == meta["length"]
    assert [r["frame_index"] for r in rows] == list(range(length))
    assert all(r["task_index"] == 0 for r in rows)

    destination = OUTPUT / f"episode_{episode_id:04d}.hdf5"
    if destination.exists():
        check_file(destination, episode_id, length)
        print("Verified existing:", destination.name, flush=True)
        continue

    temporary = destination.with_suffix(".tmp")

    with h5py.File(temporary, "w") as f:
        f.attrs["episode_index"] = episode_id
        f.attrs["fps"] = info["fps"]
        f.attrs["image_format"] = "RGB; no extra vertical flip"
        f.attrs["state_fields"] = "eef_pos,eef_quat_xyzw,gripper_qpos"

        f.create_dataset(
            "state",
            data=np.asarray(
                [r["observation.state"] for r in rows],
                dtype=np.float32,
            ),
        )
        f.create_dataset(
            "action",
            data=np.asarray([r["action"] for r in rows], dtype=np.float32),
        )

        for camera in cameras:
            key = f"observation.images.{camera}"
            prefix = f"videos/{key}"
            video_path = ROOT / info["video_path"].format(
                video_key=key,
                chunk_index=meta[f"{prefix}/chunk_index"],
                file_index=meta[f"{prefix}/file_index"],
            )

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {video_path}")

            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                assert abs(fps - info["fps"]) < 0.01
                first = round(meta[f"{prefix}/from_timestamp"] * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
                assert abs(actual - first) < 0.5, (actual, first)

                dataset = f.create_dataset(
                    f"images/{camera}",
                    shape=(length, 256, 256, 3),
                    dtype=np.uint8,
                    chunks=(1, 256, 256, 3),
                    compression="gzip",
                    compression_opts=1,
                )
                for t in range(length):
                    ok, bgr = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Video ended: episode={episode_id}, frame={t}"
                        )
                    dataset[t] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()

    check_file(temporary, episode_id, length)
    temporary.replace(destination)
    print(f"Saved: {destination.name}, frames={length}", flush=True)

print(f"PASS: {len(selected)} selected episodes cached and verified.")
