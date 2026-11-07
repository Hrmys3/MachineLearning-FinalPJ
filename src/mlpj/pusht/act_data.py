"""Push-T ACT 数据读取与动作块处理。"""

import numpy as np

def slice_action_chunk(actions, start, horizon):
    """从 actions[start] 开始取动作；is_pad=True 表示填充位置。"""
    actions = np.asarray(actions)
    if actions.ndim != 2 or len(actions) == 0:
        raise ValueError("actions 必须是非空的二维数组")
    if not 0 <= start < len(actions):
        raise ValueError("start 超出轨迹范围")
    if horizon <= 0:
        raise ValueError("horizon 必须为正数")

    valid_count = min(horizon, len(actions) - start)
    chunk = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    chunk[:valid_count] = actions[start:start + valid_count]

    is_pad = np.ones(horizon, dtype=bool)
    is_pad[:valid_count] = False
    return chunk, is_pad


def compute_norm_stats(episodes, train_ids):
    """episodes[id] 为 (state, action)；仅统计 train_ids 的真实帧。"""
    train_ids = list(train_ids)
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("训练编号必须非空且不重复")

    states, actions = [], []
    for episode_id in train_ids:
        state, action = episodes[episode_id]
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        if state.ndim != 2 or state.shape != action.shape:
            raise ValueError(f"episode {episode_id}: 状态和动作形状不匹配")
        if len(state) == 0 or state.shape[1] != 14:
            raise ValueError(f"episode {episode_id}: 需要非空的14维数据")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"episode {episode_id}: 存在非有限数值")
        states.append(state)
        actions.append(action)

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    if len(states) < 2:
        raise ValueError("计算统计量至少需要两个真实帧")

    stats = {}
    for prefix, values in (("qpos", states), ("action", actions)):
        stats[f"{prefix}_mean"] = values.mean(axis=0).astype(np.float32)
        stats[f"{prefix}_std"] = np.maximum(
            values.std(axis=0, ddof=1), 1e-2
        ).astype(np.float32)
    return stats


def load_joint_episodes(data_dir, episode_ids):
    """按官方编号读取数值字段，返回 {id: (state, action)}。"""
    from pathlib import Path
    import h5py

    episode_ids = list(episode_ids)
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode编号必须非空且不重复")
    if any(not isinstance(i, (int, np.integer)) or i < 0 for i in episode_ids):
        raise ValueError("episode编号必须为非负整数")

    paths = {
        i: Path(data_dir) / f"episode_{i:07d}.hdf5"
        for i in episode_ids
    }
    missing = [i for i, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少episode: {missing}")

    fields = (
        ("left_arm_joint_states", 6),
        ("left_ee_joint_states", 1),
        ("right_arm_joint_states", 6),
        ("right_ee_joint_states", 1),
    )
    episodes = {}
    for episode_id, path in paths.items():
        with h5py.File(path, "r") as file:
            if file["additional_info/frequency"][()] != 25:
                raise ValueError(f"episode {episode_id}: 预期25Hz")
            packed = []
            for group in ("state", "action"):
                parts = [file[f"{group}/{key}"][:] for key, _ in fields]
                n = len(parts[0])
                if n == 0 or any(
                    part.shape != (n, width)
                    for part, (_, width) in zip(parts, fields)
                ):
                    raise ValueError(f"episode {episode_id}: {group}字段形状错误")
                values = np.concatenate(parts, axis=1)
                if not np.isfinite(values).all():
                    raise ValueError(f"episode {episode_id}: 存在非有限数值")
                packed.append(values)
            if packed[0].shape != packed[1].shape:
                raise ValueError(f"episode {episode_id}: 状态和动作长度不同")
            episodes[episode_id] = tuple(packed)
    return episodes


ACT_CAMERA_NAMES = ("cam_head", "cam_right_wrist", "cam_left_wrist")


def load_rgb_observation(path, timestep):
    """返回float32 RGB图像，形状为(camera, channel, height, width)。"""
    import h5py
    from XPolicyLab.utils.process_data import decode_image_bit

    images = []
    with h5py.File(path, "r") as file:
        for camera in ACT_CAMERA_NAMES:
            encoded = file[f"vision/{camera}/colors"]
            if not 0 <= timestep < len(encoded):
                raise IndexError(f"{camera}: timestep超出范围")
            image = np.asarray(decode_image_bit(encoded[timestep]))
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{camera}: 图像形状或类型错误")
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(image)
    return np.stack(images)


def make_act_sample(data_dir, episodes, episode_id, timestep, horizon, stats):
    """返回NumPy格式的(images, qpos, actions, is_pad)。"""
    from pathlib import Path

    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        value = np.asarray(stats[key])
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError(f"{key}: 需要有限的14维统计量")
        if key.endswith("_std") and np.any(value <= 0):
            raise ValueError(f"{key}: 标准差必须为正")

    state, actions = episodes[episode_id]
    chunk, is_pad = slice_action_chunk(actions, timestep, horizon)

    qpos = (
        (state[timestep] - stats["qpos_mean"]) / stats["qpos_std"]
    ).astype(np.float32)
    chunk = (
        (chunk - stats["action_mean"]) / stats["action_std"]
    ).astype(np.float32)
    chunk[is_pad] = 0.0

    path = Path(data_dir) / f"episode_{episode_id:07d}.hdf5"
    images = load_rgb_observation(path, timestep)
    return images, qpos, chunk, is_pad
