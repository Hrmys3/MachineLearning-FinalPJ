"""Push-T原始HDF5对应的ACT训练数据集。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mlpj.pusht.act_data import make_act_sample


class PushTACTDataset(Dataset):
    def __init__(self, data_dir, episodes, episode_ids, stats, horizon=50):
        self.data_dir = Path(data_dir)
        self.episode_ids = list(episode_ids)
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode编号必须非空且不重复")
        if horizon <= 0:
            raise ValueError("horizon必须为正数")
        self.episodes = {i: episodes[i] for i in self.episode_ids}
        self.stats = {key: np.asarray(value).copy() for key, value in stats.items()}
        self.horizon = horizon

    def __len__(self):
        return len(self.episode_ids)

    def get_at(self, episode_id, timestep):
        """按指定位置取样，用于确定性检查。"""
        sample = make_act_sample(
            self.data_dir, self.episodes, episode_id, timestep,
            self.horizon, self.stats,
        )
        return tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in sample)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        timestep = int(np.random.randint(len(self.episodes[episode_id][0])))
        return self.get_at(episode_id, timestep)


class FixedPushTACTDataset(Dataset):
    """在指定的(episode_id, timestep)位置取样。"""

    def __init__(self, dataset, positions):
        self.dataset = dataset
        self.positions = list(positions)
        if not self.positions:
            raise ValueError("取样位置不能为空")
        for episode_id, timestep in self.positions:
            if episode_id not in dataset.episodes:
                raise ValueError(f"episode {episode_id} 不在数据集中")
            length = len(dataset.episodes[episode_id][0])
            if not isinstance(timestep, (int, np.integer)):
                raise ValueError("timestep必须为整数")
            if not 0 <= timestep < length:
                raise ValueError(f"episode {episode_id}: timestep超出范围")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        episode_id, timestep = self.positions[index]
        return self.dataset.get_at(episode_id, timestep)
