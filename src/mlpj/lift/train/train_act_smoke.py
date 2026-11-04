"""使用真实 Lift 数据验证 ACT 训练链路，非正式效果实验。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
from mlpj.common.paths import PROJECT_ROOT as _PROJECT_ROOT
_project_os.chdir(_PROJECT_ROOT)

import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

root = Path("third_party/lift/act_lift").resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "detr"))

from detr.main import get_args_parser
from detr.models import build_ACT_model

BASE = Path("data/lift/lift_act")
CAMERAS = ["agentview", "robot0_eye_in_hand"]
CHUNK = 20


class LiftDataset(Dataset):
    def __init__(self, episode_ids, stats):
        self.stats = stats
        self.samples = []
        for episode_id in episode_ids:
            path = BASE / "episodes" / f"episode_{episode_id:04d}.hdf5"
            with h5py.File(path, "r") as f:
                self.samples.extend(
                    (path, t) for t in range(len(f["action"]))
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, t = self.samples[index]
        with h5py.File(path, "r") as f:
            state = f["state"][t]
            images = np.stack([f[f"images/{c}"][t] for c in CAMERAS])
            future = f["action"][t:t + CHUNK]

        state = (
            state - np.asarray(self.stats["state_mean"])
        ) / np.asarray(self.stats["state_std"])

        actions = np.zeros((CHUNK, 7), dtype=np.float32)
        actions[:len(future)] = (
            future - np.asarray(self.stats["action_mean"])
        ) / np.asarray(self.stats["action_std"])

        is_pad = np.arange(CHUNK) >= len(future)

        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        images = F.interpolate(
            images, size=(128, 128),
            mode="bilinear", align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        images = (images - mean) / std

        return (
            torch.tensor(state, dtype=torch.float32),
            images,
            torch.from_numpy(actions),
            torch.from_numpy(is_pad),
        )


def masked_l1(prediction, actions, is_pad):
    valid = (~is_pad).unsqueeze(-1).expand_as(actions)
    return (prediction - actions).abs()[valid].mean()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    splits = json.loads((BASE / "splits.json").read_text())
    stats = json.loads((BASE / "stats_train_20.json").read_text())

    train = LiftDataset(splits["train_20"], stats)
    valid = LiftDataset(splits["valid"], stats)

    # 单进程加载，避免一开始引入额外多进程问题。
    train_loader = DataLoader(
        train, batch_size=8, shuffle=True, num_workers=0,
    )
    valid_loader = DataLoader(
        valid, batch_size=8, shuffle=False, num_workers=0,
    )

    args = get_args_parser().parse_args([
        "--ckpt_dir", "outputs/lift/act_smoke",
        "--policy_class", "ACT",
        "--task_name", "lift",
        "--seed", "42",
        "--num_epochs", "1",
    ])
    args.camera_names = CAMERAS
    args.num_queries = CHUNK
    args.hidden_dim = 256
    args.dim_feedforward = 1024
    args.enc_layers = 4
    args.dec_layers = 4
    args.nheads = 8
    args.backbone = "resnet18"

    model = build_ACT_model(args).cuda()
    optimizer = torch.optim.AdamW([
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" not in n
            ],
            "lr": 1e-4,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" in n
            ],
            "lr": 1e-5,
        },
    ], weight_decay=1e-4)

    print("Train frames:", len(train), "Valid frames:", len(valid))
    iterator = iter(train_loader)
    model.train()

    for step in range(1, 101):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        state, images, actions, is_pad = [
            x.cuda() for x in batch
        ]
        prediction, _, (mu, logvar) = model(
            state, images, None, actions, is_pad
        )
        l1 = masked_l1(prediction, actions, is_pad)
        kl = (
            -0.5 * (1 + logvar - mu.square() - logvar.exp())
        ).sum(dim=-1).mean()
        loss = l1 + 10 * kl
        assert torch.isfinite(loss), "Non-finite training loss"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True,
        )
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(
                f"step={step:03d} loss={loss.item():.4f} "
                f"l1={l1.item():.4f} kl={kl.item():.4f}",
                flush=True,
            )

    # 验证时不提供真实未来动作给 CVAE 编码器。
    # 使用推理时的 z=0；统计全部验证样本的有效动作元素。
    model.eval()
    error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for batch in valid_loader:
            state, images, actions, is_pad = [x.cuda() for x in batch]
            prediction, _, _ = model(state, images, None)
            assert torch.isfinite(prediction).all()
            mask = (~is_pad).unsqueeze(-1).expand_as(actions)
            error_sum += (prediction - actions).abs()[mask].sum().item()
            element_count += mask.sum().item()

    validation_l1 = error_sum / element_count
    output = Path("outputs/lift/act_smoke")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_args": vars(args),
        "stats": stats,
        "camera_names": CAMERAS,
        "image_size": [128, 128],
        "image_normalization": "ImageNet mean/std",
        "train_episode_ids": splits["train_20"],
        "valid_episode_ids": splits["valid"],
        "steps": 100,
        "validation_l1": validation_l1,
    }, output / "checkpoint.pt")

    print("Validation normalized L1:", validation_l1)
    print("Peak GPU memory (MiB):",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("PASS: real-data training, validation and checkpoint saved.")


if __name__ == "__main__":
    main()
