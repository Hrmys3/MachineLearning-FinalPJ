"""从统一配置训练PH Lift ACT；保存最后一次更新，不覆盖已有目录。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)

import argparse
import os
import subprocess
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

root = Path("third_party/act_lift").resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "detr"))

from detr.main import get_args_parser
from detr.models import build_ACT_model

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from project_config import load_config

BASE = Path("data/lift_act")
CAMERAS = ["agentview", "robot0_eye_in_hand"]


class LiftDataset(Dataset):
    def __init__(self, episode_ids, stats, chunk=20, image_size=(128, 128)):
        self.stats = stats
        self.chunk = chunk
        self.image_size = image_size
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
            future = f["action"][t:t + self.chunk]

        state = (
            state - np.asarray(self.stats["state_mean"])
        ) / np.asarray(self.stats["state_std"])

        actions = np.zeros((self.chunk, 7), dtype=np.float32)
        actions[:len(future)] = (
            future - np.asarray(self.stats["action_mean"])
        ) / np.asarray(self.stats["action_std"])

        is_pad = np.arange(self.chunk) >= len(future)

        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        images = F.interpolate(
            images, size=self.image_size,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lift.yaml"))
    cli = parser.parse_args()
    config = load_config(cli.config)
    t, m = config["train"], config["model"]
    output = Path(t["output"])
    if output.exists():
        raise FileExistsError(f"{output} 已存在，请配置新的训练输出目录")
    os.environ.setdefault("ACT_PRETRAINED", "1" if t["pretrained_backbone"] else "0")
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.cuda.manual_seed_all(t["seed"])

    splits = json.loads((BASE / "splits.json").read_text())
    stats = json.loads((BASE / f"stats_train_{t['budget']}.json").read_text())

    train = LiftDataset(splits[f"train_{t['budget']}"], stats, m["chunk_size"], tuple(m["image_size"]))
    valid = LiftDataset(splits["valid"], stats, m["chunk_size"], tuple(m["image_size"]))

    # 单进程加载，避免一开始引入额外多进程问题。
    train_loader = DataLoader(
        train, batch_size=t["batch_size"], shuffle=True, num_workers=0,
    )
    valid_loader = DataLoader(
        valid, batch_size=t["batch_size"], shuffle=False, num_workers=0,
    )

    args = get_args_parser().parse_args([
        "--ckpt_dir", str(output),
        "--policy_class", "ACT",
        "--task_name", "lift",
        "--seed", str(t["seed"]),
        "--num_epochs", "1",
    ])
    args.camera_names = CAMERAS
    args.num_queries = m["chunk_size"]
    for key in ["hidden_dim", "dim_feedforward", "enc_layers", "dec_layers", "nheads", "backbone", "dropout"]:
        setattr(args, key, m[key])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "parameters": config, "effective_pretrained": os.environ["ACT_PRETRAINED"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_source": Path(__file__).read_text(),
        "config_loader_source": (_PROJECT_ROOT / "scripts/project_config.py").read_text(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
    }, indent=2))

    model = build_ACT_model(args).cuda()
    optimizer = torch.optim.AdamW([
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" not in n
            ],
            "lr": t["lr"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and "backbone" in n
            ],
            "lr": t["lr_backbone"],
        },
    ], weight_decay=t["weight_decay"])

    print("Train frames:", len(train), "Valid frames:", len(valid))
    iterator = iter(train_loader)
    model.train()

    for step in range(1, t["updates"] + 1):
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
        loss = l1 + t["kl_weight"] * kl
        assert torch.isfinite(loss), "Non-finite training loss"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), t["gradient_clip"], error_if_nonfinite=True,
        )
        optimizer.step()

        if step == 1 or step % 100 == 0:
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
    torch.save({
        "model": model.state_dict(),
        "pretrained_backbone": os.environ["ACT_PRETRAINED"] == "1",
        "act_source_commit": (
            BASE / "act_source_commit.txt"
        ).read_text().strip(),

        "model_args": vars(args),
        "stats": stats,
        "camera_names": CAMERAS,
        "image_size": m["image_size"],
        "image_normalization": "ImageNet mean/std",
        "train_episode_ids": splits[f"train_{t['budget']}"],
        "valid_episode_ids": splits["valid"],
        "steps": t["updates"],
        "training_seed": t["seed"],
        "parameters": config,
        "validation_l1": validation_l1,
    }, output / "checkpoint.pt")

    print("Validation normalized L1:", validation_l1)
    print("Peak GPU memory (MiB):",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("PASS: candidate training, validation and checkpoint saved.")


if __name__ == "__main__":
    main()
