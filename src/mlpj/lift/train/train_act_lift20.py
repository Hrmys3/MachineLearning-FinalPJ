"""从统一配置训练PH Lift ACT；保存最后一次更新，不覆盖已有目录。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
from mlpj.common.paths import PROJECT_ROOT as _PROJECT_ROOT
_project_os.chdir(_PROJECT_ROOT)

import time
_SCRIPT_STARTED = time.perf_counter()

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

root = Path("third_party/lift/act_lift").resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "detr"))

from detr.main import get_args_parser
from detr.models import build_ACT_model

from mlpj.lift.config import load_config
from mlpj.common.performance import Timings, HardwareMonitor, precision_context, profiler_context, runtime_info, summary

BASE = Path("data/lift/lift_act")
CAMERAS = ["agentview", "robot0_eye_in_hand"]


class LiftDataset(Dataset):
    def __init__(self, episode_ids, stats, chunk=20, image_size=(128, 128), cache_mode="none"):
        self.stats, self.chunk, self.image_size = stats, chunk, image_size
        self.samples, self.cache = [], {}
        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.state_mean, self.state_std = np.asarray(stats["state_mean"]), np.asarray(stats["state_std"])
        self.action_mean, self.action_std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
        for episode_id in episode_ids:
            path = BASE / "episodes" / f"episode_{episode_id:04d}.hdf5"
            with h5py.File(path, "r") as f:
                self.samples.extend((path, t) for t in range(len(f["action"])))
                if cache_mode != "none":
                    data = {"state": f["state"][:], "action": f["action"][:]}
                    raw = np.stack([f[f"images/{c}"][:] for c in CAMERAS], axis=1)
                    if cache_mode == "processed_ram":
                        data["processed"] = torch.stack([self.preprocess(images) for images in raw])
                    else:
                        data["images"] = raw
                    self.cache[path] = data

    def preprocess(self, images):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
        images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        return (images - self.mean) / self.std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, t = self.samples[index]
        if path in self.cache:
            data = self.cache[path]
            state, future = data["state"][t], data["action"][t:t+self.chunk]
            images = data["processed"][t] if "processed" in data else self.preprocess(data["images"][t])
        else:
            with h5py.File(path, "r") as f:
                state = f["state"][t]
                images = self.preprocess(np.stack([f[f"images/{c}"][t] for c in CAMERAS]))
                future = f["action"][t:t+self.chunk]
        state = (state - self.state_mean) / self.state_std
        actions = np.zeros((self.chunk, 7), dtype=np.float32)
        actions[:len(future)] = (future - self.action_mean) / self.action_std
        return (torch.tensor(state, dtype=torch.float32), images,
                torch.from_numpy(actions), torch.from_numpy(np.arange(self.chunk) >= len(future)))


def make_loader(dataset, settings, shuffle):
    options = dict(batch_size=settings["batch_size"], shuffle=shuffle,
                   num_workers=settings["num_workers"], pin_memory=settings["pin_memory"])
    if settings["num_workers"]:
        options.update(persistent_workers=settings["persistent_workers"],
                       prefetch_factor=settings["prefetch_factor"])
    return DataLoader(dataset, **options)


def masked_l1(prediction, actions, is_pad):
    valid = (~is_pad).unsqueeze(-1).expand_as(actions)
    return (prediction - actions).abs()[valid].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lift/default.yaml"))
    cli = parser.parse_args()
    config = load_config(cli.config)
    t, m = config["train"], config["model"]
    output = Path(t["output"])
    if output.exists():
        raise FileExistsError(f"{output} 已存在，请配置新的训练输出目录")
    os.environ.setdefault("ACT_PRETRAINED", "1" if t["pretrained_backbone"] else "0")
    output.mkdir(parents=True, exist_ok=False)
    with HardwareMonitor(output / "hardware.csv", config["performance"]["hardware_monitor_enabled"],
                         config["performance"]["hardware_sample_seconds"]):
        run_training(config, output)


def run_training(config, output):
    t, m, perf = config["train"], config["model"], config["performance"]
    timings = Timings(perf["timing_enabled"], perf["cuda_events"])
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.cuda.manual_seed_all(t["seed"])
    if perf.get("torch_threads") is not None:
        torch.set_num_threads(perf["torch_threads"])
    torch.cuda.reset_peak_memory_stats()
    splits = json.loads((BASE / "splits.json").read_text())
    stats = json.loads((BASE / f"stats_train_{t['budget']}.json").read_text())
    with timings.stage("dataset_setup"):
        train = LiftDataset(splits[f"train_{t['budget']}"], stats, m["chunk_size"], tuple(m["image_size"]), t["cache_mode"])
        valid = LiftDataset(splits["valid"], stats, m["chunk_size"], tuple(m["image_size"]), t["cache_mode"])
        train_loader = make_loader(train, t, True)
        valid_loader = make_loader(valid, t, False)

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
    (output / "config.json").write_text(json.dumps({
        "parameters": config, "effective_pretrained": os.environ["ACT_PRETRAINED"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_source": Path(__file__).read_text(),
        "config_loader_source": (_PROJECT_ROOT / "src/mlpj/lift/config.py").read_text(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
    }, indent=2))

    with timings.stage("model_setup"):
        model = build_ACT_model(args).cuda()
    forward_model = torch.compile(model) if t["compile_model"] else model
    scaler = torch.amp.GradScaler("cuda", enabled=t["precision"] == "amp_fp16")
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
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    steady_start = training_start if perf["warmup_updates"] == 0 else None
    step_times = []
    losses = []
    optimizer_steps = [0]
    def count_optimizer_step(*unused):
        optimizer_steps[0] += 1
    optimizer.register_step_post_hook(count_optimizer_step)
    with profiler_context(perf, output) as profiler:
        for step in range(1, t["updates"] + 1):
            step_start = time.perf_counter()
            with timings.stage("data_wait"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
            with timings.stage("transfer", gpu=True):
                state, images, actions, is_pad = [x.to("cuda", non_blocking=t["non_blocking_transfer"]) for x in batch]
            with timings.stage("forward_loss", gpu=True):
                with precision_context(t["precision"]):
                    prediction, _, (mu, logvar) = forward_model(state, images, None, actions, is_pad)
                # KL指数和损失归约保持FP32，避免半精度溢出。
                l1 = masked_l1(prediction.float(), actions.float(), is_pad)
                mu, logvar = mu.float(), logvar.float()
                kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=-1).mean()
                loss = l1 + t["kl_weight"] * kl
                assert torch.isfinite(loss), "Non-finite training loss"
            with timings.stage("backward_update", gpu=True):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=t["precision"] != "amp_fp16")
                scaler.step(optimizer)
                scaler.update()
            if step <= 3 or step == t["updates"] or step % 100 == 0:
                losses.append({"step": step, "loss": loss.item(), "l1": l1.item(), "kl": kl.item()})
                print(losses[-1], flush=True)
            step_times.append(time.perf_counter() - step_start)
            if step == perf["warmup_updates"]:
                torch.cuda.synchronize()
                steady_start = time.perf_counter()
            if profiler is not None:
                profiler.step()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_start
    steady_updates = max(0, t["updates"] - perf["warmup_updates"])
    steady_seconds = time.perf_counter() - steady_start if steady_start is not None and steady_updates else None

    # 验证时不提供真实未来动作给 CVAE 编码器。
    # 使用推理时的 z=0；统计全部验证样本的有效动作元素。
    model.eval()
    error_sum = 0.0
    element_count = 0
    validation_start = time.perf_counter()
    with torch.no_grad():
        for batch in valid_loader:
            state, images, actions, is_pad = [x.to("cuda", non_blocking=t["non_blocking_transfer"]) for x in batch]
            with precision_context(t["precision"]):
                prediction, _, _ = forward_model(state, images, None)
            assert torch.isfinite(prediction).all()
            mask = (~is_pad).unsqueeze(-1).expand_as(actions)
            error_sum += (prediction.float() - actions).abs()[mask].sum().item()
            element_count += mask.sum().item()

    validation_l1 = error_sum / element_count
    validation_seconds = time.perf_counter() - validation_start
    save_start = time.perf_counter()
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

    save_seconds = time.perf_counter() - save_start
    report = {**timings.report(), **runtime_info(),
              "kind": "training", "parameters": config,
              "training_seconds": training_seconds,
              "updates": t["updates"], "skipped_optimizer_updates": t["updates"] - optimizer_steps[0],
              "successful_optimizer_updates": optimizer_steps[0], "batch_size": t["batch_size"],
              "updates_per_second": t["updates"] / training_seconds,
              "samples_per_second": t["updates"] * t["batch_size"] / training_seconds,
              "steady_updates": steady_updates, "steady_seconds": steady_seconds,
              "update_submit_wall": summary(step_times),
              "steady_samples_per_second": steady_updates*t["batch_size"]/steady_seconds if steady_seconds else None,
              "validation_seconds": validation_seconds, "checkpoint_seconds": save_seconds,
              "validation_l1": validation_l1, "loss_samples": losses,
              "script_wall_seconds": time.perf_counter() - _SCRIPT_STARTED}
    (output / "performance.json").write_text(json.dumps(report, indent=2))
    print("Validation normalized L1:", validation_l1)
    print("Peak GPU memory (MiB):",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("PASS: candidate training, validation and checkpoint saved.")


if __name__ == "__main__":
    main()
