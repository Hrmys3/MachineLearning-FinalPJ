"""真实示范上的ACT单批次计算检查。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-uuid", required=True)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["ACT_ACTION_DIM"] = "14"

upstream = ROOT / "third_party/pusht/RoboDojo/XPolicyLab"
sys.path[:0] = [str(upstream / "policy/ACT"), str(upstream)]

import numpy as np
import torch
from detr.act_policy import ACTPolicy
from detr.main import get_args_parser
from mlpj.pusht.act_data import (
    ACT_CAMERA_NAMES, load_joint_episodes, compute_norm_stats,
)
from mlpj.pusht.act_dataset import PushTACTDataset

out = Path(tempfile.mkdtemp(
    prefix="act_batch_", dir=ROOT / "outputs/pusht/checks",
))
record = {
    "ok": False,
    "scope": "one diagnostic batch; one optimizer update",
    "gpu_uuid": args.gpu_uuid,
    "episode": 0,
    "timestep": 12,
    "stats_episode_ids": [0, 1],
    "batch_size": 1,
    "amp": False,
}
shutil.copy2(__file__, out / "check_source.py")

try:
    torch.set_num_threads(4)
    np.random.seed(0)
    weight = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not weight.is_file():
        raise FileNotFoundError(f"缺少预训练缓存：{weight}")

    record["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    record["source_sha256"] = {}
    for name in ("act_data.py", "act_dataset.py"):
        source = ROOT / "src/mlpj/pusht" / name
        shutil.copy2(source, out / name)
        record["source_sha256"][name] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()

    base = ROOT / (
        "data/pusht/robodojo_demos/1127bda94f17/"
        "data/RoboDojo/push_T/arx_x5/data"
    )
    episodes = load_joint_episodes(base, [0, 1])
    stats = compute_norm_stats(episodes, train_ids=[0, 1])
    record["stats"] = {key: value.tolist() for key, value in stats.items()}
    dataset = PushTACTDataset(base, episodes, [0, 1], stats, horizon=50)
    images, qpos, actions, mask = dataset.get_at(0, 12)

    cfg = get_args_parser().parse_args([
        "--ckpt_dir", "unused", "--policy_class", "ACT",
        "--bench_name", "RoboDojo", "--task_name", "push_T",
        "--seed", "0", "--num_epochs", "1",
    ])
    overrides = dict(
        lr=1e-5, lr_backbone=1e-5, kl_weight=10, chunk_size=50,
        hidden_dim=512, dim_feedforward=3200, backbone="resnet18",
        enc_layers=4, dec_layers=7, nheads=8,
        camera_names=list(ACT_CAMERA_NAMES),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    record["config"] = vars(cfg)

    # 在任何CUDA初始化之前重新核查实时占用。
    assert not torch.cuda.is_initialized()
    snapshot = subprocess.check_output([
        "nvidia-smi", "-i", args.gpu_uuid,
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    (out / "gpu-before.txt").write_text(snapshot)
    index, uuid, used, utilization = [
        value.strip() for value in snapshot.strip().split(",")
    ]
    if uuid != args.gpu_uuid or int(used) >= 10 or int(utilization) != 0:
        raise RuntimeError(f"GPU不满足空闲条件：{snapshot}")
    record["physical_index"] = int(index)

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(overrides, model_cfg=cfg)
    optimizer = policy.configure_optimizers()
    images, qpos, actions, mask = [
        value.unsqueeze(0).cuda()
        for value in (images, qpos, actions, mask)
    ]

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    losses = policy(qpos, images, actions, mask)
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise RuntimeError("损失存在非有限数值")
    record["losses"] = {
        key: value.detach().item() for key, value in losses.items()
    }

    losses["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), float("inf"), error_if_nonfinite=True,
    )
    if grad_norm.item() <= 0:
        raise RuntimeError("梯度为零")

    tracked = next(
        parameter for parameter in policy.parameters()
        if parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
    )
    before = tracked.detach().clone()
    optimizer.step()
    change = (tracked.detach() - before).abs().max().item()
    if change <= 0:
        raise RuntimeError("优化器更新后参数未改变")
    optimizer.zero_grad(set_to_none=True)

    policy.eval()
    with torch.no_grad():
        prediction = policy(qpos, images)
    assert tuple(prediction.shape) == (1, 50, 14)
    assert torch.isfinite(prediction).all()
    torch.cuda.synchronize()

    record.update(
        ok=True,
        grad_norm=grad_norm.item(),
        tracked_parameter_max_change=change,
        prediction_shape=list(prediction.shape),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
    )
except BaseException as exc:
    record["error"] = repr(exc)
    raise
finally:
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("EVIDENCE:", out)
