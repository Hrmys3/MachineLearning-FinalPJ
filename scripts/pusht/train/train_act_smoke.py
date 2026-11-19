"""两条真实示范上的ACT短训练诊断，不属于正式valid20评测。"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-uuid", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--updates", type=int, default=20)
parser.add_argument("--val-every", type=int, default=10)
args = parser.parse_args()
if args.updates <= 0 or args.val_every <= 0:
    parser.error("updates和val-every必须为正数")

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
os.environ["ACT_ACTION_DIM"] = "14"
upstream = ROOT / "third_party/pusht/RoboDojo/XPolicyLab"
sys.path[:0] = [str(upstream / "policy/ACT"), str(upstream)]

import numpy as np
import torch
import yaml
from detr.act_policy import ACTPolicy
from detr.main import get_args_parser
from mlpj.pusht.act_data import (
    ACT_CAMERA_NAMES, load_joint_episodes, compute_norm_stats,
)
from mlpj.pusht.act_dataset import PushTACTDataset, FixedPushTACTDataset
from mlpj.pusht.act_training import valid_act_losses

result = {"ok": False, "scope": "short training diagnostic; not formal validation"}
started = time.perf_counter()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_event(value):
    with (out / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


try:
    torch.set_num_threads(4)
    weight = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not weight.is_file():
        raise FileNotFoundError(f"缺少预训练缓存：{weight}")

    sources = out / "sources"
    sources.mkdir()
    shutil.copy2(__file__, sources / "train_act_smoke.py")
    source_hashes = {}
    for name in ("act_data.py", "act_dataset.py", "act_training.py"):
        path = ROOT / "src/mlpj/pusht" / name
        shutil.copy2(path, sources / name)
        source_hashes[name] = sha256(path)
    shutil.copy2(ROOT / "configs/pusht/splits_v1.json", sources / "splits_v1.json")

    base = ROOT / (
        "data/pusht/robodojo_demos/1127bda94f17/"
        "data/RoboDojo/push_T/arx_x5/data"
    )
    expected = {
        0: "bd4907829ffd6f26c16f099233224282e3f52284023ca7ec11af41a46d623966",
        1: "3984f2223c4ac57946dd9cbf38c8457ebd9c5f80267583c9b406d1e410bd2a3f",
    }
    for episode_id, expected_hash in expected.items():
        path = base / f"episode_{episode_id:07d}.hdf5"
        if sha256(path) != expected_hash:
            raise ValueError(f"episode {episode_id}: SHA256不匹配")

    episodes = load_joint_episodes(base, [0, 1])
    stats = compute_norm_stats(episodes, train_ids=[0])
    with (out / "dataset_stats.pkl").open("wb") as file:
        pickle.dump(stats, file)

    dataset = PushTACTDataset(base, episodes, [0, 1], stats, horizon=50)
    train_positions = [(0, 12), (0, 179), (0, 300), (0, 358)]
    val_positions = [(1, 0), (1, 125), (1, 256), (1, 511)]
    fixed_train = FixedPushTACTDataset(dataset, train_positions)
    fixed_val = FixedPushTACTDataset(dataset, val_positions)

    cfg = get_args_parser().parse_args([
        "--ckpt_dir", str(out), "--policy_class", "ACT",
        "--bench_name", "RoboDojo", "--task_name", "push_T",
        "--seed", "0", "--num_epochs", "1",
    ])
    overrides = dict(
        lr=1e-5, lr_backbone=1e-5, kl_weight=10, chunk_size=50,
        hidden_dim=512, dim_feedforward=3200, backbone="resnet18",
        enc_layers=4, dec_layers=7, nheads=8, batch_size=1,
        camera_names=list(ACT_CAMERA_NAMES),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)

    parameters = {
        "scope": "diagnostic only; episodes 0 and 1 remain in formal training pool",
        "updates": args.updates,
        "val_every": args.val_every,
        "seed": 0,
        "train_episode_ids": [0],
        "stats_episode_ids": [0],
        "diagnostic_val_episode_ids": [1],
        "fixed_train_positions": train_positions,
        "fixed_val_positions": val_positions,
        "validation_seed": 1000,
        "masked_l1": "mean over valid action elements",
        "grad_clip_norm": 1.0,
        "amp": False,
        "model": vars(cfg),
        "data_sha256": expected,
        "source_sha256": source_hashes,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "xpolicylab_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True,
        ).strip(),
    }
    (out / "parameters.yaml").write_text(
        yaml.safe_dump(parameters, allow_unicode=True, sort_keys=False)
    )
    (sources / "xpolicylab.diff").write_text(subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=upstream, text=True,
    ))

    # 读取实时占用后才允许初始化CUDA。
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
    result.update(gpu_uuid=uuid, physical_index=int(index))

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(overrides, model_cfg=cfg)
    optimizer = policy.configure_optimizers()
    rng = np.random.default_rng(0)

    def to_device(sample):
        return tuple(value.unsqueeze(0).cuda() for value in sample)

    def losses_for(sample):
        images, qpos, actions, mask = to_device(sample)
        raw = policy(qpos, images, actions, mask)
        losses = valid_act_losses(raw, mask, kl_weight=10)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise RuntimeError("损失出现非有限数值")
        return losses

    def evaluate(fixed_dataset):
        policy.eval()
        # 恢复验证前的随机状态，避免验证改变后续训练随机序列。
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1000)
            with torch.no_grad():
                values = [
                    {key: value.item() for key, value in losses_for(sample).items()}
                    for sample in fixed_dataset
                ]
        return {
            key: sum(value[key] for value in values) / len(values)
            for key in values[0]
        }

    best_loss = float("inf")
    best_update = None
    initial_val = None

    for update in range(args.updates + 1):
        if update == 0 or update % args.val_every == 0 or update == args.updates:
            train_metrics = evaluate(fixed_train)
            val_metrics = evaluate(fixed_val)
            log_event({
                "kind": "evaluation", "updates": update,
                "fixed_train": train_metrics, "diagnostic_val": val_metrics,
            })
            if initial_val is None:
                initial_val = val_metrics
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_update = update
                torch.save(policy.state_dict(), out / "policy_best.ckpt")
                (out / "best.json").write_text(json.dumps({
                    "updates": update, "diagnostic_val_loss": best_loss,
                }, indent=2) + "\n")
        if update == args.updates:
            break

        policy.train()
        timestep = int(rng.integers(len(episodes[0][0])))
        optimizer.zero_grad(set_to_none=True)
        losses = losses_for(dataset.get_at(0, timestep))
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), 1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        log_event({
            "kind": "training", "updates": update + 1,
            "episode": 0, "timestep": timestep,
            **{key: value.detach().item() for key, value in losses.items()},
            "grad_norm_before_clip": grad_norm.item(),
        })

    policy.eval()
    images, qpos, _, _ = to_device(dataset.get_at(0, 12))
    with torch.no_grad():
        reference = policy(qpos, images).cpu()
    assert tuple(reference.shape) == (1, 50, 14)
    assert torch.isfinite(reference).all()
    torch.save(policy.state_dict(), out / "policy_last.ckpt")

    del optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()

    restored = ACTPolicy(overrides, model_cfg=cfg)
    restored.load_state_dict(torch.load(
        out / "policy_last.ckpt", map_location="cpu", weights_only=True,
    ))
    restored.eval()
    with torch.no_grad():
        prediction = restored(qpos, images).cpu()
    torch.testing.assert_close(prediction, reference, rtol=1e-5, atol=1e-6)

    with (out / "dataset_stats.pkl").open("rb") as file:
        restored_stats = pickle.load(file)
    assert all(np.array_equal(stats[key], restored_stats[key]) for key in stats)
    torch.cuda.synchronize()
    result.update(
        ok=True, updates=args.updates, best_update=best_update,
        initial_diagnostic_val=initial_val,
        final_diagnostic_val=val_metrics,
        checkpoint_reload_max_error=(prediction - reference).abs().max().item(),
        checkpoint_sha256=sha256(out / "policy_last.ckpt"),
        prediction_shape=list(prediction.shape),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
    )
except BaseException as exc:
    result["error"] = repr(exc)
    raise
finally:
    result["elapsed_seconds"] = time.perf_counter() - started
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("EVIDENCE:", out, flush=True)
