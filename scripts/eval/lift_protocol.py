"""Lift 保持协议；指标仅使用原生高度条件，不等同于接触稳定性。"""
import sys
import argparse
import hashlib
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import load_config
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


class FirstLiftMetric:
    """首次连续确认的抬升，回溯起点；确认/保持要求由配置决定。"""

    def __init__(self, confirmation_steps=10, hold_seconds=2.0, control_freq=20):
        self.confirmation_steps = confirmation_steps
        self.hold_steps = round(hold_seconds * control_freq)
        self.hold_seconds = hold_seconds
        self.control_freq = control_freq
        self.step = 0
        self.streak = 0
        self.start = None
        self.failure = None
        self.duration = 0

    def update(self, success):
        self.step += 1
        self.streak = self.streak + 1 if success else 0
        if self.start is None and self.streak >= self.confirmation_steps:
            self.start = self.step - self.confirmation_steps + 1
        if self.start is not None and self.failure is None:
            if success:
                self.duration = self.step - self.start + 1
            else:
                self.failure = self.step

    def result(self):
        passed = self.duration >= self.hold_steps
        status = (
            "success" if passed else
            "height_condition_lost" if self.failure is not None else
            "insufficient_window" if self.start is not None else
            "no_confirmed_lift"
        )
        return {
            "first_lift_step": self.start,
            "first_lift_confirmation_step": self.start + self.confirmation_steps - 1 if self.start is not None else None,
            "first_lift_hold_success": passed,
            **({"first_lift_hold_2s_success": passed} if self.hold_seconds == 2.0 else {}),
            "hold_required_seconds": self.hold_seconds,
            "first_lift_hold_status": status,
            "first_lift_height_loss_step": self.failure,
            "first_lift_hold_steps": self.duration,
            "first_lift_hold_seconds": self.duration / self.control_freq,
        }


class HoldController:
    """固定切换时的世界系位姿，用当前 OSC 缩放逆变换生成增量动作。"""

    def __init__(self, position_gain=1.0, rotation_gain=1.0):
        self.position_gain = position_gain
        self.rotation_gain = rotation_gain
        self.position = None

    def activate(self, obs, controller):
        if (controller.name != "OSC_POSE" or controller.input_type != "delta"
                or controller.input_ref_frame != "world"
                or controller.impedance_mode != "fixed"):
            raise ValueError("保持规则只支持 fixed/world/delta OSC_POSE")
        self.position = obs["robot0_eef_pos"].copy()
        self.quaternion = obs["robot0_eef_quat"].copy()
        self.rotation = Rotation.from_quat(self.quaternion)
        self.controller = controller
        # 反馈误差以当前实际位姿为基准，不能累加到上一个目标。
        controller.set_goal_update_mode("achieved")

    def action(self, obs):
        delta = np.concatenate([
            self.position_gain * (self.position - obs["robot0_eef_pos"]),
            self.rotation_gain * (self.rotation * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec(),
        ])
        c = self.controller
        lower, upper = np.asarray(c.output_min), np.asarray(c.output_max)
        if np.any(upper <= lower):
            raise ValueError("OSC 输出缩放范围无效")
        normalized = (delta - lower) / (upper - lower) * (c.input_max - c.input_min) + c.input_min
        return np.r_[np.clip(normalized, c.input_min, c.input_max), 1.0]


def parse_args(batch=False):
    parser = argparse.ArgumentParser(description="ACT Lift 验证；act_hold 使用物体真值触发规则保持")
    parser.add_argument("--config", type=Path, default=Path("configs/lift.yaml"))
    parser.add_argument("--initial-states", choices=["random", "valid"])
    parser.add_argument("--policy", choices=["act", "act_hold"])
    parser.add_argument("--output", type=Path, required=True, help="独立实验目录，不复用已有逐轮目录")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int)
    if not batch:
        parser.add_argument("--worker-indices", help="内部：同一worker顺序评测的索引列表")
        parser.add_argument("--index", type=int, default=int(os.environ.get("EVAL_INDEX", "0")))
    args = parser.parse_args()
    args.settings = load_config(args.config)
    e = args.settings["eval"]
    for name in ["policy", "steps", "checkpoint", "initial_states"]:
        value = getattr(args, name)
        key = name
        if value is not None:
            e[key] = str(value) if isinstance(value, Path) else value
        else:
            setattr(args, name, Path(e[key]) if name == "checkpoint" else e[key])
    if args.steps < round(e["metric_hold_seconds"] * 20):
        parser.error("推理步数少于保持评价所需步数")
    return args


def experiment_config(args):
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "split": "random_development" if args.initial_states == "random" else "valid",
        "parameters": args.settings,
        "random_set_sha256": hashlib.sha256(Path(args.settings["random_set"]["path"]).read_bytes()).hexdigest() if args.initial_states == "random" else None,
        "splits_sha256": hashlib.sha256(Path("data/lift_act/splits.json").read_bytes()).hexdigest(),
        "training_seed": 42 if digest.hexdigest() == "3732458e0391ce32c1d3298f1408d0eeeeefce83ceb51164f40e9c160b2b3241" else None,
        "evaluation_source_snapshot": {
            str(path): path.read_text() for path in sorted([*Path("scripts/eval").glob("*.py"), Path("scripts/project_config.py"), Path("scripts/performance.py")])
        },
        "control_freq": args.settings["eval"]["control_freq"],
        "prediction_horizon": "from checkpoint model_args.num_queries",
        "execution_horizon": args.settings["eval"]["execution_horizon"],
        "rollout_steps": args.steps,
        "lift_confirmation_steps": args.settings["eval"]["metric_confirmation_steps"],
        "hold_required_steps": round(args.settings["eval"]["metric_hold_seconds"] * 20),
        "hold_metric": "first confirmed lift, backdated start; strict native height; no retry; confirmation included",
        "hold_trigger": "configured object-height ground truth" if args.policy == "act_hold" else None,
        "hold_rule": "fixed world eef pose feedback, closed gripper, no return to ACT" if args.policy == "act_hold" else None,
    }


def evaluation_ids(args):
    if args.initial_states == "valid":
        return json.loads(Path("data/lift_act/splits.json").read_text())["valid"]
    import h5py
    path = Path(args.settings["random_set"]["path"])
    manifest = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("随机集校验值不匹配")
    with h5py.File(path) as f:
        count = len(f["states"])
    if count != args.settings["random_set"]["count"] or manifest["seed"] != args.settings["random_set"]["seed"]:
        raise ValueError("配置数量/种子与已冻结随机集不符；请生成新的数据集版本")
    return list(range(count))
