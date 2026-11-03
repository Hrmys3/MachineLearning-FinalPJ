"""Lift 保持协议；指标仅使用原生高度条件，不等同于接触稳定性。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


class FirstLiftMetric:
    """首次连续10步确认的抬升，回溯起点，40步连续达标才成功。"""

    def __init__(self):
        self.step = 0
        self.streak = 0
        self.start = None
        self.failure = None
        self.duration = 0

    def update(self, success):
        self.step += 1
        self.streak = self.streak + 1 if success else 0
        if self.start is None and self.streak >= 10:
            self.start = self.step - 9
        if self.start is not None and self.failure is None:
            if success:
                self.duration = self.step - self.start + 1
            else:
                self.failure = self.step

    def result(self):
        passed = self.duration >= 40
        status = (
            "success" if passed else
            "height_condition_lost" if self.failure is not None else
            "insufficient_window" if self.start is not None else
            "no_confirmed_lift"
        )
        return {
            "first_lift_step": self.start,
            "first_lift_confirmation_step": self.start + 9 if self.start is not None else None,
            "first_lift_hold_2s_success": passed,
            "first_lift_hold_status": status,
            "first_lift_height_loss_step": self.failure,
            "first_lift_hold_steps": self.duration,
            "first_lift_hold_seconds": self.duration / 20,
        }


class HoldController:
    """固定切换时的世界系位姿，用当前 OSC 缩放逆变换生成增量动作。"""

    def __init__(self):
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
            self.position - obs["robot0_eef_pos"],
            (self.rotation * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec(),
        ])
        c = self.controller
        lower, upper = np.asarray(c.output_min), np.asarray(c.output_max)
        if np.any(upper <= lower):
            raise ValueError("OSC 输出缩放范围无效")
        normalized = (delta - lower) / (upper - lower) * (c.input_max - c.input_min) + c.input_min
        return np.r_[np.clip(normalized, c.input_min, c.input_max), 1.0]


def parse_args(batch=False):
    parser = argparse.ArgumentParser(description="ACT Lift 验证；act_hold 使用物体真值触发规则保持")
    parser.add_argument("--policy", choices=["act", "act_hold"], default="act")
    parser.add_argument("--output", type=Path, required=True, help="独立实验目录，不复用已有逐轮目录")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/act_lift20_candidate/checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=100)
    if not batch:
        parser.add_argument("--index", type=int, default=int(os.environ.get("EVAL_INDEX", "0")))
    args = parser.parse_args()
    if args.steps < 40:
        parser.error("--steps 至少为40，才能观察2秒保持")
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
        "split": "valid",
        "splits_sha256": hashlib.sha256(Path("data/lift_act/splits.json").read_bytes()).hexdigest(),
        "training_seed": 42 if digest.hexdigest() == "3732458e0391ce32c1d3298f1408d0eeeeefce83ceb51164f40e9c160b2b3241" else None,
        "evaluation_source_snapshot": {
            str(path): path.read_text() for path in sorted(Path("scripts/eval").glob("*.py"))
        },
        "control_freq": 20,
        "prediction_horizon": 20,
        "execution_horizon": 5,
        "rollout_steps": args.steps,
        "lift_confirmation_steps": 10,
        "hold_required_steps": 40,
        "hold_metric": "first confirmed lift, backdated start; strict native height; no retry; confirmation included",
        "hold_trigger": "native object-height ground truth" if args.policy == "act_hold" else None,
        "hold_rule": "fixed world eef pose feedback, closed gripper, no return to ACT" if args.policy == "act_hold" else None,
    }
