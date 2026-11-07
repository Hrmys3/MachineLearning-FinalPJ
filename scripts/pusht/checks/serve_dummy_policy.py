"""为真实 EvalEnv 单步检查提供本机确定性 WebSocket 策略。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/pusht/RoboDojo/XPolicyLab"))

import numpy as np
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


class DummyPolicy:
    def __init__(self):
        self.observed = None
        self.resets = 0
        self.requests = 0
        self.summary = {}

    def reset(self):
        self.resets += 1

    def update_obs(self, obs):
        self.observed = obs
        vision = obs.get("vision", {})
        self.summary = {
            "cameras": {name: {"shape": list(np.asarray(value["color"]).shape),
                               "dtype": str(np.asarray(value["color"]).dtype),
                               "finite": bool(np.isfinite(value["color"]).all())}
                        for name, value in vision.items() if "color" in value},
            "state_keys": sorted(obs.get("state", {})),
        }

    def get_action(self):
        if self.observed is None:
            raise RuntimeError("missing observation")
        state = self.observed["state"]
        arms = {}
        for side in ("left", "right"):
            value = np.asarray(state[f"{side}_arm_joint_state"], dtype=float).reshape(-1)
            if value.size != 6 or not np.isfinite(value).all():
                raise ValueError(f"invalid {side} arm state")
            arms[side] = value.copy()
        arms["left"][0] += 0.01
        self.requests += 1
        return [{"left_arm_joint_state": arms["left"].tolist(),
                 "left_ee_joint_state": [float(state["left_ee_joint_state"][0])],
                 "right_arm_joint_state": arms["right"].tolist(),
                 "right_ee_joint_state": [float(state["right_ee_joint_state"][0])]}]


async def main(args):
    policy = DummyPolicy()
    server = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "port.json").write_text(json.dumps({"url": server.url}) + "\n")
    print("READY", server.url, flush=True)
    try:
        await stop.wait()
    finally:
        await server.stop()
        (args.output / "result.json").write_text(json.dumps({
            "resets": policy.resets, "action_requests": policy.requests,
            **policy.summary}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), "127.0.0.1", "localhost"]))
    asyncio.run(main(args))
