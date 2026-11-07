"""在策略Python环境中验证XPolicyLab真实WebSocket往返；使用确定性假策略。"""
import argparse
import asyncio
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/pusht/RoboDojo/XPolicyLab"))

import numpy as np
from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state


class DummyPolicy:
    def __init__(self):
        self.observed = None
        self.resets = 0

    def reset(self):
        self.resets += 1

    def update_obs(self, obs):
        self.observed = obs

    def get_action(self):
        if self.observed is None:
            raise RuntimeError("observation was not delivered")
        return [{
            "left_arm_joint_state": [0.01] + [0.] * 5,
            "left_ee_joint_state": [1.],
            "right_arm_joint_state": [0.] * 6,
            "right_ee_joint_state": [1.],
        }]


async def run():
    policy = DummyPolicy()
    server = PolicyServer(policy, PolicyServerConfig(host="127.0.0.1", port=0))
    await server.start()
    try:
        def client_roundtrip():
            with WsModelClient(url=server.url, evaluation_id="pusht-check",
                               trial_id="fixed-layout-0", action_case_id="push_T_case",
                               connect_timeout_s=5, handshake_timeout_s=5,
                               request_timeout_s=10) as client:
                client.call(func_name="reset")
                images = {
                    "cam_head": np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8),
                    "cam_right_wrist": np.full((8, 8, 3), [0, 255, 0], dtype=np.uint8),
                    "cam_left_wrist": np.full((8, 8, 3), [0, 0, 255], dtype=np.uint8),
                }
                obs = {"vision": {name: {"colors": image} for name, image in images.items()},
                       "state": {"left_arm_joint_state": np.zeros(6, dtype=np.float32)}}
                client.call(func_name="update_obs", obs=obs)
                actions = client.call(func_name="get_action")
                return actions, images
        actions, images = await asyncio.wait_for(asyncio.to_thread(client_roundtrip), timeout=20)
        assert policy.resets == 1 and len(actions) == 1
        dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        packed = pack_robot_state({"state": actions[0]}, "joint", dims)
        assert packed.shape == (14,)
        assert np.allclose(packed, [0.01] + [0.] * 5 + [1.] + [0.] * 6 + [1.])
        unpacked = unpack_robot_state(packed, "joint", dims)
        assert all(np.array_equal(unpacked[key], np.asarray(value))
                   for key, value in actions[0].items())
        assert list(actions[0]) == ["left_arm_joint_state", "left_ee_joint_state",
                                    "right_arm_joint_state", "right_ee_joint_state"]
        for name, expected in images.items():
            actual = policy.observed["vision"][name]["colors"]
            assert isinstance(actual, np.ndarray) and actual.dtype == np.uint8
            assert np.array_equal(actual, expected), name
        return {"ok": True, "resets": policy.resets, "cameras": list(images),
                "action_dims": [len(v) for v in actions[0].values()], "packed_dim": int(packed.shape[0]),
                "scope": "real local WebSocket with deterministic dummy policy; no ACT model or simulator"}
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # 本机回环地址应直连；外部代理变量在某些机器上会拦截WebSocket握手。
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        os.environ[key] = ",".join(filter(None, [current, "127.0.0.1", "localhost"]))
    result = asyncio.run(run())
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
