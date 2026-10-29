"""恢复官方示范初态，执行记录动作，检查环境与动作兼容性。"""

import json
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite


SOURCE = Path(
    "data/robomimic_source/v1.5/lift/ph/demo_v15.hdf5"
)


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def main():
    env = None
    results = []

    try:
        with h5py.File(SOURCE, "r") as f:
            metadata = json.loads(decode(f["data"].attrs["env_args"]))
            kwargs = metadata["env_kwargs"].copy()
            kwargs.update(
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                lite_physics=False,
            )

            env = suite.make(metadata["env_name"], **kwargs)

            for name in ("demo_0", "demo_1", "demo_2"):
                demo = f["data"][name]
                states = demo["states"][:]
                actions = demo["actions"][:]
                xml = decode(demo.attrs["model_file"])

                # 恢复该条示范的模型和初始状态。
                env.reset()
                xml = env.edit_model_xml(xml)
                env.reset_from_xml_string(xml)
                env.sim.reset()
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()

                # 同步控制器目标与观测缓存。
                env.robots[0].reset(deterministic=True)
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()
                env._update_observables(force=True)

                max_qpos_error = 0.0
                success_any = bool(env._check_success())

                for t, action in enumerate(actions):
                    # 执行动作前，对比当前状态与记录的对应状态。
                    current = env.sim.get_state().flatten()
                    nq = env.sim.model.nq
                    qpos_error = np.max(
                        np.abs(current[1:1+nq] - states[t, 1:1+nq])
                    )
                    max_qpos_error = max(max_qpos_error, float(qpos_error))

                    obs, reward, done, info = env.step(action)
                    success_any = success_any or bool(env._check_success())

                result = {
                    "demo": name,
                    "steps": len(actions),
                    "success_any": success_any,
                    "success_final": bool(env._check_success()),
                    "max_qpos_error": max_qpos_error,
                }
                results.append(result)
                print(json.dumps(result, indent=2), flush=True)

        output = Path("outputs/source_replay_results.json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Saved:", output.resolve())

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
