"""只读阶段诊断：完整预测块与控制步边界状态，不增加前向或仿真步。"""
import json
import numpy as np


class PhaseDiagnostics:
    def __init__(self, output, env):
        self.env = env
        self.gripper = env.robots[0].gripper['right']
        names = self.gripper.important_geoms
        groups = {key: names[key] for key in ['left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad']}
        groups['cube'] = env.cube.contact_geoms
        groups['table'] = [env.model.mujoco_arena.table_collision.get('name')]
        self.groups = {key: {env.sim.model.geom_name2id(n) for n in values} for key, values in groups.items()}
        if any(not values or min(values) < 0 for values in self.groups.values()):
            raise ValueError('诊断接触几何解析失败')
        (output/'diagnostics_metadata.json').write_text(json.dumps({
            'geom_names': groups, 'geom_ids': {k: sorted(v) for k,v in self.groups.items()},
            'contact_sampling': 'control-step boundary only; no substep sampling or contact force',
            'quaternion': 'xyzw from observations',
            'prediction_index': 't=executed steps before prediction; row k would execute at step t+k+1',
            'gripper_target': 'PandaGripper.current_action; accumulated normalized target, not force',
        }, indent=2))
        self.predictions = (output/'predictions.jsonl').open('x')
        self.states = (output/'physical_states.jsonl').open('x')

    def prediction(self, t, chunk_id, normalized, raw, clipped, execution_horizon, remaining, obs):
        arrays = [np.asarray(a) for a in [normalized, raw, clipped]]
        if any(a.shape != arrays[0].shape or a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() for a in arrays):
            raise ValueError('预测块须为有限K×7矩阵')
        record = dict(t=t, chunk_id=chunk_id, K=len(clipped), H=execution_horizon,
                      selected_offsets=list(range(min(execution_horizon, remaining))),
                      normalized=arrays[0].tolist(), raw=arrays[1].tolist(), clipped=arrays[2].tolist(),
                      eef_position=obs['robot0_eef_pos'].tolist(), eef_quaternion_xyzw=obs['robot0_eef_quat'].tolist(),
                      gripper_qpos=obs['robot0_gripper_qpos'].tolist())
        self.predictions.write(json.dumps(record)+'\n')
        self.predictions.flush()

    def snapshot(self, obs):
        pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in self.env.sim.data.contact[:self.env.sim.data.ncon]}
        def touches(a, b):
            return any(frozenset((i,j)) in pairs for i in self.groups[a] for j in self.groups[b])
        contacts = {a+'_'+b: touches(a,b) for a in ['left_finger','right_finger','left_fingerpad','right_fingerpad'] for b in ['cube','table']}
        robot = self.env.robots[0]
        return dict(eef_position=obs['robot0_eef_pos'].tolist(), eef_quaternion_xyzw=obs['robot0_eef_quat'].tolist(),
                    cube_position=obs['cube_pos'].tolist(), cube_quaternion_xyzw=obs['cube_quat'].tolist(),
                    gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
                    gripper_qvel=np.asarray(self.env.sim.data.qvel[robot._ref_gripper_joint_vel_indexes['right']]).tolist(),
                    gripper_target=np.asarray(self.gripper.current_action).tolist(), contacts=contacts)

    def step(self, step, source, action, before, after):
        self.states.write(json.dumps(dict(step=step, action_source=source, action=np.asarray(action).tolist(), before=before, after=after))+'\n')
        self.states.flush()

    def close(self):
        self.predictions.close()
        self.states.close()
