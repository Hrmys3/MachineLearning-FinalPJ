"""读取实际上游EvalEnv方法，检查14维joint动作到物理目标的映射。"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'third_party/pusht/RoboDojo/src/eval_client/eval_env.py'


def methods():
    tree = ast.parse(SOURCE.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_eval_env')
    cls = next(n for n in factory.body if isinstance(n, ast.ClassDef) and n.name == 'EvalEnv')
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                {'validate_action_dict', 'get_action_type', 'take_action_batch', 'process_control_info'}]
    ns = {'deepcopy': deepcopy, 'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), 'exec'), ns)
    return ns


@unittest.skipUnless(SOURCE.is_file(), 'requires pinned RoboDojo source')
class EvalActionMappingTests(unittest.TestCase):
    def test_two_arm_joint_action_and_interpolation(self):
        ns = methods()
        class Queue:
            def __init__(self):self.records = []
            def push(self, indices, sequences):self.records.append((indices, sequences))
        robots = [SimpleNamespace(type='target', arm_name=f'{side}_arm', gripper_name=f'{side}_ee',
                                  ee_type='gripper', gripper_scale=[-.01,.044],
                                  gripper_move={'sign':1,'mimic':['joint8',1.,0.]})
                  for side in ['left','right']]
        queue = Queue()
        manager = SimpleNamespace(robot_list=robots, process_name=lambda name:name+'_joint_state',
                                  get_joint=lambda robot,env_idx_list:{0:np.zeros(6)},
                                  get_end_effector_real_val=lambda robot,env_idx_list:{0:np.array([.044,.044])},
                                  control_manager=queue)
        cls=type('Methods',(),{key:ns[key] for key in ['validate_action_dict','get_action_type',
                                                       'take_action_batch','process_control_info']})
        env=cls();env.robot_manager=manager;env.robot_action_dim_info={'arm_dim':[6,6],'ee_dim':[1,1]}
        env.physx_monitor_enabled=False;env.num_envs=1;env.take_action_cnt=[0];env.step_lim=600;env.end_flag=[False]
        env.obs_manager=SimpleNamespace(collect_interval=10);env.reward_manager=SimpleNamespace(step=lambda **kwargs:None)
        env.have_empty=lambda indices:True;env.is_episode_end=lambda:False
        action={'left_arm_joint_state':[.02,0,0,0,0,0],'left_ee_joint_state':[.5],
                'right_arm_joint_state':[0]*6,'right_ee_joint_state':[0.]}
        env.take_action_batch([action],[0])
        self.assertEqual(env.take_action_cnt,[1])
        indices,sequences=queue.records[0]
        self.assertEqual(indices,[0]);self.assertEqual(len(sequences[0]),10)
        first,last=sequences[0][0],sequences[0][-1]
        self.assertAlmostEqual(first['left_arm_joint_state']['position'][0],.02/9)
        self.assertAlmostEqual(last['left_arm_joint_state']['position'][0],.02)
        self.assertAlmostEqual(last['left_ee_joint_state']['position'][0],.017)
        self.assertAlmostEqual(last['right_ee_joint_state']['position'][0],-.01)
        for side in ['left','right']:
            key=f'{side}_ee_joint_state'
            self.assertEqual(len(last[key]['position']),2)
            self.assertAlmostEqual(last[key]['position'][0],last[key]['position'][1])
        with self.assertRaises(ValueError):
            env.validate_action_dict({**action,'left_arm_joint_state':[0]*5})


if __name__ == '__main__':unittest.main()
