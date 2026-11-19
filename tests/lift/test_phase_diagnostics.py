"""编号选择与预测日志对齐验证，不运行仿真。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mlpj.lift.config import load_config, validate_diagnostics
from mlpj.lift.eval.lift_protocol import select_episode_ids
from mlpj.lift.eval.phase_diagnostics import PhaseDiagnostics


class PhaseTests(unittest.TestCase):
    def test_selection_preserves_ids_and_rejects_invalid(self):
        self.assertEqual(select_episode_ids([0,2,11], {'episode_ids':[11,0]}), [11,0])
        with self.assertRaises(ValueError):
            select_episode_ids([0,2], {'episode_ids':[11]})
        settings = load_config()['eval']
        for ids,limit in [([1,1],None),([],None),([True],None),([1],2)]:
            settings.update(episode_ids=ids, episode_limit=limit)
            with self.assertRaises(ValueError):
                validate_diagnostics(settings)

    def test_prediction_records_full_chunk_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            d=PhaseDiagnostics.__new__(PhaseDiagnostics)
            path=Path(directory)/'predictions.jsonl'
            d.predictions=path.open('w')
            actions=np.arange(140,dtype=float).reshape(20,7)/100
            original=actions.copy()
            obs={'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0,0,0,1]),'robot0_gripper_qpos':np.zeros(2)}
            d.prediction(98,50,actions,actions,np.clip(actions,-1,1),5,2,obs)
            with self.assertRaises(ValueError):
                d.prediction(0,1,actions[:,:6],actions,actions,2,100,obs)
            d.predictions.close()
            row=json.loads(path.read_text())
            self.assertEqual(row['t'],98)
            self.assertEqual(row['selected_offsets'],[0,1])
            self.assertEqual(len(row['clipped']),20)
            np.testing.assert_array_equal(actions,original)

    def test_boundary_contacts_use_exact_geom_pairs(self):
        from types import SimpleNamespace
        d=PhaseDiagnostics.__new__(PhaseDiagnostics)
        d.groups={'left_finger':{1,2},'right_finger':{3},'left_fingerpad':{2},'right_fingerpad':{3},'cube':{9},'table':{10}}
        d.gripper=SimpleNamespace(current_action=np.array([1.,-1.]))
        robot=SimpleNamespace(_ref_gripper_joint_vel_indexes={'right':[0,1]})
        data=SimpleNamespace(contact=[SimpleNamespace(geom1=9,geom2=2),SimpleNamespace(geom1=3,geom2=10)],ncon=2,qvel=np.array([.1,-.1]))
        d.env=SimpleNamespace(sim=SimpleNamespace(data=data),robots=[robot])
        obs={key:np.zeros(n) for key,n in [('robot0_eef_pos',3),('robot0_eef_quat',4),('cube_pos',3),('cube_quat',4),('robot0_gripper_qpos',2)]}
        state=d.snapshot(obs)
        self.assertTrue(state['contacts']['left_fingerpad_cube'])
        self.assertTrue(state['contacts']['right_finger_table'])
        self.assertFalse(state['contacts']['right_finger_cube'])
        self.assertEqual(state['gripper_qvel'],[.1,-.1])
