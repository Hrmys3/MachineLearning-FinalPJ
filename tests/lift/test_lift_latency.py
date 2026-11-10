"""抬升时延汇总必须区分未观测与成功，且允许空批次。"""
import unittest
from mlpj.lift.eval.run_act_validation import lift_step_summary


class LiftLatencyTests(unittest.TestCase):
    def test_success_only_and_incomplete_denominator(self):
        rows = [{'first_lift_step': 40}, {'first_lift_step': 60}, {'first_lift_step': None}]
        result = lift_step_summary(rows, 'first_lift_step', 4, 20)
        self.assertEqual(result['observed'], 2)
        self.assertEqual(result['not_observed_or_incomplete'], 2)
        self.assertEqual(result['mean_steps'], 50)
        self.assertEqual(result['median_seconds'], 2.5)

    def test_no_lift_has_no_latency(self):
        result = lift_step_summary([{}], 'first_lift_step', 2, 20)
        self.assertEqual(result['observed'], 0)
        self.assertIsNone(result['mean_steps'])
        self.assertIsNone(result['min_steps'])
        self.assertEqual(result['not_observed_or_incomplete'], 2)
