"""保持指标边界检查，无需 CUDA 或仿真。"""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from lift_protocol import FirstLiftMetric


class FirstLiftTests(unittest.TestCase):
    def evaluate(self, sequence):
        metric = FirstLiftMetric()
        for value in sequence:
            metric.update(value)
        return metric.result()

    def test_configurable_confirmation_and_hold_duration(self):
        metric = FirstLiftMetric(confirmation_steps=3, hold_seconds=1.0)
        for _ in range(19):
            metric.update(True)
        self.assertFalse(metric.result()["first_lift_hold_success"])
        metric.update(True)
        result = metric.result()
        self.assertTrue(result["first_lift_hold_success"])
        self.assertEqual(result["first_lift_confirmation_step"], 3)
        self.assertNotIn("first_lift_hold_2s_success", result)

    def test_boundary_and_window(self):
        r = self.evaluate([False] * 61 + [True] * 39)
        self.assertEqual(r["first_lift_step"], 62)
        self.assertEqual(r["first_lift_hold_status"], "insufficient_window")
        self.assertTrue(self.evaluate([False] * 60 + [True] * 40)["first_lift_hold_2s_success"])

    def test_regrasp_cannot_erase_failure(self):
        r = self.evaluate([True] * 10 + [False] + [True] * 60)
        self.assertFalse(r["first_lift_hold_2s_success"])
        self.assertEqual(r["first_lift_height_loss_step"], 11)
        self.assertEqual(r["first_lift_hold_steps"], 10)

    def test_success_then_drop_retains_completed_two_seconds(self):
        r = self.evaluate([True] * 40 + [False])
        self.assertTrue(r["first_lift_hold_2s_success"])
        self.assertEqual(r["first_lift_height_loss_step"], 41)

    def test_short_transient_is_not_confirmed_lift(self):
        r = self.evaluate([True] * 9 + [False] * 91)
        self.assertEqual(r["first_lift_hold_status"], "no_confirmed_lift")


if __name__ == "__main__":
    unittest.main()
