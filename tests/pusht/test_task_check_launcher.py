"""检查启动器的资源守卫与结果保存，不导入仿真或占用GPU。"""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlpj.pusht import task_checks


class TaskCheckLauncherTests(unittest.TestCase):
    def test_gpu_uuid_and_busy_checks(self):
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), device_id=3, gpu_uuid="GPU-test")
            for row, accepted in [("3, GPU-test, 2, 0\n", True),
                                  ("3, GPU-test, 2048, 0\n", False),
                                  ("3, GPU-test, 2, 80\n", False),
                                  ("3, GPU-other, 2, 0\n", False)]:
                with patch.dict(os.environ, {}, clear=True), patch.object(task_checks.subprocess, "check_output", return_value=row):
                    if accepted:
                        task_checks.check_idle_gpu(args, "gpu.txt")
                    else:
                        with self.assertRaises(ValueError):
                            task_checks.check_idle_gpu(args, "gpu.txt")
                self.assertEqual((Path(tmp) / "gpu.txt").read_text(), row)

    def test_direct_worker_does_not_start_simulation(self):
        argv = ["check", "--worker", "--output", "/unused", "--device-id", "3", "--gpu-uuid", "GPU-test"]
        with patch.object(task_checks.sys, "argv", argv), patch.dict(os.environ, {}, clear=True), \
             patch.object(task_checks, "worker") as worker, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                task_checks.main()
            self.assertEqual(caught.exception.code, 2)
            worker.assert_not_called()

    def test_omegaconf_and_array_result_serialization(self):
        try:
            import numpy as np
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("requires simulation environment data dependencies")
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            task_checks.save_json(p, {"robot": OmegaConf.create({"joints": ["j1", "j2"]}),
                                     "values": np.array([1., 2.])})
            self.assertEqual(json.loads(p.read_text()),
                             {"robot": {"joints": ["j1", "j2"]}, "values": [1., 2.]})


if __name__ == "__main__":
    unittest.main()
