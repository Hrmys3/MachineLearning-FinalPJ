"""设备路由回归：读取实际上游方法，不导入Isaac或初始化CUDA。"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/pusht/RoboDojo"


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "upstream-device-routing", "exec"), namespace)


@unittest.skipUnless((UPSTREAM / "env/environment/task_env.py").exists(), "requires pinned RoboDojo checkout and device patch")
class DeviceConfigurationTests(unittest.TestCase):
    def test_cpu_physics_keeps_explicit_cuda_planner(self):
        tree = ast.parse((UPSTREAM / "env/environment/task_env.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TaskEnv")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        nodes = [n for n in init.body if
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "planner_device" for t in n.targets)
                 or isinstance(n, ast.If) and ast.unparse(n.test) == "planner_device is None"]
        for explicit, expected in [(None, "cuda:3"), ("cuda:5", "cuda:5")]:
            config = {"sim": {"device": "cpu"}, "eval_cfg": {"device_id": 3}}
            if explicit is not None:
                config["planner_device"] = explicit
            ns = {"config": config}
            execute(nodes, ns)
            self.assertEqual(ns["planner_device"], expected)
            self.assertEqual(config["sim"]["device"], "cpu")

    def test_robot_manager_forwards_planner_device(self):
        tree = ast.parse((UPSTREAM / "env/robot_manager/robot_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotManager")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_planner")
        calls = []
        ns = {"deepcopy": deepcopy, "CuroboPlanner": lambda **kwargs: calls.append(kwargs) or object()}
        execute([method], ns)
        for device in ["cuda:0", "cuda:3"]:
            manager = SimpleNamespace(device="cpu", planner_device=device, dt=.004, planner={}, ik_solver={})
            robot = SimpleNamespace(robot_type="arm", entity_origin_pose=[0, 0, .1], robot_name="x5",
                                    arm_joints_name=["j1", "j2"], curobo_yml_path="not-opened.yml")
            ns["_setup_planner"](manager, robot)
            self.assertEqual(calls[-1]["device"], device)
            self.assertEqual(calls[-1]["dt"], .004)
            self.assertIs(manager.planner["x5"], manager.ik_solver["x5"])

    def test_capture_buffer_uses_explicit_cuda_device(self):
        tree = ast.parse((UPSTREAM / "env/camera_manager/capture/tiled_capture_manager.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "zeros"]
        self.assertEqual(len(calls), 1)
        devices = [k.value for k in calls[0].keywords if k.arg == "device"]
        self.assertEqual(len(devices), 1)
        expression = ast.Expression(body=devices[0])
        for device in ["cuda:0", "cuda:3"]:
            self.assertEqual(eval(compile(expression, "capture-device", "eval"),
                                  {"self": SimpleNamespace(capture_device=device)}), device)
        tree = ast.parse((UPSTREAM / "env/environment/task_env.py").read_text())
        capture_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == "TiledCaptureManager")
        keyword = next(k for k in capture_call.keywords if k.arg == "capture_device")
        self.assertEqual(ast.unparse(keyword.value), "planner_device")

    def test_cli_routes_rendering_and_disables_multi_gpu(self):
        tree = ast.parse((UPSTREAM / "src/eval_client/main.py").read_text())
        start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "args_cli" for t in n.targets))
        end = next(i for i in range(start + 1, len(tree.body))
                   if isinstance(tree.body[i], (ast.Import, ast.ImportFrom)))
        nodes = tree.body[start + 1:end]
        def error(message):
            raise ValueError(message)
        for device_id in [0, 3]:
            args = SimpleNamespace(device_id=device_id, device="cuda:0", multi_gpu=True)
            execute(nodes, {"args_cli": args, "parser": SimpleNamespace(error=error)})
            self.assertEqual(args.device, f"cuda:{device_id}")
            self.assertFalse(args.multi_gpu)
        with self.assertRaises(ValueError):
            execute(nodes, {"args_cli": SimpleNamespace(device_id=-1), "parser": SimpleNamespace(error=error)})
        with self.assertRaises(ValueError):
            execute(nodes, {"args_cli": SimpleNamespace(device_id=3, device="cpu", device_explicit=True),
                            "parser": SimpleNamespace(error=error)})


if __name__ == "__main__":
    unittest.main()
