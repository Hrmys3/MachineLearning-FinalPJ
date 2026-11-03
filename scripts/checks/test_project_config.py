"""验证YAML配置与历史Python快照的等价读取，不运行仿真。"""
import copy
from pathlib import Path
import pprint
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from project_config import load_config
import yaml


class ConfigTests(unittest.TestCase):
    def test_legacy_snapshot_equivalence(self):
        expected = load_config(ROOT / 'configs/lift.yaml')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parameters.py'
            path.write_text('CONFIG = ' + pprint.pformat(expected))
            self.assertEqual(load_config(path), expected)
            path = Path(directory) / 'parameters.yaml'
            path.write_text(yaml.safe_dump(expected))
            self.assertEqual(load_config(path), expected)

    def test_invalid_protocol_is_rejected(self):
        config = copy.deepcopy(load_config(ROOT / 'configs/lift.yaml'))
        config['eval']['control_freq'] = 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
