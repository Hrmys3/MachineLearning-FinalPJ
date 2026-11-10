"""训练入口CLI覆盖与YAML回退，不执行训练。"""
import sys
import unittest
from unittest.mock import patch
from mlpj.lift.config import load_config
from mlpj.lift.train.train_act_lift import parse_training_config


class TrainingCliTests(unittest.TestCase):
    def test_yaml_fallback(self):
        with patch.object(sys, 'argv', ['train']):
            config = parse_training_config()
        self.assertEqual(config, load_config())

    def test_budget_and_output_override(self):
        original = load_config()
        with patch.object(sys, 'argv', ['train', '--budget', '50', '--output', 'outputs/lift/new_cli_test']):
            config = parse_training_config()
        self.assertEqual(config['train']['budget'], 50)
        self.assertEqual(config['train']['output'], 'outputs/lift/new_cli_test')
        original['train'].update(budget=50, output='outputs/lift/new_cli_test')
        self.assertEqual(config, original)

    def test_invalid_budget(self):
        with patch.object(sys, 'argv', ['train', '--budget', '30']):
            with self.assertRaises(SystemExit) as caught:
                parse_training_config()
        self.assertEqual(caught.exception.code, 2)
