"""项目配置读取与基本约束，无仿真/GPU依赖。"""
from copy import deepcopy
from pathlib import Path
import runpy
import math

DEFAULT_CONFIG = Path('configs/lift.yaml')


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            raise ValueError('YAML配置顶层必须是映射')
    elif path.suffix.lower() == '.py':
        # 保留旧实验parameters.py兼容；只加载本地可信配置。
        config = deepcopy(runpy.run_path(str(path))['CONFIG'])
    else:
        raise ValueError('配置仅支持.yaml、.yml或历史.py快照')
    e, r, m, t = (config[k] for k in ['eval', 'random_set', 'model', 'train'])
    if e['control_freq'] != 20:
        raise ValueError('当前PH动作语义只验证20Hz，禁止直接修改control_freq')
    for value in [r['count'], e['steps'], e['execution_horizon'], e['hold_trigger_steps'],
                  e['metric_confirmation_steps'], m['chunk_size'], t['updates'], t['batch_size']]:
        if not isinstance(value, int) or value <= 0:
            raise ValueError('数量与步数必须为正整数')
    if e['initial_states'] not in ['random', 'valid'] or e['policy'] not in ['act', 'act_hold']:
        raise ValueError('未知初始化或策略')
    hold_steps = round(e['metric_hold_seconds'] * e['control_freq'])
    if e['metric_hold_seconds'] <= 0 or abs(hold_steps - e['metric_hold_seconds'] * 20) > 1e-9:
        raise ValueError('保持时长须对应整数控制步')
    if e['metric_confirmation_steps'] > hold_steps or e['steps'] < hold_steps:
        raise ValueError('确认步数不能超过保持要求，评测不能少于保持要求')
    if e['hold_trigger_height'] <= 0 or min(e['hold_position_gain'], e['hold_rotation_gain']) <= 0:
        raise ValueError('高度与保持增益必须为正数')
    positive = [e['hold_trigger_height'], e['hold_position_gain'], e['hold_rotation_gain'],
                t['lr'], t['lr_backbone'], t['gradient_clip'],
                m['hidden_dim'], m['dim_feedforward'], m['enc_layers'], m['dec_layers'], m['nheads'],
                *m['image_size']]
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError('学习率、梯度裁剪、网络维度、图像尺寸及保持参数必须为正有限数值')
    if not 0 <= m['dropout'] < 1 or min(t['weight_decay'], t['kl_weight']) < 0:
        raise ValueError('dropout须在[0,1)，正则权重不能为负')
    if t['budget'] not in [20, 50] or m['hidden_dim'] % m['nheads']:
        raise ValueError('预算只支持20/50，hidden_dim须能被nheads整除')
    return config
