"""Push-T ACT训练中的损失处理。"""

import torch


def valid_act_losses(policy_losses, is_pad, kl_weight):
    """将官方ACT的masked L1换算为有效元素平均；只转换一次。

    is_pad须与传给ACT的动作块长度一致。
    """
    if is_pad.dtype != torch.bool or is_pad.ndim != 2:
        raise ValueError("is_pad必须是(batch, horizon)布尔张量")
    valid_count = (~is_pad).sum()
    if valid_count.item() == 0:
        raise ValueError("动作块不能全部是填充")
    if kl_weight < 0:
        raise ValueError("kl_weight不能为负")

    l1 = policy_losses["l1"] * (is_pad.numel() / valid_count)
    kl = policy_losses["kl"]
    return {
        "l1": l1,
        "kl": kl,
        "loss": l1 + kl_weight * kl,
    }
