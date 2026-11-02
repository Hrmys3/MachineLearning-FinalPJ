"""验证 ACT 的 9 维状态、双相机输入、7 维动作与梯度。"""

# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_os.chdir(_PROJECT_ROOT)

import sys
from pathlib import Path

import torch

root = Path("third_party/act_lift").resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "detr"))

from detr.main import get_args_parser
from detr.models import build_ACT_model


def main():
    torch.manual_seed(42)

    # 原版解析器包含必填参数；这里显式提供，不读取外部命令行。
    args = get_args_parser().parse_args([
        "--ckpt_dir", "outputs/act_smoke",
        "--policy_class", "ACT",
        "--task_name", "lift",
        "--seed", "42",
        "--num_epochs", "1",
    ])

    args.camera_names = ["agentview", "robot0_eye_in_hand"]
    args.num_queries = 20
    args.hidden_dim = 256
    args.dim_feedforward = 1024
    args.enc_layers = 4
    args.dec_layers = 4
    args.nheads = 8
    args.backbone = "resnet18"

    model = build_ACT_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    batch_size = 2
    state = torch.randn(batch_size, 9, device="cuda")
    images = torch.rand(batch_size, 2, 3, 128, 128, device="cuda")
    actions = torch.randn(batch_size, 20, 7, device="cuda")
    is_pad = torch.zeros(
        batch_size, 20, dtype=torch.bool, device="cuda"
    )
    # 模拟第二条样本在序列尾部不足 20 步。
    is_pad[1, 15:] = True

    prediction, _, (mu, logvar) = model(
        state, images, None, actions, is_pad
    )
    assert prediction.shape == actions.shape, prediction.shape

    valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
    l1 = (prediction - actions).abs()[valid].mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())) \
        .sum(dim=-1).mean()
    loss = l1 + 10.0 * kl
    assert torch.isfinite(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [
        p.grad for p in model.parameters() if p.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print("Training output:", tuple(prediction.shape))
    print("L1:", l1.item(), "KL:", kl.item())
    print("Loss:", loss.item())

    model.eval()
    with torch.no_grad():
        inference, _, _ = model(state, images, None)

    assert inference.shape == (batch_size, 20, 7)
    assert torch.isfinite(inference).all()
    torch.cuda.synchronize()

    print("Inference output:", tuple(inference.shape))
    print("Peak GPU memory (MiB):",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("PASS: ACT forward, backward and inference completed.")


if __name__ == "__main__":
    main()
