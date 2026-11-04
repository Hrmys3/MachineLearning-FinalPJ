
# Resolve project-relative paths from the repository root.
import os as _project_os
from pathlib import Path as _ProjectPath
from mlpj.common.paths import PROJECT_ROOT as _PROJECT_ROOT
_project_os.chdir(_PROJECT_ROOT)
from pathlib import Path
import shutil

source = Path("third_party/lift/act")
target = Path("third_party/lift/act_lift")

if target.exists():
    raise SystemExit(
        "third_party/lift/act_lift 已存在，停止以避免覆盖，请先检查。"
    )

shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"))

path = target / "detr/models/detr_vae.py"
text = path.read_text()

replacements = {
    "self.action_head = nn.Linear(hidden_dim, state_dim)":
        "self.action_head = nn.Linear(hidden_dim, 7)",
    "self.input_proj_robot_state = nn.Linear(14, hidden_dim)":
        "self.input_proj_robot_state = nn.Linear(9, hidden_dim)",
    "self.encoder_action_proj = nn.Linear(14, hidden_dim)":
        "self.encoder_action_proj = nn.Linear(7, hidden_dim)",
    "self.encoder_joint_proj = nn.Linear(14, hidden_dim)":
        "self.encoder_joint_proj = nn.Linear(9, hidden_dim)",
}
for old, new in replacements.items():
    assert old in text, f"找不到预期源码: {old}"
    text = text.replace(old, new)

compile(text, str(path), "exec")
path.write_text(text)

# 用当前 torchvision 权重接口；仅在显式设置环境变量后下载权重。
path = target / "detr/models/backbone.py"
text = path.read_text()
old = "pretrained=is_main_process(),"
assert old in text
text = "import os\n" + text
text = text.replace(
    old,
    'weights=(torchvision.models.get_model_weights(name).DEFAULT '
    'if os.environ.get("ACT_PRETRAINED", "0") == "1" else None),',
)
compile(text, str(path), "exec")
path.write_text(text)

print("PASS: 已创建适配副本", target.resolve())
