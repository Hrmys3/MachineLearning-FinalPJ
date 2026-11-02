# ACT Lift

基于 MuJoCo、robosuite 和 Panda 机械臂的 Lift 模仿学习项目，使用 RoboMimic PH 示范数据和适配后的官方 ACT 实现。

## 目录结构

```text
scripts/
├── checks/       # 环境、相机与模型检查
├── data/         # 数据回放、检查、划分与缓存
├── train/        # 训练入口
├── eval/         # 推理、评测与可视化
├── experts/      # 脚本专家与示范采集
└── tools/        # ACT 源码适配工具
experiments/      # 实验配置与小型结果记录
docs/             # 项目文档
env/              # 环境说明与依赖快照
archive/          # 历史辅助脚本与文档
data/             # 下载数据、数据划分与训练缓存
outputs/          # 权重、日志、评测记录和动画
third_party/      # 上游代码及适配副本
```

详细说明见 [仓库结构](docs/REPOSITORY_LAYOUT.md)。

## 运行环境

使用 Linux 或 WSL2 Ubuntu，并在独立 Python 环境中安装依赖。核心版本组合为：

- Python 3.11
- robosuite 1.5.1
- MuJoCo 3.3.0
- Mink 0.0.13
- 相互匹配的 PyTorch 和 torchvision

环境说明见 [env/README.md](env/README.md)。`env/snapshots/` 中的文件是历史依赖快照，可能包含本机路径，不应直接作为通用安装清单。

以下命令均从项目根目录执行。环境名和项目路径可按实际情况替换：

```bash
conda activate ml
cd ~/ml/pj-arm
```

## 基础检查

先检查无相机仿真，再检查相机与模型计算：

```bash
python scripts/checks/test_lift.py
MUJOCO_GL=glfw python scripts/checks/test_lift_camera.py
ACT_PRETRAINED=0 python scripts/checks/test_act_model.py
```

`MUJOCO_GL=glfw` 指定图形后端，需要可用的显示环境。无显示服务器应根据驱动和系统条件配置适合的离屏渲染方式。

## 数据准备

### 数据来源

- [社区 RoboMimic PH 数据](https://huggingface.co/datasets/chaoqi-liu/robomimic_mt4_ph)：图像、状态、动作与轨迹索引。
- [官方 RoboMimic 数据](https://huggingface.co/datasets/robomimic/robomimic_datasets)：环境配置、模型与仿真状态。

安装下载工具后获取数据：

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    repo_id="chaoqi-liu/robomimic_mt4_ph",
    repo_type="dataset",
    local_dir="data/robomimic_mt4_ph",
    max_workers=2,
)
hf_hub_download(
    repo_id="robomimic/robomimic_datasets",
    repo_type="dataset",
    filename="v1.5/lift/ph/demo_v15.hdf5",
    local_dir="data/robomimic_source",
)
PY
```

### 检查与预处理

```bash
python scripts/data/replay_lift_source.py
python scripts/data/inspect_lift_dataset.py
python scripts/data/prepare_lift_splits.py
python scripts/data/cache_lift_data.py
```

数据划分与归一化参数位于 `data/lift_act/`，逐轨迹缓存位于 `data/lift_act/episodes/`。训练与推理应使用相同的状态字段、相机顺序和归一化参数。

## ACT 源码

官方 ACT 源码版本固定为：

```text
742c753c0d4a5d87076c8f69e5628c79a8cc5488
```

适配代码位于 `third_party/act_lift/`。如需从原始代码重新生成适配副本：

```bash
git clone https://github.com/tonyzhaozh/act.git third_party/act
git -C third_party/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
python scripts/tools/adapt_act.py
```

原始仓库存在时不必重复克隆。适配工具不会覆盖已有 `third_party/act_lift/`，重新生成前应检查并保留自己的修改。使用第三方代码时保留原许可证与来源说明。

## 模型训练

先进行小规模流程检查，再运行训练入口：

```bash
ACT_PRETRAINED=0 python scripts/train/train_act_smoke.py
ACT_PRETRAINED=1 python scripts/train/train_act_lift20.py
```

`ACT_PRETRAINED=1` 在创建模型时加载 ImageNet 视觉骨干权重；`0` 表示不加载该初始化。训练参数和输出目录见对应脚本。

训练脚本可能覆盖配置中的权重输出。开始新实验前，应区分实验目录并保存已有模型。

## 推理与评测

推理入口默认读取：

```text
outputs/act_lift20_candidate/checkpoint.pt
```

此外还需要社区数据元信息、相关轨迹缓存，以及官方源文件：

```text
data/robomimic_source/v1.5/lift/ph/demo_v15.hdf5
```

权重和大体积数据不随普通 Git 仓库提供。运行前需准备对应资源，并按上述路径放置。

单初态推理、批量验证与动画总览：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/eval_act_logged.py
python scripts/eval/run_act_validation.py
python scripts/eval/make_eval_grid.py
```

推理时设置 `ACT_PRETRAINED=0` 是为避免重复下载初始化权重，之后仍会加载 checkpoint 中的完整模型参数。

单初态脚本可能覆盖原输出；批量评测发现已有结果时会停止。重新运行前应明确使用哪个实验目录，不要混合不同模型的结果。

## 文件管理

代码、配置、数据划分和小型结果记录使用 Git 管理；下载数据、训练缓存、权重、日志和动画由 `.gitignore` 排除。

需要共享模型和完整评测附件时，可单独发布资源，并在文档中记录下载地址、对应代码版本及文件校验值。
