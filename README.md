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

## 环境要求

- **操作系统**：Linux；Windows 使用 WSL2 Ubuntu。已验证的发行版为 Ubuntu 22.04，其他系统组合需要单独验证。
- **Python**：3.11，使用独立的 Python 环境；环境名称可自行选择。
- **GPU**：训练和 ACT 推理脚本直接使用 CUDA，需要可被 PyTorch 访问的 NVIDIA GPU，以及匹配的驱动和 CUDA 版 PyTorch。
- **深度学习框架**：配套的 PyTorch 与 torchvision。已验证版本为 PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130。
- **仿真依赖**：robosuite 1.5.1、MuJoCo 3.3.0、Mink 0.0.13。
- **数值与数据工具**：NumPy、SciPy、h5py、PyArrow、OpenCV、Pillow、huggingface_hub、IPython；已验证的 NumPy 版本为 2.4.6。
- **图形支持**：相机测试与闭环评测需要可用的 OpenGL 渲染环境。示例命令使用 GLFW，需要可用的显示连接；无显示服务器需自行准备兼容的离屏后端。
- **开发工具**：Git，以及用于运行示例命令的 Linux Shell。

更多说明见 [环境要求](env/README.md)。`env/snapshots/` 是历史依赖记录，不是经过验证的可移植锁文件。

## 运行位置

以下命令均在**克隆后的仓库根目录**执行，即包含 `README.md`、`scripts/`、`data/` 和 `third_party/` 的目录。仓库可以放在任意位置，不依赖特定用户名、绝对路径或 Conda 环境名。执行命令前，应确保当前 `python` 属于已满足上述要求的环境。

文中的 `data/...`、`outputs/...` 和 `third_party/...` 均为相对于仓库根目录的路径。

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

使用环境中已有的 `huggingface_hub` 获取数据：

```bash
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
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/eval_act_logged.py --output outputs/act_single_v2
python scripts/eval/run_act_validation.py --output outputs/act_valid_v2
# 已完成的实验可单独拼接；若批量入口已自动生成，则不必重复执行
python scripts/eval/make_eval_grid.py --input outputs/act_valid_v2 --output outputs/act_valid_v2_grid.gif
```

推理时设置 `ACT_PRETRAINED=0` 是为避免重复下载初始化权重，之后仍会加载 checkpoint 中的完整模型参数。

评测必须通过 `--output` 指定新实验目录。批量入口拒绝已有实验目录，单轮入口拒绝已有逐轮目录；包括失败运行在内均不自动覆盖。默认 `--policy act` 为纯 ACT，`--steps 100`，`--index 0` 为单轮使用的验证集索引。

启用额外保持规则：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/run_act_validation.py --policy act_hold --output outputs/act_hold_valid_v1
```

`act_hold` 连续10步满足原生高度条件后，从下一步开始固定当前末端世界系位姿，并持续发送夹爪闭合命令；使用 OSC_POSE 当前缩放生成位置/旋转反馈增量，不直接锁定关节。切换依赖仿真物体高度真值，切换后不返回 ACT，必须标为“ACT＋保持控制器”。规则效果需要闭环实测，不能视作模型学会保持。

两种策略均新增 `first_lift_hold_2s_success`：首次连续10步确认的抬升回溯到起点，累计连续40步（20Hz下约2秒）原生高度达标。确认时间计入40步，首次高度条件中断后重抓不抵消失败；达标40步之后的掉落不撤销已完成的2秒保持，但仍记录首次高度条件中断与最终成功。该指标只衡量高度保持，不证明稳定接触。100步内第61步为最后一个能完整观察40步的起点；更晚起点若未观察满，记为 `insufficient_window`，主成功率仍计失败，不删除分母。

每轮 `episode_XXXX/` 保存 `config.json`、`steps.csv`、`result.json`、`rollout.gif`，CSV 的 `action_source` 标注 ACT 或规则动作。批量实验根目录保存逐轮 `.log`、`config.json`、`process_status.json` 和实时更新的 `metrics.json`；汇总成功率以全部预定初态为分母，未完成与异常轮次不计成功，并单独报告。配置记录 checkpoint SHA256、代码提交与工作区状态、数据划分校验值、训练种子和评测协议。配置同时保存评测源码文本快照，以便追溯尚未提交的代码；训练种子仅对校验值匹配的已知基线记录为42，未知 checkpoint 记为 null。

批量推理全部结束后自动按本次实验的初态顺序，每20轮生成一份4行×5列同步动画，保存在实验目录的 `grid_001.gif`、`grid_002.gif` 等。最后一组不足20轮保留空格，失败或缺失轮次在原位标记 `MISSING/FAILED`；较短动画结束后保持末帧。标签 `H`、`S`、`F` 分别表示首次抬升保持2秒、连续10步和最终成功，旧结果缺少H时显示 `-`。拼接不改变评测数据，拒绝覆盖已有总览。

已有实验可手动拼接，无需重新推理：

```bash
python scripts/eval/make_eval_grid.py --input outputs/act_hold_valid_v1
```

可选 `--output outputs/overview.gif` 自定义文件名，多组时自动命名为 `overview_001.gif` 等。读取本次 `config.json` 的 `episode_ids`，旧实验回退到 `process_status.json` 或已有 episode 目录，不依赖当前数据划分。批量进程若尚未退出，手动拼接可能包含尚未完成的位置，应等待推理结束。

无需仿真即可检查指标边界：

```bash
python scripts/checks/test_lift_protocol.py
```

## 文件管理

代码、配置、数据划分和小型结果记录使用 Git 管理；下载数据、训练缓存、权重、日志和动画由 `.gitignore` 排除。

需要共享模型和完整评测附件时，可单独发布资源，并在文档中记录下载地址、对应代码版本及文件校验值。
