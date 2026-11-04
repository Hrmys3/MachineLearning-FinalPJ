# 机器人模仿学习：Lift 与 Push-T

Lift 使用 MuJoCo、robosuite、Panda、RoboMimic PH 示范和适配后的官方 ACT。Push-T 预留独立任务目录，拟接入 RoboDojo 与 XPolicyLab；尚无可运行的 Push-T 训练或评测入口。以下运行命令用于 Lift。

## 目录结构

```text
src/mlpj/        # Python包：common共享工具、lift实现、pusht预留
scripts/         # 按lift/pusht分开的运行入口；common放共享入口
tests/           # 按common/lift/pusht分开的自动化测试
configs/         # lift/default.yaml；pusht配置目录预留
experiments/     # lift/pusht的小型实验记录
docs/            # 公共结构说明及lift/pusht任务文档
env/             # lift/pusht环境要求与历史快照
archive/         # lift/pusht历史辅助材料
data/            # lift/pusht原始数据、划分与缓存
outputs/         # lift/pusht权重、日志、评测结果和动画
third_party/     # lift/pusht第三方源码及适配副本
```

详细说明见 [仓库结构](docs/REPOSITORY_LAYOUT.md)。[Lift 交接](HANDOFF_LIFT.md) 与 [Push-T 交接](HANDOFF_PUSHT.md) 分别维护各自任务的进度、验证和待办；影响两项任务的更改同时记录在两份交接中。

## 环境要求

- **操作系统**：Linux；Windows 使用 WSL2 Ubuntu。已验证的发行版为 Ubuntu 22.04，其他系统组合需要单独验证。
- **Python**：3.11，使用独立的 Python 环境；环境名称可自行选择。
- **GPU**：训练和 ACT 推理脚本直接使用 CUDA，需要可被 PyTorch 访问的 NVIDIA GPU，以及匹配的驱动和 CUDA 版 PyTorch。
- **深度学习框架**：配套的 PyTorch 与 torchvision。已验证版本为 PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130。
- **仿真依赖**：robosuite 1.5.1、MuJoCo 3.3.0、Mink 0.0.13。
- **数值与数据工具**：NumPy、SciPy、h5py、PyArrow、PyYAML、OpenCV、Pillow、huggingface_hub、IPython；已验证的 NumPy 版本为 2.4.6。
- **图形支持**：相机测试与闭环评测需要可用的 OpenGL 渲染环境。示例命令使用 GLFW，需要可用的显示连接；无显示服务器需自行准备兼容的离屏后端。
- **开发工具**：Git，以及用于运行示例命令的 Linux Shell。

更多说明见 [环境要求](env/README.md)。`env/lift/snapshots/` 是历史依赖记录，不是经过验证的可移植锁文件。

## 运行位置

以下命令均在**克隆后的仓库根目录**执行，即包含 `README.md`、`scripts/`、`data/` 和 `third_party/` 的目录。仓库可以放在任意位置，不依赖特定用户名、绝对路径或 Conda 环境名。执行命令前，应确保当前 `python` 属于已满足上述要求的环境。

文中的 `data/...`、`outputs/...` 和 `third_party/...` 均为相对于仓库根目录的路径。

运行入口通过 `mlpj` 包导入实现。先在实际运行任务的 Python 环境中注册源码包，不自动安装或升级仿真、训练依赖：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

不同 Python 环境需分别注册。当前包使用仓库检出与可编辑安装，数据、配置和第三方源码仍从仓库读取。

## 基础检查

先检查无相机仿真，再检查相机与模型计算：

```bash
python scripts/lift/checks/test_lift.py
MUJOCO_GL=glfw python scripts/lift/checks/test_lift_camera.py
ACT_PRETRAINED=0 python scripts/lift/checks/test_act_model.py
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
    local_dir="data/lift/robomimic_mt4_ph",
    max_workers=2,
)
hf_hub_download(
    repo_id="robomimic/robomimic_datasets",
    repo_type="dataset",
    filename="v1.5/lift/ph/demo_v15.hdf5",
    local_dir="data/lift/robomimic_source",
)
PY
```

### 检查与预处理

```bash
python scripts/lift/data/replay_lift_source.py
python scripts/lift/data/inspect_lift_dataset.py
python scripts/lift/data/prepare_lift_splits.py
python scripts/lift/data/cache_lift_data.py
```

数据划分与归一化参数位于 `data/lift/lift_act/`，逐轨迹缓存位于 `data/lift/lift_act/episodes/`。训练与推理应使用相同的状态字段、相机顺序和归一化参数。

## ACT 源码

官方 ACT 源码版本固定为：

```text
742c753c0d4a5d87076c8f69e5628c79a8cc5488
```

适配代码位于 `third_party/lift/act_lift/`。如需从原始代码重新生成适配副本：

```bash
git clone https://github.com/tonyzhaozh/act.git third_party/lift/act
git -C third_party/lift/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
python scripts/lift/tools/adapt_act.py
```

原始仓库存在时不必重复克隆。适配工具不会覆盖已有 `third_party/lift/act_lift/`，重新生成前应检查并保留自己的修改。使用第三方代码时保留原许可证与来源说明。

## 模型训练

先进行小规模流程检查，再运行训练入口：

```bash
ACT_PRETRAINED=0 python scripts/lift/train/train_act_smoke.py
ACT_PRETRAINED=1 python scripts/lift/train/train_act_lift20.py
```

`ACT_PRETRAINED=1` 在创建模型时加载 ImageNet 视觉骨干权重；`0` 表示不加载该初始化。训练参数和输出目录统一在 `configs/lift/default.yaml`，也可通过 `--config configs/lift/your_experiment.yaml` 指定配置副本。

训练入口拒绝复用已有输出目录；修改配置中的 `train.output` 开始独立实验。已有基线 checkpoint 不会被覆盖。

## 推理与评测

推理入口默认读取：

```text
outputs/lift/act_lift20_candidate/checkpoint.pt
```

建立随机初态集需要官方源文件；使用旧验证示范初态时另需社区数据元信息及对应轨迹缓存。官方源文件：

```text
data/lift/robomimic_source/v1.5/lift/ph/demo_v15.hdf5
```

权重和大体积数据不随普通 Git 仓库提供。运行前需准备对应资源，并按上述路径放置。

统一修改 [configs/lift/default.yaml](configs/lift/default.yaml) 中的参数：

| 配置分组 | 可调整内容 |
| --- | --- |
| `random_set` | 固定随机初态数量 `count`、主种子 `seed`、数据集路径 `path` |
| `eval` | 总步数 `steps`、执行长度 `execution_horizon`、策略 `policy`、初始化来源 `initial_states`、checkpoint、保持触发高度/连续步数、位姿反馈增益、评价确认步数/保持秒数 |
| `model` | 预测长度 `chunk_size`、Transformer维度/层数/注意力头、骨干、dropout、输入图像尺寸；用于下一次训练 |
| `train` | 示范预算20/50、种子、更新次数、batch size、学习率、损失权重、预训练、缓存/数据加载与独立输出目录 |
| `performance` | 阶段计时、可选CUDA事件、硬件采样、训练Profiler与CPU线程数 |
| `video` | 单轮GIF保存、保存帧间隔和自动总览；不改变模型观测或控制频率 |

模型结构和输入图像尺寸在推理时读取 checkpoint，修改配置的 `model` 不会改造已有模型。训练预算自动对应相应划分和统计量。PH数据的20Hz控制频率固定；时间集成目前未实现。命令行的 `--policy`、`--steps`、`--checkpoint`、`--initial-states` 可覆盖配置。批量运行开始时将实际配置冻结为实验目录的 `parameters.yaml`，后续编辑原配置不会影响已启动的子进程。读取器仍兼容旧实验的可信 `.py` 配置快照；分析历史实验时以其冻结配置和源码快照为准，不用当前YAML替代当时参数。

默认使用100个固定随机开发初态。初次准备：

```bash
python scripts/lift/data/prepare_random_lift_states.py --config configs/lift/default.yaml
```

数据集保存完整模型XML、逐轮仿真状态和采样种子，并校验每个初态可准确恢复；不执行策略或筛掉困难样本。主种子确定一个固定模型，随后使用原生Lift reset采样方块xy位置、yaw和机器人初始化噪声；这是固定几何条件下的开发评测，不等同于所有模型尺寸/场景的泛化或独立最终测试。数据和同名JSON清单位于 `random_set.path`；清单的小型记录见 `experiments/lift/random_dev100/`。数据文件不随普通Git提供，重复生成会拒绝覆盖；调整数量/种子时必须使用新路径建立新版本。

随机初态评测不依赖对应示范缓存，双相机采用已验证的上下翻转、RGB和checkpoint输入尺寸约定。恢复时核对完整仿真状态，运行前核对数据集SHA256、数量和种子。旧20个验证示范初态仍可通过 `--initial-states valid` 使用，并保留缓存首帧图像方向核对。

单初态推理、批量开发评测与动画总览：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/lift/eval/eval_act_logged.py --output outputs/lift/act_random_single_v2
python scripts/lift/eval/run_act_validation.py --output outputs/lift/act_random100_v2
# 已完成的实验可单独拼接；若批量入口已自动生成，则不必重复执行
python scripts/lift/eval/make_eval_grid.py --input outputs/lift/act_random100_v2 --output outputs/lift/act_random100_v2_grid.gif
```

推理时设置 `ACT_PRETRAINED=0` 是为避免重复下载初始化权重，之后仍会加载 checkpoint 中的完整模型参数。

评测必须通过 `--output` 指定新实验目录。批量入口拒绝已有实验目录，单轮入口拒绝已有逐轮目录；包括失败运行在内均不自动覆盖。默认配置为纯 ACT、100步、100个随机开发初态；`--index 0` 为单轮在所选初态集中的索引。旧验证集使用 `--initial-states valid`。

启用额外保持规则：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/lift/eval/run_act_validation.py --policy act_hold --output outputs/lift/act_hold_random100_v2
```

`act_hold` 默认连续10步满足配置高度条件后，从下一步开始固定当前末端世界系位姿，并持续发送夹爪闭合命令；使用 OSC_POSE 当前缩放生成位置/旋转反馈增量，不直接锁定关节。`eval.hold_trigger_height` 为方块中心相对桌面的高度差，默认0.04米；只改变规则切换时机，原生成功高度仍固定为0.04米。切换依赖仿真物体高度真值，切换后不返回 ACT，必须标为“ACT＋保持控制器”。持续闭合不保证夹持牢固；抬升后掉落也需检查夹持姿态和滑动，不能一律归因为主动松手。规则结果不能视作纯ACT学会保持。

两种策略均记录通用 `first_lift_hold_success`；默认2秒时同时保留 `first_lift_hold_2s_success`：首次连续10步确认的抬升回溯到起点，累计连续40步（20Hz下约2秒）原生高度达标。确认时间计入40步，首次高度条件中断后重抓不抵消失败；达标40步之后的掉落不撤销已完成的2秒保持，但仍记录首次高度条件中断与最终成功。评价确认步数和保持秒数由 `eval.metric_confirmation_steps`、`eval.metric_hold_seconds` 控制；修改后应显式记录协议，指标字段不再冒称2秒。该指标只衡量高度保持，不证明稳定接触。100步内第61步为最后一个能完整观察40步的起点；更晚起点若未观察满，记为 `insufficient_window`，主成功率仍计失败，不删除分母。

每轮 `episode_XXXX/` 保存 `config.json`、`steps.csv`、`result.json` 和 `performance.json`，启用视频保存时另有 `rollout.gif`；CSV 的 `action_source` 标注 ACT 或规则动作。批量实验根目录保存日志、`config.json`、`process_status.json`、`metrics.json` 和性能汇总；汇总成功率以全部预定初态为分母，未完成与异常轮次不计成功，并单独报告。配置记录 checkpoint SHA256、代码提交与工作区状态、数据划分校验值、训练种子和评测协议。配置同时保存评测源码文本快照，以便追溯尚未提交的代码；训练种子读取checkpoint记录；旧checkpoint仅对校验值匹配的已知基线记录为42，其余缺少记录时为null。

查看某次实验汇总与进程状态（将目录替换为实际输出目录）：

```bash
python -m json.tool outputs/lift/act_hold_random100_v2/metrics.json
python -m json.tool outputs/lift/act_hold_random100_v2/process_status.json
```

`metrics.json` 的 `n`、`attempted`、`completed` 分别为计划、已尝试、正常完成轮数；`process_failures` 为未正常完成的轮数，`worker_failures` 为异常worker数，`counts`、`rates` 为各指标成功数和成功率。复用模型时，同组先完成的有效结果不会因后续轮次崩溃而被丢弃；日志路径记录在 `process_status.json` 的 `worker_log` 中。汇总在worker结束后更新，多轮复用时不一定每轮结束就立即刷新。运行中成功率仍以全部计划轮数为分母，等所有轮次结束后再作为最终结果使用。

`hold_status_counts` 区分成功、首次确认后高度条件中断、观察窗口不足和未确认抬升。`insufficient_window` 表示尚未观察满要求时长，不等于已发生掉落；即使 `success_final` 为真也不能补算保持成功。`no_confirmed_lift` 不自动等于抓空，应继续查看 `success_any`、高度、夹爪及视频。不同策略应使用同一组初态作配对比较；不同初态集上的百分比不能直接解释为优化收益。

批量推理全部结束后自动按本次实验的初态顺序，每20轮生成一份4行×5列同步动画，保存在实验目录的 `grid_001.gif`、`grid_002.gif` 等。最后一组不足20轮保留空格，失败或缺失轮次在原位标记 `MISSING/FAILED`；较短动画结束后保持末帧。标签 `H`、`S`、`F` 分别表示首次抬升保持配置时长、连续10步和最终成功，旧结果缺少H时显示 `-`。拼接不改变评测数据，拒绝覆盖已有总览。

已有实验可手动拼接，无需重新推理：

```bash
python scripts/lift/eval/make_eval_grid.py --input outputs/lift/act_hold_valid_v1
```

可选 `--output outputs/lift/overview.gif` 自定义文件名，多组时自动命名为 `overview_001.gif` 等。读取本次 `config.json` 的 `episode_ids`，旧实验回退到 `process_status.json` 或已有 episode 目录，不依赖当前数据划分。批量进程若尚未退出，手动拼接可能包含尚未完成的位置，应等待推理结束。

无需仿真即可检查指标边界：

```bash
python tests/lift/test_lift_protocol.py
python tests/lift/test_project_config.py
```

## 性能测量与加速

速度选项也在 `configs/lift/default.yaml`：训练支持内存缓存、DataLoader预取、固定页内存、异步传输、混合精度；评测支持有限轮次模型复用、多worker、inference_mode和视频保存开关。默认仍为FP32、逐轮进程隔离及完整视频，计时单独输出为 `performance.json`，不修改成功指标。编译为实验选项，效果需要另测。

正式训练/评测直接运行原入口，不经过测速脚本。可先在现有YAML分组中修改以下字段（其余训练与评测参数保留）：

```yaml
train:
  output: outputs/lift/act_lift20_cached_v1  # 新目录；使用缓存前确认内存充足
  cache_mode: processed_ram
  num_workers: 0                      # 内存缓存不与多worker同时开启
  pin_memory: true
  non_blocking_transfer: true
  precision: fp32

eval:
  precision: fp32
  episodes_per_worker: 5              # 复用模型；每轮仍重建环境
  num_workers: 1                      # 可单独比较2；复用＋并行组合需另测
  episode_limit: null                 # 正式随机开发评测使用全部100初态

video:
  save_rollout: true
  auto_make_grids: true
```

训练运行 `ACT_PRETRAINED=1 python scripts/lift/train/train_act_lift20.py`；评测运行 `MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/lift/eval/run_act_validation.py --policy act_hold --output outputs/lift/act_hold_optimized_v1`。评测checkpoint路径仍须指向希望评价的实际模型；纯ACT使用 `--policy act`。以上配置示例没有自动修改默认值，混合精度和编译需单独核对效果。

自动对照使用相同任务、独立输出与交替顺序：

```bash
ACT_PRETRAINED=0 python scripts/lift/tools/benchmark_speed.py --kind train --variants baseline,cache,workers --updates 200 --repeats 3 --output outputs/lift/speed_train_v1
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/lift/tools/benchmark_speed.py --kind eval --policy act_hold --variants baseline,reuse --episodes 10 --repeats 3 --no-video --output outputs/lift/speed_eval_check_v1
```

测速专用的方案、重复次数、更新数、初态数量和输出目录仍在命令行指定，不写入统一YAML；测速脚本会为各方案生成实际配置，不能用当前YAML推断测速时的最终参数。

查看各测速目录的 `comparison.json`；速度比是“基线总耗时中位数÷该方案总耗时中位数”，大于1表示更快。1.276倍对应约21.6%的耗时减少，不能把两者混用。同时核对完成情况、训练实际更新数与验证误差、推理配对指标及动作差异；总耗时包含启动和输出，不等于纯模型前向时间。`--no-video` 对两侧共同关闭保存，仍保留双相机观测和全部仿真步数。对并行方案同时报告批量吞吐和单轮延迟，重叠阶段的累计时间不能直接相加。重复同一批初态用于测速度稳定性，不会增加独立初态数量。可先小规模测速，再用100初态验证正确性。选项、硬件监控、计时含义及限制见 [性能对照说明](docs/lift/PERFORMANCE.md)。

## 文件管理

代码、配置、数据划分和小型结果记录使用 Git 管理；下载数据、训练缓存、权重、日志和动画由 `.gitignore` 排除。

需要共享模型和完整评测附件时，可单独发布资源，并在文档中记录下载地址、对应代码版本及文件校验值。
