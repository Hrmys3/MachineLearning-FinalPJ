# Push-T 项目交接

更新于 2026-11-04。本文面向首次接手本项目的会话。课程项目已完成 Panda Lift 的初步实验，Push-T 目前处于路线选择和环境验收阶段。Lift 的最新结果、配置与运行入口见 [HANDOFF_LIFT.md](HANDOFF_LIFT.md) 和 [README.md](README.md)。仓库工作约定见 [AGENTS.md](AGENTS.md)。

本次共享目录重排和交接命名调整按用户要求一并纳入本地提交，父提交为 `1ef5ed2`，具体提交号以 `git log` 为准；下文未提交描述此前阶段状态。本次不push或发布，也不增加Push-T实验结果。

## 维护范围

本文维护 Push-T 任务的进度、环境验收、路线选择和待办；Lift 任务进度归 [HANDOFF_LIFT.md](HANDOFF_LIFT.md)。Push-T 专属更改更新本文；共享代码、目录、环境或协作规则等同时影响两项任务的更改，必须同时更新两份交接，分别记录任务影响、验证和限制。

## 目录约定

项目包名为 `mlpj`。Push-T实现归 `src/mlpj/pusht/`，入口归 `scripts/pusht/`，配置、数据、结果、实验说明和第三方依赖分别归对应顶层目录的 `pusht/`。当前仅建立目录边界，未下载RoboDojo、安装Isaac或新增训练/闭环实现。Lift已归各目录的 `lift/`，配置为 `configs/lift/default.yaml`。共享工具归 `src/mlpj/common/`。

## 共享结构调整与验证（2026-11-04）

代码按 `src/mlpj/common/`、`src/mlpj/lift/`、`src/mlpj/pusht/` 组织，各任务的scripts、configs、data、outputs、experiments、third_party、docs、env及archive分开。Push-T目前仅有包和目录边界；后续仿真端和策略端分别注册mlpj，依赖环境保持独立，不能直接复用Lift的20Hz、状态动作语义和高度指标。

先前目录调整在/tmp独立验证环境完成mlpj可编辑安装，测试发现12项，11项通过、1项因缺少Lift轨迹缓存跳过；入口、根目录定位、源码快照、资源边界和链接检查通过。这些是公共结构与Lift现有代码的验证，不代表Isaac、RoboDojo、Push-T数据或闭环已验收。当前仓库没有完整示范、权重和历史outputs，本次没有迁移这些大资源。

本轮统一交接文件名为HANDOFF_LIFT.md与HANDOFF_PUSHT.md，README及AGENTS明确“任务分别维护、共同改动同时更新”。只进行文档链接、引用、命令路径和diff检查，未运行训练或仿真；未commit、push或发布。

## 任务和选择

目标是让机械臂用原有夹爪直接推动桌面上的 T 形物体，使其位置和朝向与目标对齐。用户要求不加额外推杆，并希望探索张开平行夹爪、用两根手指推动。迄今未找到核验过的、同时提供张开双指 Push-T 示范与配套环境的公开资源。

| 路线 | 可复用资源 | 首个验收目标 |
| --- | --- | --- |
| X5 + RoboDojo + Isaac Sim | 官方双臂任务、配套 HDF5、XPolicyLab ACT 适配器 | 一张获分配的 GPU 上无窗口相机出图 |
| Panda + robosuite + MuJoCo | 现有 Panda、桌面、相机、OSC 和数据训练框架 | 固定物体朝向和有限初态下可靠推动 |

近期工作集中在 X5 路线的服务器验收，两条方案均保留。Lift 和 Push-T 可以分别使用不同机器人、环境和模型，共享实验方法。成功率受任务、数据和评价规则影响，不能据两个任务的分数比较机械臂能力。

经典二维 gym-pusht 缺少完整机械臂，ManiSkill 官方 PushT 使用 PandaStick，真实 Panda 夹笔示范带额外工具。这些可作为参考。 [gym-pusht](https://github.com/huggingface/gym-pusht) · [ManiSkill 任务源码](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_t.py)

## 已核验的资源和服务器结果

- RoboDojo 官方 `push_T/arx_x5` 文件列表有 100 个 HDF5，任务目录约 28.1 GB。已下载并抽查第 0 条示范的头部和右腕预览，看到右臂用闭合夹爪直接推物体，未见额外工具。此结论仅覆盖抽查样本。完整 HDF5 字段、动作回放和训练均待验收。 [官方轨迹](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo/push_T/arx_x5/data) · [预览](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/main/data/RoboDojo/push_T/arx_x5/preview_video)
- 远程服务器的终端输出由用户提供。Ubuntu 24.04.1，约 376 GiB 总内存、347 GiB 可用，当前层的 cgroup `memory.max=max`，家目录所在文件系统约 449 GiB 可用。个人配额待确认。
- `nvidia-smi` 报告 8 张 GPU，每张总显存 32760 MiB。Vulkan 将它们识别为 RTX 4080 SUPER，使用 NVIDIA 专有驱动 570.86.10，并另外列出 CPU 软件设备 llvmpipe。无 DISPLAY 与 XDG_RUNTIME_DIR 的提示未阻止设备枚举。显存数字来自服务器报告，不作为零售型号规格引用。
- 当时 GPU 0 和 6 分别占用 27697 和 24473 MiB，2、3、4、5、7 各占 1 MiB。仅是历史快照。进程列表为空和 GPU 利用率为 0% 都不能当作显存无人占用的证据。
- `act` 环境为 torch 2.4.1+cu121、torchvision 0.19.1。在物理编号 2 上用 CUDA 张量求和得到 `1024.0`，基础 GPU 运算通过。RoboTwin 环境列出 SAPIEN 3.0.0b1、torch 2.4.1、torchvision 0.19.1。
- 两个环境的包列表未发现 Isaac Sim 或 Isaac Lab。`command -v isaacsim` 未返回路径。家目录四层内只找到 `RoboTwin/XPolicyLab/scripts/RoboDojo`，属于集成脚本目录。尚未找到可复用的 Isaac 安装。
- **当前验收点是 CUDA 和 Vulkan 枚举已通过，Isaac 无窗口启动、实际相机渲染、RoboDojo 场景、数据转换、ACT Push-T 训练与闭环均未完成。** 已提供安装和相机测试方案，尚无用户执行结果。GPU 使用权限仍待协调。

## X5 路线的接手顺序

### 1. 预约 GPU 并验收渲染

先申请管理员或调度系统分配一张卡，首轮约 30～60 分钟，预留约 16 GiB 空闲显存。这是预约建议和预算，实际用量待测。仿真与训练分开测，记录启动峰值、持续占用和用时。开始前确认实时占用和允许使用的 GPU，结束后确认进程退出、显存释放。宿主机驱动与容器图形权限问题交由管理员处理。

RoboDojo 的安装脚本使用 Isaac Sim 5.1.0 和 Python 3.11。NVIDIA 文档列出 32 GB 内存、16 GB 显存，Linux 测试驱动为 580.65.06。当前 570.86.10 与测试版本不同，是否可用以启动和出图实测为准。`nvidia-smi` 的 CUDA 12.8 是驱动报告值，`act` 的 cu121 是 PyTorch 构建版本。 [RoboDojo 安装脚本](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/scripts/install.sh) · [Isaac Sim 5.1 要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

在获准使用的 GPU 上，建立独立 Python 3.11 测试环境，按官方文档安装 `isaacsim[all,extscache]==5.1.0`。首次运行由用户自行处理 NVIDIA 许可提示。先用 `SimulationApp` 无窗口启动，创建地面、红色方块和一台 256×256 相机，调用 `world.step(render=True)` 和 `camera.get_rgba()` 检查连续画面。保留日志与 PNG，目视确认画面包含方块，核对退出码。CUDA 逻辑编号与 Vulkan 渲染编号分别核实，按 UUID 或 PCI 地址确认渲染落在分配的卡上。首轮使用单卡、单场景。若失败，从第一条驱动、Vulkan 或 RTX 错误入手。 [安装文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) · [相机示例](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html)

### 2. 验收官方场景

固定 RoboDojo、XPolicyLab、Isaac 版本和资产、数据 revision。先跑单个 `push_T / arx_x5`，确认双臂、物体、目标和相机加载，少量受控动作正常执行并可退出。RoboDojo 提供仿真与评价，XPolicyLab 提供策略服务与 ACT 适配。仿真和策略使用独立依赖环境，必要时分机运行。官方曾修复观测一帧错位和 RGB 字节顺序，版本选择需检查对应更新记录。 [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) · [XPolicyLab ACT](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/README.md)

原生任务代码含 `step_lim=600`、平面位置阈值 0.007 m、姿态检查参数 `dis_threshold=7`、机器人回位和抬升相关逻辑。姿态参数单位、奖励组合、终止语义须继续追踪实现。 [任务代码](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/push_T.py)

### 3. 核验并转换少量示范

先取少数 HDF5，核对左右臂顺序、观测与动作时序、控制目标、三路相机、颜色与采样频率。优先试官方 `joint` 路线，双臂每侧 6 关节加 1 夹爪量，共 14 维，具体语义以字段和打包代码为准。采集配置为 25 Hz。图像用官方 `decode_image_bit`。位姿字段使用 wxyz 四元数顺序，Lift 的状态使用 xyzw。 [环境配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/env_cfg/arx_x5.yml) · [接口定义](https://github.com/XPolicyLab/XPolicyLab#-standard-data-formats)

检查是否有足够初态信息恢复示范，再回放动作。若不能恢复，记录限制并做局部动作接口验收。核对动作标签是记录的目标还是经转换得到的目标，以及训练读取器是否再做时间移位。

按完整 episode 固定划分。若 100 条均通过，可用训练池 60、验证 20、保留 20，train20 嵌套于 train50。归一化仅使用对应预算训练集。官方转换入口见 ACT README，当前转换脚本按连续编号读取并生成配置，需显式审计非连续划分、统计范围和 padding。 [转换实现](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/detr/process_data.py)

### 4. 训练和闭环

先用一批真实数据做前向、反向、保存重载和推理。测量 GPU 峰值与每次更新耗时，再预约正式训练。官方入口的配置包括三路 640×480 图像、动作块 50、隐藏维 512、batch 16，部署默认启用时间集成。Lift 的 481 MiB 历史峰值仅对应当时轻量模型的 PyTorch 分配。若调整分辨率或动作执行方式，训练与部署同步记录。 [训练脚本](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/train.sh) · [部署配置](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/deploy.yml)

顺序为离线接口检查、单初态闭环、固定验证初态批评测。区分动作块长度、每次实际执行步数和每轮上限。记录物体对齐、回位、整体成功、位置与角度误差、首次成功步数、异常和视频。保留未接触、推偏、过冲、卡住、退开时碰乱和未回位的失败证据。先比较 20/50 条或执行间隔，冻结方案后再用保留集。

## Panda + robosuite 路线

复用已有 Panda、桌面、双相机和 OSC，新增 T 形碰撞物体、非阻挡目标标记、初态随机化和成功判定。先固定朝向与夹爪状态，只改变有限范围内的位置。核对碰撞几何、指尖高度、摩擦、桌面干涉和偏心推产生的转动，再扩大初始角度分布。

示范可用人工遥操作或读取物体真值的规则专家生成。复杂初态需要选择接触点、调整角度和重新接触。记录动作前 RGB 与自身状态、实际执行动作、后继观测、初态和控制器配置。物体位姿、接触与成功记录作为诊断字段。沿用 9 维自身状态和 7 维 OSC 动作时仍须检查字段语义。预先定义位置、角度、留在桌面和退开后的对齐要求。先采几条、做回放和短训练，再扩大数据。历史 Lift 自采遇到 SIGABRT，采集应逐轮保存、验证后改名并支持续采。 [robosuite 示范文档](https://robosuite.ai/docs/algorithms/demonstrations.html)

Panda 单臂 7 旋转关节，X5 单臂 6 个，仿真夹爪每指行程分别 40 mm 与 44 mm。TCP、碰撞面和控制响应不同。当前 Panda 使用 OSC，RoboDojo X5 使用关节伺服且关闭机器人重力。20 Hz 与 25 Hz 下执行 5 步分别是 250 ms 与 200 ms。若借鉴 X5 轨迹，需要坐标转换、重采样、在 Panda 重新执行并录制配套数据。规格不足以推算成功率。 [Panda 夹爪模型](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/models/assets/grippers/panda_gripper.xml) · [X5 资产](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/blob/main/Assets/Robots/x5/X5A.urdf) · [X5 仿真配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/env/robot_manager/robot_config/x5.py)

闭合夹爪适合首个基线。张开双指时先固定开口宽度，重新采集匹配示范，检查实际单指或双指接触、卡住和意外夹持。使用相同初态、数据预算和评价协议比较。两根手指随同一末端运动，双指优势是待验证的假设。

## 当前接手点

先协调服务器 GPU 时段，执行 Isaac 5.1 单相机无窗口测试。通过后加载官方 X5 Push-T 场景，再读少量原始 HDF5。若仿真机器暂不可用，可先核验数据字段和 XPolicyLab 版本，Panda 路线仍可独立推进。使用独立输出目录，保留 Lift 基线与失败证据。大数据、模型和完整视频按仓库已有资源边界管理。提交与推送分别授权。
