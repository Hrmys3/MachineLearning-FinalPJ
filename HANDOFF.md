# Lift 项目交接

更新时间：2026-11-04。本文写给没有历史上下文的新会话。先阅读本文，再检查实际文件；不要把计划、历史探索或一次调试成功当成已经完成的正式实验。

## 最新状态摘要（2026-11-04）

- 最近代码提交为 `f867ead`，统一配置是 `configs/lift.yaml`；本轮仅同步三份根文档。第1—16节保留历史上下文，最新实验与待办以本摘要和第17—20节为准。第15节“未执行100轮”是当时状态，已被本次完成记录替代。
- 用户的 `outputs/act_hold_random100_v1/` 已完成100个固定随机开发初态，每轮100步；无进程异常。ACT＋规则保持，any/连续10步/最后成功分别91/100、91/100、90/100，首次抬升保持2秒89/100。
- 失败类别：9轮从未达到原生高度条件；random episode32保持阶段掉落；random episode91抬升晚，最后成功但只观察到1.6秒保持。不得将11个未过保持指标样本全称为掉落或抓空。
- 这不是纯ACT结果，也不是独立最终测试。没有同一100初态的纯ACT配对结果，不能把旧20初态的成绩变化归因于某个优化。
- 固定随机集与YAML配置实现提交为e2bc953，性能代码与测速分析提交为f867ead；本轮按用户新要求提交README、AGENTS和本交接，未新跑训练/推理，不push或发布。性能实现见第18节；推理测速10初态×3重复见第19节。此前“暂不提交三份文档”的约束已被本次明确提交授权替代；测速参数仍暂不迁入YAML。

## 1. 我们在做什么

这是机器学习课程 Final Project 题目7的具身模仿学习项目。主线是在 MuJoCo + robosuite 的 Panda Lift 环境中，用人类示范训练 ACT，再让模型根据相机图像和机器人自身状态抓起方块。Push-T 曾讨论为进阶方向，但尚未实现。

课程没有指定必须使用 Panda。选择它是为了匹配标准 Lift 和现成 RoboMimic 示范，减少机器人与动作接口的适配成本。RoboDojo 只讨论过概念与 Push-T 参考资源，实际 Lift 没有运行 RoboDojo 或 Isaac Sim。

目前已跑通数据、训练、推理、日志和验证全过程，重点问题是抓取覆盖与抓起后的稳定保持。当前不是已经完成全部对照实验的项目，也不能声称任意机器克隆后即可完美复现。

## 2. 工作位置与协作约定

- 当前 Windows 工作空间：`F:\Academy\ML`。
- 实际 Git 仓库在 WSL Ubuntu-22.04：`/home/reptac22/ml/pj-arm`。
- Windows 可访问：`\\wsl.localhost\Ubuntu-22.04\home\reptac22\ml\pj-arm`。
- 仓库远程：`https://github.com/Hrmys3/Fudan-MachineLearning-FinalPJ.git`，分支 `main`。
- 历史 Python 环境为 Conda `ml`，解释器 `/home/reptac22/miniforge3/envs/ml/bin/python`。项目本身不要求特定用户名、仓库目录或环境名。
- 命令从克隆后的仓库根目录执行。运行脚本通过自身位置寻找根目录，不要随意移动 data、outputs、third_party。
- Windows 的 Python 与 WSL Python 不共享已安装的包。操作项目时明确当前运行在哪一侧。
- WSL 没有 rg 时，使用 Python 或其他可用工具检索即可。wsl.exe 冷启动可能比较慢，也可能出现编码异常的启动警告；应继续检查命令的实际返回结果。

用户偏好：中文、清楚直接；对新手的材料用“我们”并完整讲解，对项目交接保留技术细节。倾向直接执行已授权的工作，不要反复索要确认。用户曾明确要求普通本地 commit，不要发布；后续请求 commit 不等于授权 push 或创建 Release。

## 3. 当前 Git 状态

本次写交接前已检查：工作区干净，HEAD 为 `253617d`；本地跟踪分支显示与 origin/main 一致，但没有联网查询远程最新状态，不能据此保证远端此刻完全一致。

关键提交：

| 提交 | 内容 |
| --- | --- |
| d38b73c | 原始 ACT Lift 基线 |
| 47a4b16 | 仓库分类整理、实验小型记录与资源说明 |
| 9ff49a9 | 去除个人路径要求，环境章节只列要求 |
| 253617d | 同步遗漏文档、纠正资源发布表述、删除旧报告 |

本会话执行过本地 commit，没有执行 push，也没有发布模型资源。`docs/LIFT_PROJECT_REPORT.md` 已在 253617d 中删除，不要恢复为当前教程。它是落后于现状的旧快照。

本文是结束会话时新增文件；除非随后用户要求，否则不自动提交或推送。

## 4. 仓库结构

```text
README.md                         中文运行说明，不放项目进度或末尾文档清单
HANDOFF.md                        本交接
THIRD_PARTY_NOTICES.md             上游与数据来源
scripts/
  checks/                         test_lift、test_lift_camera、test_act_model
  data/                           replay_lift_source、inspect_lift_dataset、prepare_lift_splits、cache_lift_data
  train/                          train_act_smoke、train_act_lift20
  eval/                           eval_act_debug、eval_act_logged、eval_act_one、run_act_validation、make_eval_grid
  experts/                        lift_scripted、lift_scripted_camera、lift_aligned_camera、lift_aligned_eval50、collect_lift
  tools/adapt_act.py               ACT适配生成工具
experiments/lift20_baseline/        配置、指标、20条结果JSON、进程状态、资源清单
  results/episode_XXXX.json
  artifact_manifest.json
  config.json
  metrics.json / metrics.csv
  process_status.json
docs/REPOSITORY_LAYOUT.md          结构说明
env/README.md                     环境要求，不给具体配置操作
env/snapshots/                    两份历史依赖快照
archive/generators/                add_eval_logging.py等历史生成器
third_party/act_lift/              已提交的适配ACT
third_party/act/                   本地上游副本，未跟踪
data/                             下载数据、划分、统计量、缓存
outputs/                          权重、完整推理结果与GIF
outputs/logs/                     从旧根目录迁入的日志
```

所有原根目录 Python 入口已移动，没有同名兼容包装器。批量运行器已改为调用 `scripts/eval/eval_act_one.py`。历史日志生成器不是当前推理必经步骤。

普通克隆包含代码、适配ACT、固定划分/统计量和小型实验记录；不包含原始示范、缓存、权重、完整CSV、日志和GIF。third_party/act_lift 的 README 保留上游内容，但顶部已提示：其 ALOHA、Python3.8、MuJoCo2.3.7、双臂示例和性能数字不适用于本项目。环境以根目录 README 和 env/README 为准。

## 5. 已验证环境与限制

- WSL2 Ubuntu22.04.5，历史内核6.18.33.1-microsoft-standard-WSL2。
- NVIDIA RTX4060 Laptop，约8GB显存；历史WSL内存约11GiB，swap约12GiB。
- Python3.11.17；torch2.14.1+cu130；torchvision0.29.1+cu130；NumPy2.4.6。
- robosuite1.5.1、MuJoCo3.3.0、Mink0.0.13。
- 数据工具包括 h5py、PyArrow、OpenCV、Pillow、huggingface_hub；上游辅助导入包括 IPython。
- GLFW + WSL图形环境已验证相机。EGL等后端没有同等验证记录。

最早自动安装了 MuJoCo3.17.0，Panda创建时出现关节类型断言。降至3.3.0后又与Mink1.3.0冲突，最终Mink0.0.13组合实测通过。不要只因为 pip check 通过就认定仿真兼容。

当前训练和推理入口直接用 CUDA；CPU数据处理可用不代表模型入口有CPU支持。缺少 private macros、扩展机器人包、全身IK等警告没有阻止当前Panda OSC路径，但这些可选功能本身未经验证。

env/snapshots 是历史记录，含本机路径且可能缺后续包，不是最终完整可移植锁文件。未完成空白机器端到端验收，数据revision也未固定。

## 6. 数据与主线变化

最初做脚本专家，是先确认环境和动作接口能抓取，不是课程要求一定自采。后来自采39条发生底层崩溃，现成PH数据更紧凑且可用，所以训练主线改为开源示范。

来源：

1. `chaoqi-liu/robomimic_mt4_ph`：Hugging Face社区LeRobot v3转换版，提供图像、状态、动作和来源映射。
2. `robomimic/robomimic_datasets`：官方源文件 `v1.5/lift/ph/demo_v15.hdf5`，提供环境配置、每条demo模型XML和仿真状态。

社区有4任务800条；这里只使用 task_index=0 的Lift200条，共9666帧，平均48.33帧（20Hz下约2.4秒）。没有混入其他任务或自采数据。

模型状态9维：末端位置3 + xyzw四元数4 + 夹爪位置2。动作7维：world系位置增量3 + 轴角旋转增量3 + 夹爪1；OSC_POSE，位置缩放0.05、旋转缩放0.5，夹爪-1开/+1闭。双相机 agentview 和 robot0_eye_in_hand。

注意自采状态虽然也是9维，却是关节位置7+夹爪2，与当前PH字段不同，不能按维数相同直接混用。

官方恢复必须使用对应demo的XML与初态、同步控制器/观测；`lite_physics=False` 是重要配置。不是每帧强行回填专家状态。已回放demo0/1/2，59/58/57步，均最终成功，最大qpos误差约0.00425/0.00376/0.00697；qpos混合坐标，不能统一解释成米。

社区与官方动作最大差异约2.956e-8，首轨迹双视频解码59帧且人工画面正常。缓存OpenCV解码后BGR转RGB，不额外上下翻转；实时渲染两路均需要上下翻转。episode57恢复状态差约2.94e-8，图像翻转MSE明显更小。

## 7. 划分与缓存

NumPy default_rng(42)，200条打乱：训练池160、验证20、保留20。train20是train50子集，按完整episode划分，统计量只用对应预算训练数据，标准差下限0.01。

- train20：20条944帧，当前模型实际使用。
- train50：50条2357帧，只准备了数据，未训练该预算模型。
- valid：20条954帧。
- heldout：20条，未评测，不能反复拿来调参。
- 已缓存train50+valid共70条，并通过检查。

权威文件 `data/lift_act/splits.json`、`stats_train_20.json`、`stats_train_50.json`。

验证编号：57、162、23、169、15、60、109、137、31、16、44、184、34、30、58、35、77、49、6、102。

数据路径：社区在 `data/robomimic_mt4_ph/`，官方在 `data/robomimic_source/v1.5/lift/ph/demo_v15.hdf5`，缓存在 `data/lift_act/episodes/`。每缓存包含 state[T,9]、action[T,7] 和双相机RGB[T,256,256,3]。临时写入后验证再改名，已有合格缓存跳过。

## 8. ACT与训练

上游 tonyzhaozh/act 固定提交 `742c753c0d4a5d87076c8f69e5628c79a8cc5488`，记录在 `data/lift_act/act_source_commit.txt`。

适配副本在 third_party/act_lift，已随Git提交，正常运行不必重新生成。适配器遇到已有目录会停止；不要盲目删掉已有修改。状态投影14改9、动作投影/输出14改7，不是全局把14改7；torchvision改用weights接口。

当前配置：共享ResNet18双相机，128x128输入，ImageNet归一化；hidden256、FF1024、编码/解码各4层、8头、动作块20、约21.87M参数。训练CVAE用动作后验，推理z=0。损失为正确屏蔽padding的标准化L1 + 10*KL。

训练：20条、seed42、batch8、2000次更新、AdamW，主学习率1e-4、骨干1e-5、weight_decay1e-4、梯度裁剪1、workers0。脚本中的num_epochs占位不代表实际训练轮数，实际循环为2000更新。

- 随机模型前后向/推理通过，峰值约438.2MiB。
- 100更新真实数据smoke（无预训练）验证L1约0.612768。
- 2000更新候选（ImageNet骨干初始化）验证L1约0.473614749，峰值约481.3MiB。

训练保存最后更新，不是验证选最佳；checkpoint不含优化器状态，不是完整resume。没有多训练种子或50条模型。标准化L1不是成功率或物理单位。

当前模型 `outputs/act_lift20_candidate/checkpoint.pt`，87693107字节，SHA256：
`3732458e0391ce32c1d3298f1408d0eeeeefce83ceb51164f40e9c160b2b3241`。

## 9. 闭环评测与结果

预测20步，每次执行前5步再观测重规划，20Hz；没有时间集成，没有人工夹爪锁定，也没有专家接管。ACT输入只有双图像和自身状态，物体真值仅用于恢复与诊断/判定。

早期单初态200步：episode57第38步成功，夹持较久后松开，最终失败。后来用户选择100步（5秒）作为探索基线；这在观察结果后调整，不是事先注册的最终评测协议。不能继续缩短时长来隐藏掉落。

20个验证示范初态，每轮100步，独立子进程：

| 指标 | 结果 |
| --- | --- |
| 任意一步成功 | 19/20，95% |
| 原生成功连续至少10步 | 19/20，95% |
| 最后一步成功 | 3/20，15% |
| 成功样本平均首次成功 | 41.947368步，约2.10秒 |
| 进程异常 | 0/20 |

最终成功编号57、169、109；102从未成功。用户看过GIF，纠正过“所有失败都是末期张开”：也有起初抓空或夹持不良。最后一步成功不证明全过程持续持有，可能掉落后重抓；连续10步也不证明有稳定接触，原生主要是高度条件。

短示范缺少长保持监督是合理假设，但尚未证明是唯一原因；需要同时检查状态分布外运行、夹爪预测、动作块执行与终止协议。

结果位置：`outputs/act_logged/` 单轮steps.csv/result.json/rollout.gif；`outputs/act_valid100/` 批量证据；`make_eval_grid.py`生成4行5列同步总览。单轮目录可能被后续100步覆盖，不保证仍保存早期200步证据。

评测通过缓存首帧自动比较raw/flipped方向，依赖对应源示范初态和缓存；这是当前工具限制，不是任意新初态独立泛化测试。验证集已用于调试，不能改称独立测试集。

## 10. 历史专家与排错经验

规则专家使用物体真值、阶段位置控制：接近100、下降100、闭合30、抬升120、保持30，共380步19秒。位置增益6限幅0.6；对齐版从方块四重对称方向选最近向下姿态，旋转误差Log(R_target R_current^T)，增益0.8限幅0.15，下降前误差门槛10度。

对齐专家种子1000–1049的50次，any/final/自定义稳定成功均50/50；首次原生成功平均234步。这不是ACT的成功率，专家稳定标准还是相对reset中心升高0.10m连续10步，与ACT原生标准不同。

自采39条每条380步后出现 `corrupted double-linked list`、SIGABRT。结构与数值检查通过，但根因未定位，不能直接断言OOM、显卡或HDF5问题。stdout缓冲使日志最后seed不等于实际最后保存seed。独立进程评测只是隔离，不是修复崩溃。

静态GIF曾由保存方法和修改错文件/文件未更新导致；先确认实际执行文件及输出更新时间，再调控制器。动作与图像语义、四元数顺序、状态字段、归一化和翻转应逐项核对。物理、相机、模型计算分开检查能大幅缩小排错范围。

## 11. 常用运行顺序

以下在仓库根目录、已激活合适环境时执行，不是安装教程：

```bash
python scripts/checks/test_lift.py
MUJOCO_GL=glfw python scripts/checks/test_lift_camera.py
ACT_PRETRAINED=0 python scripts/checks/test_act_model.py
python scripts/data/replay_lift_source.py
python scripts/data/inspect_lift_dataset.py
python scripts/data/prepare_lift_splits.py
python scripts/data/cache_lift_data.py
ACT_PRETRAINED=0 python scripts/train/train_act_smoke.py
ACT_PRETRAINED=1 python scripts/train/train_act_lift20.py
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/eval_act_logged.py --output outputs/act_single_v2
python scripts/eval/run_act_validation.py --output outputs/act_valid_v2
python scripts/eval/make_eval_grid.py --input outputs/act_valid_v2 --output outputs/act_valid_v2_grid.gif
```

数据下载见根README。ACT_PRETRAINED=1只控制模型初始化；推理设0后仍加载完整checkpoint，不代表随机推理。批量运行器会给子进程设置评测环境变量，细节看代码。

不要照此整串盲目重跑：训练/单轮推理可能覆盖现有输出；批量评测遇到已有结果会停止。先检查目录、配置和结果存在性。新实验独立命名，不覆盖基线。

## 12. 文档与对外共享

Notion项目页：`https://app.notion.com/p/40c844e76a6b80bf8283ec3e9032e017`。

子页：

- 流程篇 `40c844e76a6b80d98207eb3b719d3bae`：面向新人、九章、概念首次出现使用框内介绍。
- 算法篇 `40c844e76a6b802a9ffddce81c4499c4`：数学、ACT与代码阅读。
- 进度交接 `40c844e76a6b808ca31dd8f52935897b`：原题为进度文档-10.29，保留历史数值。

已按新结构同步以上三页脚本路径、目录树、命令、日志/快照位置与资源边界，并更新父页。父页有连接器无法完整读取的GitHub嵌入块，未改动它；使用定点替换保留未知块和子页。在线文档与本地文件没有自动同步。

Windows曾有 `F:\Academy\ML\LIFT_PROJECT_REPORT.md` 教学稿，不能假定它也已同步最终结构。当前根README仅介绍用法，用户明确不希望加入项目进度或末尾“文档”部分。环境说明仅列环境要求，不提供具体配置步骤。

本地曾准备 `outputs/releases/lift20-baseline-v0.1/` 的权重、评测tar、GIF、元信息和校验值，均忽略、未发布。存在此目录不等于有Release。是否还存在应实际检查。模型、GIF不适合直接塞普通Git历史；需要分享时另定资源分发方式并记录hash和代码版本，但必须先取得发布授权。

尚未选定原创代码的统一许可证；保留ACT MIT与DETR Apache来源许可，不应暗示整个项目可任意按某许可证使用。

## 13. 下一步建议（均尚未完成）

用户当前要求只是结束会话交接，不要自动启动训练或发布。恢复项目时先确认最新目标并检查实际工作区。

1. 固定当前基线、评测协议和输出目录；按逐步日志分类抓空、夹持失败、晚期开夹与重抓，不能只看any-success。
2. 用同一checkpoint/初态比较执行长度H=1和H=5（必要再H=10），预测块K保持20，先检验反馈间隔影响。
3. 若检验短示范保持问题，增加真实保持/恢复监督或有合理语义的数据增广，再重训练；简单复制终帧不等于有效动态示范。
4. 在其他条件一致下训练50条预算，与20条比较；逐步加入周期验证与最佳模型选择。保持统计量只来自对应训练子集。
5. 增加训练种子，冻结方案后评测独立heldout，完整报告any、连续成功、final及必要的稳定持有指标。
6. 在新机器检验安装与端到端流程，补可移植依赖约束、数据revision和资源哈希。
7. 经用户授权后共享权重/完整证据；之后再讨论Push-T。

不要未经实验就认定保持段是唯一原因；不要悄悄强制闭合夹爪后把效果归功于纯ACT；若加入规则，明确命名为额外控制策略并单独比较。
## 14. 保持规则与新评测接口（2026-11-03）

现有协作与交接文档已本地提交为 db1c0ad，未 push。新增代码工作尚未提交。

评测入口现要求 `--output` 指定独立实验目录；单轮 logged 与 one 共享实现。`--policy act` 保持纯 ACT；`--policy act_hold` 使用原生高度连续10步触发固定末端位姿、持续闭合夹爪的规则，下一步生效且不返回 ACT。新增首次确认抬升后连续40步原生高度达标指标，确认10步计入保持，首次中断后重抓不能覆盖失败。窗口不足单列，仍在主成功率分母内。详细命令、数据位置与协议见 README 推理与评测章节。基线小型记录及 outputs/act_valid100 均未覆盖。

规则与新指标是新增实现，不能将旧基线结果当作新规则效果。

已完成验证：4项指标单元测试、评测脚本语法检查、git diff --check；现有 WSL Python 环境下无相机40步保持控制正常（空载末端位置漂移约2.91e-10m）。CUDA＋GLFW单初态100步集成检查保存在 `outputs/act_hold_smoke_v1/episode_0057/`：第38步首次成功，第47步确认，第48步规则生效，原生高度连续达标63步，2秒保持及最终成功均为真。该轮只验证集成，不能推断20轮成功率；未执行批量实验或重训练。沙箱内无法访问CUDA，Numba缓存也受写权限限制，物理和相机集成检查经授权在沙箱外执行。源码快照记录功能在此次单轮之后添加，故该轮旧config不含新增快照；没有回填或改写这轮运行配置。

GIF总览更新：`run_act_validation.py` 在全部子进程结束后自动调用 `make_eval_grid.make_grids`，按本次实验顺序每20轮生成一份4×5动画 `grid_001.gif` 等；不足20轮保留空格，失败/缺失保留原位。独立命令 `python scripts/eval/make_eval_grid.py --input outputs/act_hold_valid_v1` 可用于已完成实验。21轮合成数据验证了分组、缺失位置、动画及防覆盖，不重跑推理。
已为现有 `outputs/act_hold_valid_v1/` 生成 `grid_001.gif`，实际检查为960×888、101帧；未改写逐轮评测结果，未提交或推送本次代码。

用户随后要求提交上述实现：提交范围为保持规则、首次抬升保持指标、评测输出保护与源码快照、GIF分组自动拼接及相关文档/测试；提交前再次通过4项指标测试、5个评测模块语法检查和 git diff --check。用户补充要求将 `Final_PJ.pdf` 一并纳入本地提交，提交信息不单独提及PDF；无 push 或资源发布。

## 15. 固定随机开发集与统一配置（2026-11-03）

上一轮实现已提交为 `0f718f5`（包含用户要求的PDF），本次新增实现尚未提交或push。统一入口 `configs/lift.py`，训练/单轮/批量入口接受 `--config`；默认评测改为随机开发集，旧20个示范验证初态用 `--initial-states valid`。配置含推理步数、H、规则触发高度与连续步数、保持反馈增益、指标确认步数/时长、随机集数量/种子/路径，以及模型架构、图像尺寸、训练预算/种子/学习率/损失/更新次数等。模型结构与图像尺寸在推理时以checkpoint为准，不能通过改配置原地改造已有权重。20Hz固定。保持触发阈值独立于原生4cm评价标准，指标通用字段为first_lift_hold_success，2秒协议另保留旧字段。批量将有效参数冻结为parameters.py供子进程读取。训练拒绝复用输出。

已实际建立 `data/lift_act/random_dev100_v1.hdf5`，100个初态，主种子20261103，SHA256 `9d672389f5308799b490be0d8dfec4d1a7fef964e6786ea6ccfe30302e0248b9`。小清单在experiments/random_dev100/manifest.json；数据与本地清单仍按资源边界忽略。源环境配置来自官方PH文件；主种子确定固定几何模型，随后原生reset采样物体xy/yaw与机器人初始化噪声。全体状态有限、唯一、初始不成功，每条重建XML后完整状态与方块位置恢复校验通过；没有根据策略成功筛样本。该集用于开发调参，不是独立最终测试，也未覆盖全部几何随机化。生成器拒绝覆盖，改数量/种子须用新路径。

验证：5项指标测试（含自定义确认步数/1秒协议）、配置与文件语法检查；随机集首个初态100步CUDA+GLFW闭环在outputs/random_dev100_smoke_v1/，第44步抬升，第54步保持，2秒及最终成功。旧示范初态回归在outputs/config_eval_valid_smoke_v1/，H=1、规则高度0.08米/连续3步、指标确认3步/保持1秒、共60步；实际第39步抬升，第46步规则生效，指标及最终成功，旧缓存双相机翻转核对仍通过。2次真实数据训练smoke在outputs/config_train_smoke_v1/，预测长度10、输入96×96、batch2、无ImageNet初始化；前反向、验证与checkpoint保存通过，验证标准化L1约1.1380，仅是流程检查。临时配置在/tmp，不作为团队运行要求。100条合成结果的批量集成检查通过冻结参数、汇总和5份4×5动画，但不是实际100轮推理结果。未执行新的100轮正式评测、未覆盖旧checkpoint/旧20轮证据。仿真/模型检查因Numba缓存和CUDA访问限制经授权在沙箱外运行。

## 16. YAML迁移与提交（2026-11-03）

用户要求在不影响进行中的100轮推理的前提下改为YAML并提交。迁移检查时 `outputs/act_hold_random100_v1/process_status.json` 已有54轮记录；该任务冻结文件为 `parameters.py`，父进程显式传给各子进程。读取器完整保留可信 `.py` 快照分支，不修改本次冻结配置、数据集或运行参数，后续任务默认使用 `configs/lift.yaml` 并冻结为 `parameters.yaml`。YAML通过PyYAML safe_load读取；现有环境PyYAML6.0.3，不安装或升级依赖。旧生成清单中的历史.py路径与源码文本保留为原始证据，不伪造迁移后的生成记录。

验证：新YAML有效参数逐项等于迁移前配置；正在运行的parameters.py与原runpy读取结果完全一致；2项配置测试、5项指标测试、语法与diff检查通过；使用合成子进程结果验证新批量入口的YAML冻结、读取、汇总及GIF生成。未重启、停止或重跑用户的100轮推理，也未改动模型推理与仿真控制算法。本地提交包括上一节全部未提交实现和本次迁移；不push、不发布。

## 17. 100轮随机开发评测完成与简要分析（2026-11-04）

### 运行证据与协议

- 目录：`outputs/act_hold_random100_v1/`；100条result.json、100份steps.csv各100步、100条正常进程记录，grid_001.gif至grid_005.gif共5份4×5总览均存在。大资源继续忽略，未发布。
- 策略：ACT＋规则保持。预测20、执行5、20Hz、总100步；方块中心高于桌面0.04m连续10步后，下一步固定末端世界位姿、持续闭合，切换后不回ACT。原生成功仍是高度条件，不检查接触。
- 首次确认抬升回溯起点，连续40步即2秒达标，确认10步计入其中；首次高度中断后重抓不抵消失败。窗口不足保留在100轮分母中。
- 初态：`data/lift_act/random_dev100_v1.hdf5`，种子20261103、固定几何模型，SHA256 `9d672389f5308799b490be0d8dfec4d1a7fef964e6786ea6ccfe30302e0248b9`；100轮逐轮配置的checkpoint和eval参数一致。
- checkpoint仍为20条PH、seed42、2000更新的原模型，SHA256 `3732458e0391ce32c1d3298f1408d0eeeeefce83ceb51164f40e9c160b2b3241`，本次未重训练。
- 运行在YAML迁移前启动，冻结文件实际为parameters.py；根config记录启动HEAD为0f718f5且有未提交改动，迁移提交为e2bc953。不得将根记录改写成由干净e2bc953启动；依当次冻结参数和逐轮源码快照追溯。未改写任何实验证据。

### 核对后的结果

| 指标 | 数值 |
| --- | --- |
| 计划/尝试/完成 | 100/100/100 |
| 进程异常 | 0 |
| 任意成功 | 91/100，91% |
| 原生高度连续10步成功 | 91/100，91% |
| 首次有效抬升保持2秒 | 89/100，89% |
| 最后一步成功 | 90/100，90% |
| 成功样本首次抬升步数 | 平均42.7582、中位42、范围37—69，平均约2.14秒 |

汇总由100份CSV的native_success序列重新计算any、final和首次保持，与metrics.json一致；核对100份result及进程状态，未重跑实验。连续10步统计亦从100份CSV重算并与逐轮汇总核对。未做性能分段计时，不根据文件时间戳编造吞吐。

### 失败与限制

- 未曾达标的9个随机初态编号：1、11、15、19、26、34、46、84、95；保持规则均未触发。可归为抬升前失败，具体抓空、对齐、闭合时机或夹持不良需结合GIF/接触信息进一步区分，不能仅凭最终高度定根因。
- random episode32：第50步首次成功，第59步确认，第60步进入规则；第73步高度失效，此前连续23步、约1.15秒；最终失败。第60—100步持续闭合命令+1，末端保持在约0.947m，而方块降至桌面。支持夹持不稳/滑落方向的排查，不能归因于后期开夹；尚未通过接触信息证明具体机制。
- random episode91：第69步首次成功、第79步规则生效，至100步连续32步、1.6秒，final为真；hold_status为insufficient_window，不能推断若继续运行一定能成功或一定会掉落，也不事后缩短保持要求。
- 在91轮曾抬升的样本中，89轮满足2秒保持；这是条件描述，主成功率分母仍是100。保持策略表现较好，但未消除抓取覆盖和不稳夹持问题。
- 旧20初态act_hold为19/20，纯ACT为any19/20、final3/20；与本次初态集不同，不能直接用百分比差异评估算法变化。需要同一100初态、同checkpoint的纯ACT对照。

### 下一步（尚未执行）

1. 保留本次数据及失败证据；同一100初态比较纯ACT与规则保持，再比较H=1/2/5，固定其他条件。
2. 分类9个抬升前失败，重点看对齐和闭合时机；检查32的夹持接触/相对运动。不要通过提前规则接管、加大闭合或缩短时长掩盖问题。
3. 训练50条预算模型、多种子与独立最终测试仍未完成；当前100初态只用于开发，不因数量增大就变成独立测试。
4. 优化速度前先实现分段计时与硬件监控；数据缓存、DataLoader并行、混合精度、模型复用、环境并行、延迟视频编码均仍是建议，不是已有功能。新增配置必须接入实现后才能宣称支持。

本次文档验证：核对运行命令、配置字段、相对链接及git diff --check；不因文档更新重跑训练或推理。未commit、push或发布。

## 18. 性能工具与第一阶段加速实现（2026-11-04）

用户授权实现速度优化及测试方法。本次在前一轮未提交的README/AGENTS/HANDOFF修改上继续工作，没有覆盖100轮评测、原模型或历史基线。全部新增速度选项在configs/lift.yaml，加载器对历史配置补默认值，默认FP32、训练worker0/无缓存、评测每轮独立进程、完整视频；计时默认开启，CUDA详细事件/硬件采样/Profiler默认关闭。

### 实现范围

- scripts/performance.py：CPU墙钟阶段统计、可选CUDA事件末尾统一读取、Linux内存/逐核CPU和nvidia-smi只读采样；训练与批量评测输出performance.json，可选hardware.csv。PyTorch显存峰值不包含OpenGL，评测单轮报告的峰值范围为worker生命周期。
- 训练：none/raw_ram/processed_ram缓存、可调DataLoader worker与预取/保留worker、pin memory和nonblocking传输、FP16/BF16 autocast、FP16 GradScaler、FP32 KL/loss归约；记录跳过与实际optimizer更新。缓存当前禁止与多worker组合，以免复制大缓存。可选torch.compile接入但未实际验证兼容性/收益。
- 评测：PolicyRuntime可复用模型，每轮仍完整创建/恢复/关闭环境，episodes_per_worker控制定期进程重启；num_workers控制同时运行的子进程数。默认1/1。Python异常后记录并继续，底层崩溃保留组内先前结果，未完成轮次计入分母并退出非零。worker_failures与未完成轮次单列。
- inference_mode与推理精度可选；视频可关闭或按frame_stride抽帧，抽帧保持原时间轴，不减少观测、物理步骤或20Hz。CSV/result始终保存。环境复用、降低相机更新频率、异步编码尚未实现。
- scripts/tools/benchmark_speed.py：相同任务的自动A/B，独立配置与目录、顺序交替、固定共同CPU线程（默认1），训练更新数/评测前N初态可设置；--no-video两侧一起关闭保存，--diagnostics两侧一起开监控和CUDA事件。输出runs.json与comparison.json，失败时不生成有效速度比；评测比较配对指标、动作与高度差，训练列验证L1及稳定吞吐。
- scripts/tools/monitor_hardware.py：独立只读采样。训练Profiler只采短窗口。使用方法、限制和公平对照命令见docs/PERFORMANCE.md。

### 实际检查与局限

- 测试共12项：5项速度正确性（两种缓存与原样本逐元素相等、配置冲突、视频时长、计时和worker崩溃保留分母）、2项配置、5项保持指标，均通过；语法/diff/相对链接检查。计时统计测试与worker崩溃模拟不等于真实闭环成功率。
- 真实GPU训练smoke：outputs/speed_train_smoke_v1/，4组各30次更新尝试，batch8、seed42、无ImageNet初始化、共同torch_threads=1、预热5次，开启详细CUDA事件和硬件采样。外层总时间baseline25.354s/cache22.919s/workers18.447s/amp22.707s；训练循环8.398/5.260/6.824/7.570s，稳定吞吐51.887/65.656/61.440/45.360 samples/s。processed缓存建立约5.090s，因此看循环和总时间的结论不同。FP32验证L1约0.759914/0.760143/0.762301；AMP约0.804842，跳过6次更新（实际24次），不能称等价训练提速。
- 上述仅各1次短检查，后续改进了AMP更新计数实现为optimizer post-hook以避免每步读取scale引入同步；不能把这些记录包装为最终版本稳定基准。另在outputs/speed_profile_smoke_v1/做8次FP16尝试、batch2，短窗口Profiler、CUDA事件、checkpoint与performance输出通过，实际更新计数已核对。测试在独立目录进行，临时配置仅作本次检查，不作为用户运行要求。
- 真实GPU+GLFW评测smoke：outputs/speed_eval_smoke_v1/，同一原checkpoint，固定随机初态0/1，每轮100步、ACT+保持规则，所有方案共同关闭视频，详细事件/硬件采样开，CPU线程1。baseline/reuse/inference/parallel均完成；外层时间84.439/64.829/65.184/45.673s，单次相对速度约1.30/1.30/1.85。三方案对基线逐步动作与方块高度最大差均为0，四项配对成功指标无变化。只测2个初态、单次且存在缓存/温度/顺序影响，不能承诺完整100轮同样收益。计时元数据与环境关闭计时的小改动在检查期间完成，记录源码快照保留各次实际版本。
- hardware.csv实际记录到GPU利用率/显存/温度/功耗，未报采样错误；WSL其他机器可能为N/A。Profiler trace已导出。混合精度推理、BF16、torch.compile及完整100初态速度/质量回归未实际跑，保留opt-in，不声称验证完成。
- 当前建议先比较baseline/cache/workers和baseline/reuse；至少3次重复看总时间中位数与稳定阶段吞吐，最终完整100初态和完整训练确认正确性。性能工具不调整原生4cm、保持2秒、总100步、PH20Hz等实验语义。

代码尚未提交，未push或发布；用户本轮没有要求commit。

## 19. 推理测速定量分析与选择性提交（2026-11-04）

用户要求分析新测速结果，同步README/HANDOFF，并提交除README.md、HANDOFF.md、AGENTS.md之外的更新。三份根文档保持本地修改、不纳入本次提交；其他性能代码、配置、测试、性能说明及小型分析记录纳入本地提交，不push、不发布。用户此前要求暂不把测速专用参数迁入YAML，故方案/重复次数/updates/episodes/output仍由测速CLI指定，未落实被中断的迁移提议。

### 数据与定量结论

实际目录outputs/speed_eval_v1/；baseline、reuse、parallel各3次，每次随机开发初态0—9、每轮100步、20Hz、ACT＋保持规则、FP32、inference_mode关闭、CPU线程1。完整保存单轮和总览GIF，实际没有使用--no-video；CUDA事件和硬件采样关闭。共90轮、10个不同初态，不是90个独立样本。各方案使用相同checkpoint、初态集及源码快照；启动HEAD为e2bc953且优化代码尚未提交，不能回填新提交号伪造来源。

| 方案 | 第1/2/3次总时间（秒） | 中位数 | 相对速度 | 耗时减少 | 端到端episode/分钟 |
| --- | --- | --- | --- | --- | --- |
| baseline，1worker/1轮 | 343.92 / 337.19 / 327.33 | 337.19 | 1.000× | — | 1.78 |
| reuse，1worker/5轮 | 270.29 / 264.31 / 253.35 | 264.31 | 1.276× | 21.61% | 2.27 |
| parallel，2worker/1轮 | 192.46 / 178.41 / 218.19 | 192.46 | 1.752× | 42.92% | 3.12 |

所有90轮正常，无worker异常；每次四项成功指标均9/10，失败为random episode1。重新读取90份100步CSV，按重复编号配对核对7维动作和方块高度，差值全部为0，与comparison.json一致。没有重新运行训练或推理。

模型复用每10轮节省约72.88秒，但模型加载累计中位数只从9.72秒降到2.28秒；串行批量时间减去各轮内部总时间的剩余开销约54.19→12.56秒，包含启动、模块导入和进程生命周期，不能精确归因为某一种操作。环境建立累计95.47→81.17秒，env.step（含渲染）约120.95→120.23秒；实际瓶颈不限于模型前向。model_forward累计墙钟16.57→8.91秒不是纯GPU计时，未测CUDA事件，不能声称GPU算子本身快一倍。

双worker各轮rollout累计中位数146.31秒，基线为137.58秒，说明吞吐提升不等于单轮更快；并行阶段重叠，不能将累计阶段耗时与整批墙钟相减。其总时间范围178.41—218.19秒，波动更大；无硬件采样不能确定原因。总览编码中位数约13—14秒，仍计入本次总时间。

### 建议和记录

- 在本机这10个初态上，parallel收益更大，reuse也稳定减少总耗时；正式配置先做100初态完整回归。parallel＋reuse组合没有在本次9组运行中测试，不能相乘推导组合收益。
- 统一YAML仍保留保守默认值，README补充正式入口启用FP32缓存/模型复用的方法和测速参数优先级；没有擅自更换当前模型、优化开关或实验协议。
- 小型分析记录已保存experiments/speed_eval10_v1/REPORT.md及summary.json，包含9次实际耗时、阶段统计、源码/模型/初态校验值、正确性与限制；完整资源仍在忽略的outputs下。此记录不等于发布原始附件。
- 验证范围：代码测试、语法、文档相对链接、diff检查，以及实验证据重算。此次不额外重跑速度基准，避免覆盖或混合已有结果。

本次本地提交已完成：`f867ead`（Add Lift performance optimizations and reproducible speed benchmarks），含16个代码/配置/测试/性能文档及小型分析文件。已检查提交列表不含README.md、HANDOFF.md、AGENTS.md；三份根文档仍保留本地修改。12项测试及语法、相对链接、git diff --check通过，无push或发布。历史测速启动来源仍为带未提交改动的e2bc953，未改写其结果文件。

## 20. 三份根文档同步提交（2026-11-04）

用户明确要求更新并提交README、HANDOFF、AGENTS，替代上一轮对这三份文件的提交排除。仅提交三份文档，不改代码、参数或实验证据。README同步正式优化用法、性能配置分组、可选GIF、worker日志和汇总刷新时机，保留中文运行说明定位，不加入项目进度表。AGENTS补充小型测速记录边界、测速CLI参数约定及worker失败统计规则；本交接修正最新提交状态，保留历史实验来源和未完成事项。

本次只做文档检查：核对入口、配置键、相对链接与git diff --check；不重跑训练、推理或测速。提交前最近代码版本为f867ead，此文档提交后的HEAD应以git log为准，避免在提交内容中自引尚未产生的提交号。仅本地commit，不push或发布。
