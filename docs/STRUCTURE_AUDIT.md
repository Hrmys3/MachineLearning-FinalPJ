# 目录结构与本机资源核查（2026-11-04）

核查基于提交 `d4a4bc3` 及当前工作区。检查对象包括跟踪文件、Git忽略资源、未跟踪嵌套仓库、配置与入口、包导入、文档链接和现有实验附件。初查只检查并记录；用户随后授权补迁，当前完成状态如下。后面的初查章节保留发现时的状态，不能当作迁移后的现状。

## 已完成补迁与验证

用户授权后完成49项移动，覆盖1703个文件、3264943853字节；逐文件计算迁移前后SHA256，全部一致，目标冲突检查通过。数据/输出/上游源码/归档没有字节内容改写，已跟踪统计量与ACT适配代码也未被覆盖。

- 数据与缓存已归入data/lift/，历史输出、权重、日志、GIF、release材料已归入outputs/lift/。data、outputs、third_party、archive均只剩lift/pusht；scripts只剩common/lift/pusht，configs只剩lift/pusht。
- 可选上游已移至third_party/lift/act/，HEAD仍为742c753c0d4a5d87076c8f69e5628c79a8cc5488，受现有忽略规则保护；未覆盖适配副本。
- 旧归档备份移至archive/lift/docs/；旧pycache目录保存在archive/lift/cache_snapshots/20261104/，没有删除其中内容。
- 现有WSL ml环境已执行README的可编辑注册，仅安装本仓库mlpj，不安装或升级其他依赖。数据/权重/上游和包元信息继续被Git忽略。
- 测试12项全部通过、不再跳过缓存测试；新配置的100初态预检、必需资源、跟踪实验清单引用、包根目录定位及CLI帮助均通过。
- 无相机20步物理检查通过；新脚本完成100步GPU/GLFW双相机单初态回归，输出独立保存在outputs/lift/migration_smoke_v1/episode_0000/。第44步抬升，第54步规则生效，2秒保持和final成功。与迁移前对应smoke的100步动作、方块高度、末端位置和夹爪开度比较，最大数值差为0。

历史冻结配置、源码快照、归档文件中的旧路径按原文保留。要在当前架构下重新运行，应使用新配置入口和独立输出，不直接把旧冻结配置当成当前路径配置。未恢复未找到的原理稿；未重新训练或重跑100轮正式评测。注册环境、物理和GPU验证因工作区外写入/设备访问需要经授权在沙箱外执行。未commit、push或发布。

## 迁移前总体判断

`src/mlpj/{common,lift,pusht}`、按任务拆分的scripts/tests/configs，以及两份交接的职责清楚。运行实现中没有发现仍调用旧scripts或旧资源路径的情况；包初始化不会加载MuJoCo/Isaac，根目录定位正确，项目Markdown相对链接检查通过。Push-T仍是明确的预留边界，不应当作已实现任务。

但**当前检出只完成了跟踪文件的结构切换，本机忽略资源没有跟着迁移**。因此新默认配置当前不可直接用于真实训练或评测。Git状态只有 `?? third_party/act/`，不能据此认为本机资源已整理完毕。

## 初查发现的问题（上述补迁已处理）

### 1. 数据与默认配置路径不一致

| 本机实际位置 | 新代码期望位置 | 实际情况 |
| --- | --- | --- |
| data/lift_act/episodes/ | data/lift/lift_act/episodes/ | 旧位置有70条，新位置不存在 |
| data/lift_act/random_dev100_v1.hdf5及.json | data/lift/lift_act/下同名文件 | 旧位置存在，新位置不存在 |
| data/robomimic_source/ | data/lift/robomimic_source/ | 旧位置有官方HDF5，新位置不存在 |
| data/robomimic_mt4_ph/ | data/lift/robomimic_mt4_ph/ | 旧位置有社区数据，新位置不存在 |
| data/lift_aligned_v1/ | data/lift/lift_aligned_v1/ | 历史自采39条仍在旧位置 |

新data/lift/lift_act/目前仅有跟踪的splits、来源提交及20/50条统计量。缓存与小型元信息被分在新旧两处，不能整体用旧目录覆盖新目录。

已只读核对：缓存编号恰为train50与valid的并集，70条无缺失或额外编号；每条state为[T,9]、action为[T,7]、两路图像为[T,256,256,3]。官方源文件包含200条示范；随机集为100个初态、主种子20261103。这些是结构/形状检查，不等于重跑物理、图像方向或数值全量验收。

随机集实际SHA256为 `9d672389f5308799b490be0d8dfec4d1a7fef964e6786ea6ccfe30302e0248b9`，与已记录值一致。

### 2. 权重与全部历史输出仍在旧位置

`outputs/lift/`只有.gitkeep，而本机的act_lift20_candidate、act_hold_random100_v1、speed_eval_v1、其他smoke、日志、GIF及releases仍直接位于outputs/。

默认checkpoint `outputs/lift/act_lift20_candidate/checkpoint.pt` 不存在；实际文件为 `outputs/act_lift20_candidate/checkpoint.pt`，SHA256仍是 `3732458e0391ce32c1d3298f1408d0eeeeefce83ceb51164f40e9c160b2b3241`，未发现该文件被替换或损坏。

100初态实验的100份result.json、速度对照的90份result.json及汇总仍在旧位置。新experiments/lift/中的路径字段已写成outputs/lift/...，故当前这些定位引用会落空；这是迁移未完成，不是实验结果需要重跑。

统计文件逻辑大小（不等于物理磁盘占用）：data约0.831 GiB，outputs约2.197 GiB，合计3.028 GiB。releases目录仍只是本地准备的分发材料，不代表已发布。冻结参数、源码快照、归档内部的旧路径应视为历史证据，不应为使路径“看起来统一”而批量改写其内容。

### 3. 旧上游ACT成为未跟踪嵌套Git仓库

`third_party/act/` 是完整的旧上游检出，HEAD为 `742c753c0d4a5d87076c8f69e5628c79a8cc5488`。新规则只忽略 `third_party/lift/act/`，所以旧位置在git status中显示未跟踪。

不能直接git add这个目录，避免把嵌套仓库误加为gitlink；也不能将它覆盖到已跟踪的 `third_party/lift/act_lift/`。旧上游与新适配副本至少在README、detr_vae.py和backbone.py有差异。正确的可选上游归属是 `third_party/lift/act/`，目前该目标不存在。

### 4. 现有Python环境尚未注册源码包

本次找到并复用了WSL的ml解释器。它有torch、h5py、PyYAML、robosuite及MuJoCo，但没有安装/注册mlpj；base环境连主要依赖也没有。这是当前环境准备状态，不是包目录本身的错误。

README的可编辑注册步骤是必要前置。核查使用临时PYTHONPATH=src进行只读导入与测试，没有安装或升级环境依赖。mlpj/common及任务包初始化未导入仿真；PROJECT_ROOT指向当前检出根目录。

测试发现12项：11项通过、缓存等价测试1项跳过。跳过原因是新目录data/lift/lift_act/episodes不存在，不能据此宣称本机没有缓存，或完整资源迁移已验收。

## 初查残留与待核实项

- 旧scripts/data、scripts/eval、scripts/train、scripts/tools、third_party/act_lift以及根__pycache__等残留，仅发现Python缓存，没有发现第二套旧运行入口。它们不影响当前调用，但容易混淆目录阅读；本次未删除。
- `archive/docs/LIFT_PROJECT_REPORT.md.previous` 仍是忽略的旧备份，未归入archive/lift；属于低优先级历史整理项，不应恢复成当前项目报告。
- 上次准备的未提交原理稿 `docs/PERFORMANCE_THEORY.md` 在当前仓库及新docs/lift/中都未找到；全仓（含忽略文件、排除.git）文件名检索也未找到THEORY/theory文件。仅能确认当前检出中没有该稿，不能推断丢失原因，应查原工作区或备份。
- 本次运行前跟踪文件干净，唯一普通未跟踪项为third_party/act/；另有大量资源被.gitignore隐藏。新规则对数据、权重、GIF、日志的排除符合资源边界，问题在物理位置，不应靠恢复旧忽略规则或强制添加大文件掩盖。

## 初查时的建议处理顺序

1. 在无训练/评测写入时，将现有Lift资源逐项归入新task目录；先检查目标冲突，保留原字节与校验值，保留已跟踪的小型元信息，不直接覆盖整个父目录。
2. 处理可选上游ACT的归属，再按需要清理纯缓存目录和旧归档备份；不删除唯一源码、权重或失败证据。
3. 在实际运行任务的既有Python环境中，按README注册mlpj。
4. 重新核对配置资源、实验清单引用和完整测试，再按物理→相机→模型→单初态顺序做最小回归，使用独立新输出，不能覆盖历史结果。

上述是初查时的处理建议；资源补迁、环境注册和最小回归现已完成，原理稿来源仍待查。

用户随后授权将本核查记录及相关结构/交接说明本地提交；实际大资源不纳入Git，未push或发布。
