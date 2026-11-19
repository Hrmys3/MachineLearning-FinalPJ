# Push-T 项目交接

更新于 2026-11-19。本文面向首次接手本项目的会话。课程项目已完成 Panda Lift 的初步实验，Push-T已完成环境与假策略单步验收、两条示范基础审计及CPU数据组件检查；真实ACT单批次计算及20次更新短训练/checkpoint重载已通过，正式训练和完整闭环尚未完成。Lift 的最新结果、配置与运行入口见 [HANDOFF_LIFT.md](HANDOFF_LIFT.md) 和 [README.md](README.md)。仓库工作约定见 [AGENTS.md](AGENTS.md)。

历史目录重排已提交；本次用户明确授权更新文档、创建Push-T分支并提交相关未提交改动。下方阶段记录中的“当前”“未提交”和待办属于各自记录时间，接手以本节及最新进度为准；不改写既有实验来源提交。


用户随后授权将上述短训练结果、离线部署对照、真实模型单动作入口及配套文档提交到pusht分支。本次提交前检查Python语法、实验JSON、文档链接和diff；沿用已完成GPU验收，不重复训练或仿真。下方各阶段“未commit”指当时状态，当前提交号以git log为准；未push或发布。

## 最新结果：真实ACT→WebSocket→EvalEnv单动作通过（2026-11-19）

用户明确授权完成真实模型单动作联通并汇报文件。新增`serve_act_policy.py`和`check_act_eval_link.py`，复用既有act策略环境和pusht_sim仿真环境，不安装依赖、不改上游/Lift实现。策略端torch2.4.1+cu121/torchvision0.19.1+cu121/websockets16.1.1实际加载torch2.7训练产物并推理通过；两端固定FP32、关闭时间集成。未比较跨PyTorch版本逐值一致性。

实际策略GPU6、仿真GPU3，各次占用前均实时检查并由脚本再次守卫。证据`outputs/pusht/checks/act_eval_20261119_v1/`：服务/仿真退出码0，总审计及两端ok:true；reset、观测、动作请求各1。三路640×480 RGB图像SHA256、状态和返回14维动作两端一致。真实EvalEnv.take_action执行一次，控制队列长度10，sim_step实计10，dt=.004即0.04秒。左右臂最大实际关节变化0.018414/0.199725rad；右臂目标最大变化1.888925rad，尚未跟踪到该目标。左右夹爪预测0.999483/0.004692，本次未触发裁剪，原生米制映射与控制目标一致。T位移0，不是推动成功。

保存前后三路6张PNG、真实状态、原始预测、原生插值及夹爪映射、checkpoint/布局/源码校验、有效配置、两端日志和监督收尾源码。主代理目视前后头部图像正常；日志无[Error]/Traceback，保留上游材质/显示/弃用警告。服务端首请求推理约0.488秒、往返0.491秒，不证明25Hz持续实时性；服务端PyTorch峰值641.95MiB。两进程正常关闭后GPU3/6均2MiB/0%。

小型记录`experiments/pusht/act_eval_link_20261119/summary.json`，复现与范围见[真实模型单动作说明](docs/pusht/ACT_EVAL_LINK.md)。环境版本文档同步，所有图像、日志、权重继续忽略；未commit、push或发布。之前未提交的短训练/离线部署结果和子模块补丁均保留。

下一步明确官方布局与示范初态差异、冻结多动作执行协议后，再授权开展短闭环。时间集成/多次取动作、越界夹爪在线裁剪、持续25Hz推理、正式train20/50及成功率均未验收。

## 最新结果：真实ACT部署适配器离线对照通过，须固定FP32精度（2026-11-19）

用户明确授权完成训练端到部署端数值一致性检查。新增`scripts/pusht/checks/check_act_deployment.py`，使用刚完成的last模型和同份统计量，比较项目Dataset/ACTPolicy与固定XPolicyLab真实Model的update_obs/get_action。时间集成关闭，episode0的12/179/358帧每帧reset；未启动Isaac或WebSocket。

- 首轮v1默认精度超出rtol1e-5/atol1e-6严格阈值，退出码1。v2保留阈值并补充诊断，三帧输入/权重逐值相同，但CHW内存布局不同；同布局重算差0。默认设置下完整动作块最大差2.2519e-4、反归一化首动作最大差7.7486e-5，v2仍失败，未计为通过。
- 仅在检查进程显式关闭cudnn/matmul TF32后，用原阈值在新目录v3重跑通过：3/3、ok:true、退出码0。模型图像与标准化状态输入差0，完整50步预测块最大差7.1525574e-7，首个14维动作最大差1.1920929e-7，左右臂/夹爪字段及统计量检查通过。结论限定显式FP32；后续模型服务精度须同步冻结，不能称默认精度逐值一致。
- t=179右夹爪原始输出-0.0009475872，记录为模型输出越界，未裁剪或执行；其余检查帧夹爪范围正常。这不属于接口映射错误，也不证明策略推动有效。
- 证据`outputs/pusht/checks/act_deployment_20261119_v1/`、v2、v3保留源码、protocol、预测NPZ、runtime日志及退出状态；最终GPU3前后1MiB/0%，PyTorch峰值962.80MiB，脚本计时5.46秒。小型记录`experiments/pusht/act_deployment_20261119/summary.json`，复现与局限见[部署对照说明](docs/pusht/ACT_DEPLOYMENT_CHECK.md)。本轮未安装依赖、改上游或Lift代码，未commit、push或发布。

下一步是冻结实际模型服务精度及动作配置，随后在用户授权下接通真实模型WebSocket→EvalEnv单动作，再扩展短闭环；尚未验证时间集成/多次取动作、初态恢复、物理动作或正式成功率。

## 最新结果：20次更新诊断训练与checkpoint重载通过（2026-11-19）

用户明确授权执行短训练。主代理复用已有环境（pip check通过）和提交`6538def`的脚本，未修改源码，运行前确认物理GPU3完整UUID、1MiB/0%，脚本内部再次守卫。命令为`python scripts/pusht/train/train_act_smoke.py --gpu-uuid <GPU_UUID> --output outputs/pusht/act_smoke_20261119_v1 --updates 20 --val-every 10`。实际GPU UUID记录在冻结参数/结果与设备快照，其他GPU上的用户进程未操作。

- 训练episode0、统计量仅episode0，episode1是流程诊断验证；两条均属于正式训练池。固定训练/验证各4个位置，验证种子1000，每10次更新检查，batch1、chunk50、三路640×480、hidden512、KL10、有效动作元素平均L1、梯度裁剪1、无AMP。
- `outputs/pusht/act_smoke_20261119_v1/result.json` ok:true，进程退出码0；metrics核对训练1—20连续无缺失，固定评估在0/10/20完成，所有记录损失/梯度范数有限。固定训练总损失80.14254→17.15081、L1 0.85653→0.79829；诊断验证总损失100.66808→66.99278，但L1 9.41976→9.46229略升，总损失下降主要来自KL。最佳总损失选在update20，不代表最佳动作L1或闭环成功。
- 保存policy_best.ckpt、policy_last.ckpt、dataset_stats.pkl，实际重新实例化模型加载last并推理，(1,50,14)输出与保存前最大差0；统计量重载逐值一致。last SHA256=`a00e0b85b1c054f5ac7bdc254263e754503d0a04bfea84cfde32eddb1b0cb726`。PyTorch峰值分配1782.00MiB，脚本报告10.85秒；这是张量分配峰值，不是总GPU显存。结束后GPU3回到1MiB/0%。
- 冻结parameters.yaml、源码、数据/源码校验值、runtime.log、metrics.jsonl、process-status.json、GPU前后快照均留在独立输出目录。小型记录`experiments/pusht/act_smoke_20261119/summary.json`；原始数据、完整日志和模型不进入普通Git。本轮新增记录与文档未commit、push或发布。

下一步应先复核真实模型部署的反归一化、动作解释、初态恢复及短闭环接口，再准备正式训练预算/验证协议；短诊断不能证明模型收敛或泛化，不直接扩大训练步数来替代验收。valid20、heldout20和正式闭环成功率均未运行。下方“短训练未执行”保留当时阶段边界。

## 本次提交与验证（2026-11-19）

用户要求提交当前更改。本次在pusht分支纳入单批次检查脚本、FixedPushTACTDataset、valid_act_losses、短训练诊断脚本及相应README/数据文档/交接。主代理重新完成掩码L1数值与梯度检查、真实固定帧重复一致性及末帧掩码、四个相关Python文件语法检查，均通过且CUDA未初始化；并读取单批次result.json确认历史一次更新和推理结果。短训练仍未执行，不把提交写成新训练结果。提交号以git log为准；无push/发布。RoboDojo本地已应用补丁保留，未另造子模块提交。下节的“未commit”指当时创建脚本阶段。

## 最新增量：短训练脚本已创建并通过语法检查（2026-11-19）

用户授权执行上一条给出的命令。检查发现`scripts/pusht/train/train_act_smoke.py`尚未保存，因此主代理先按对话代码创建，再使用已有仿真环境Python执行`python -m py_compile scripts/pusht/train/train_act_smoke.py`，退出码0、无输出。版本字符串显式转换为普通str后写YAML，避免TorchVersion对象序列化问题。默认20次更新的诊断脚本已存在，但本次只检查语法，没有训练、GPU占用、checkpoint或新的损失结果。下一步执行前仍须实时核验空闲GPU，指定新的输出目录。原有用户修改和未跟踪文件保持；未commit、push或发布。

此前用户已反馈真实ACT单批次检查通过，证据`outputs/pusht/checks/act_batch_ukd37nn7/`：一次更新、非零有限梯度、参数变化、推理(1,50,14)，PyTorch峰值约1513.5MiB，退出后GPU4回到1MiB/0%；固定样本数据集与有效元素平均L1也已按指导检查通过。下方11月7日的“尚未创建/执行单批次检查”属于历史状态。短训练与checkpoint重载仍未执行。

## 最新接手状态：示范审计、CPU数据接口与pusht分支（2026-11-07）

`.gitattributes`仅对`env/pusht/*.patch`关闭空白检查：统一diff的空白上下文行含必需空格，不按普通源码尾空格处理；其余源码与文档仍做正常diff检查。

- 用户逐步执行了官方episode0/1的下载和检查，两条大小与官方SHA256一致，25Hz、359/512帧、joint state/action均(T,14)且有限；全部可比较位置满足action[t]==state[t+1]。第0条全部关节/位姿数值有限、三路首中末帧解码及预览图目视正常；第1条joint和首末帧解码通过，未目视确认。末动作变化小不证明是填充，图像/状态物理时间对齐和动作回放仍待验。
- 用户创建`configs/pusht/splits_v1.json`，seed20261107，train20嵌套train50嵌套train60，valid20/heldout20互斥，覆盖100条。已检查的0/1放入train20。当前仅有两条原始HDF5，尚无正式20/50条训练统计量。
- 原XPolicyLab ACT读取器is_sim=None导致额外取action[t-1]；统计量混入验证数据并重复末帧，转换器要求连续编号。已按指导由用户创建`src/mlpj/pusht/act_data.py`与`act_dataset.py`，实现显式编号、同索引动作块、仅指定训练编号真实帧统计、官方RGB解码与Dataset；上游ACT训练代码未改、原train.sh尚未接入这些组件。
- 用户反馈合成与真实数据检查通过：动作块t=12有效50项，t=358有效1项/填充49项；标准差ddof=1、下限1e-2；组合样本反归一化恢复原值。CPU批量张量images(2,3,3,480,640)、qpos(2,14)、actions(2,50,14)、is_pad(2,50)通过，三路顺序head/right_wrist/left_wrist。两条诊断统计量不是正式train20统计量。
- 当前环境能导入ACTPolicy、train_bc、forward_pass，torch2.7.0+cu126、torchvision0.22.0+cu126。刚提出的`check_act_batch.py`尚未创建且未执行，没有模型前向/反向/参数更新或推理结果。下一步由用户按指导准备单批次检查，运行前重新选实时空闲GPU；之后接通训练入口、冻结验证采样和预算统计，再准备全量训练预算和闭环。
- 详细数据SHA256、字段、局限及可执行CPU示例见[数据与ACT适配](docs/pusht/DATA_AND_ACT.md)。本次文档/提交工作仅做CPU与文档核对，不代替上述待执行GPU检查。
- 用户授权创建`pusht`分支，起点`0fdbaa9f2b5283f8a4239ea1aba74a92741c5c74`；本次提交包含此前Push-T环境、设备补丁、检查入口、小型记录、数据组件与划分及配套文档，提交号以git log为准。保留main的Lift基线，无push/发布。RoboDojo/IsaacLab本地改动以env/pusht两份补丁保存；父gitlink仍固定上游版本，当前应用补丁的子模块显示修改属于预期。完整HDF5、资产、相机图、日志和权重均不纳入Git。
- 提交前主代理重新校验两条HDF5 SHA256、固定划分、三个真实CPU样本(0/12、0/358、1/125)与反归一化，解析全部Push-T Python语法；8项既有Push-T测试通过，CUDA未初始化。两份设备/兼容补丁reverse检查通过，文档路径与diff检查通过。
- Lift与Push-T的实现、配置、第三方适配和依赖保持独立，当前无Push-T到Lift实现的导入。共享README/AGENTS/结构说明更新同步记录于HANDOFF_LIFT；分支不隔离工作目录资源、环境或GPU，后续公共代码变动仍须两份交接同步。
- 后续继续遵循“指导用户—用户反馈—下一步”；本次明确授权的是文档更新和Git分支/提交，不代表自动开始训练或仿真。

## 历史阶段记录

以下保留环境与仿真验收过程；其中尚未下载/尚未联通等状态，以顶部最新接手状态为准。

## 当前状态：XPolicyLab本机WebSocket与14维打包已通过（2026-11-07）

后续真实仿真单步验证已通过，详情见下节；本节保留先前独立传输检查的过程和边界。

用户要求继续，指定最多两个只读子代理使用`gpt-5.6-luna`、low。已按指定模型启动两个：分别核对EvalEnv/WebSocket/归一化调用链，以及ACT部署与数据字段/相机顺序；均未修改文件、安装、运行GPU/仿真或提交。主代理独立核对并实施下述检查。

- 当前`act`环境Python3.10.22、websockets16.1.1，可导入固定XPolicyLab的PolicyServer与WsModelClient，pip check正常；`pusht_sim`内websockets12.0缺`websockets.asyncio`，不能承载服务端。未升级仿真环境。
- 新增`scripts/pusht/checks/check_policy_transport.py`：本机真实WebSocket服务端＋客户端，确定性假策略完成握手、reset、三路uint8 RGB像素一致传输和一次get_action，返回双臂6+1+6+1动作；官方pack/unpack 14维往返通过。首次连接受本机HTTP代理变量影响；脚本随后明确设置回环直连，v1与加入打包核对的v2均通过。证据`outputs/pusht/checks/ws_transport_v2/result.json` ok:true、退出码0；小型记录在`experiments/pusht/ws_transport_20261107/summary.json`，用法与限制见[设备与客户端说明](docs/pusht/DEVICE_AND_CLIENT.md)。
- 子代理核对：官方数据打包顺序为左臂6、左夹爪1、右臂6、右夹爪1；EvalEnv夹爪0～1映射为原始行程，动作块逐条执行，观测采样与25Hz控制需保持对齐。这些是源码分析，不是已运行ACT模型的证据。
- 主代理使用实际上游EvalEnv方法做无GPU映射检查：14维字段校验、夹爪映射（u=0→-0.01m，u=0.5→0.017m）与10步控制序列通过，前8步插值、后2步保持。初次测试预期误把首步写成1/5，按源码改为1/9后通过；最后Push-T 8项无GPU测试全部通过。证据入口`tests/pusht/test_eval_action_mapping.py`。这仍不等于真实EvalEnv与仿真物理联动。

下一步验收实际仿真进程与独立策略进程连接、EvalEnv.take_action及归一化/插值，随后核验少量示范字段、图像和时间对齐；真实ACT权重、训练与闭环尚未运行。本轮无GPU使用、未commit、push或发布。

## 最新进度：真实EvalEnv与独立假策略单步联通（2026-11-07）

主代理新增`scripts/pusht/checks/serve_dummy_policy.py`和`check_eval_link.py`，用既有`act`与`pusht_sim`两个环境分别运行真实PolicyServer和Isaac EvalEnv。启动前及仿真导入前均检查GPU3完整UUID和空闲；服务仅监听127.0.0.1。v2证据`outputs/pusht/checks/eval_link_sim_20261107_v2/`和`eval_link_server_20261107_v2/`：reset与动作请求各1次，三路640×480×3 uint8相机图像抵达策略进程，双臂6＋1＋6＋1动作返回并经EvalEnv.take_action执行；collect_interval=10，左臂首关节实际增量0.008826rad，目标增量0.01rad。仿真退出码0、ok:true、日志无[Error]/Traceback，GPU结束后5MiB、0%。第一轮v1日志显示动作曾执行，但结果在Isaac关闭后才写导致缺失，相机摘要误用`colors`字段；修正后重新跑v2，不计v1通过。v2后增加显式图像/动作字段断言，依据v2两端数据可满足，尚未重跑新断言。复现见[设备与客户端说明](docs/pusht/DEVICE_AND_CLIENT.md)，小型记录见`experiments/pusht/eval_link_20261107/summary.json`。

本次主代理负责脚本、环境及GPU仿真。两个按用户指定`gpt-5.6-luna`、low运行的只读子代理完成EvalEnv/PolicyServer调用链和ACT字段、数据读取器审计；没有安装、占用GPU、修改文件或提交。此次仍是假策略、单布局、单动作；真实ACT模型、完整评测及官方示范内容和时间对齐未验收。下一步先抽查少量示范HDF5与图像/动作时序，再准备真实ACT最小训练和闭环。未commit、push或发布。

## 当前状态：布局0稳定性、重复重置与低层动作检查通过（2026-11-07）

本节为最新状态，下方阶段记录保留历史边界。用户授权继续初态与动作接口验收，并要求主代理/子代理分别汇报。

- 主代理新增 `src/mlpj/pusht/task_checks.py` 及薄入口 `scripts/pusht/checks/check_task_interfaces.py`。调用原生TaskEnv、固定布局0，记录冻结参数/源码/上游diff/布局SHA256，独立worker有超时、输出不覆盖与父入口及worker两次GPU UUID/空闲核验；不安装依赖或改动物理、任务成功条件。内部父PID检查用于防误调用，不是面对主动伪造的安全认证。
- 两个只读gpt-6-luna/low子代理分别核对布局恢复/稳定性副作用和动作单位/文档，并汇总冻结参数及日志。主代理复核纠正TaskEnv与闭包EvalEnv.take_action、配置来源的混淆；子代理发现worker可误绕过入口守卫，主代理补上启动前复核与重复进入保护。
- 首轮 `outputs/pusht/checks/task_interfaces_v1/` 因主代理新增结果中的OmegaConf ListConfig不能JSON序列化失败，异常后未正常收尾，主代理仅停止自己对应worker；退出码-9、未超时，日志与冻结源码保留，GPU释放2MiB/0%。未把首轮计为通过。修复序列化并通过真实OmegaConf/NumPy无GPU回归后，使用新目录v2重跑。
- 最终证据 `outputs/pusht/checks/task_interfaces_v2/`：计划/完成重置3/3，全部原生稳定性检查及额外250步（dt=.004下1秒）静置通过；记录最大位置漂移0，角漂移约2.41e-6度。两组重复比较通过，T位置最大差7.03e-8米、角差约0.000133度。这里是同一初态重复，不是三个独立测试初态或任务成功率。
- T保存位置[-0.4174957,-0.1169327,0.7725]与首轮实际位置差约2.90e-8米，说明靠左来自官方布局设置；静置与小幅动作未记录到T位置变化，但没有采集接触力或穿透深度，不能宣称完全无接触。
- 两臂分别joint1增加0.02rad、夹爪原始关节减少0.002m再返回保存目标，4/4响应及回位通过。通过TaskEnv.step([MetaControl])调用原生伺服，不经过CuRobo轨迹规划。臂最大目标误差3.21e-8rad，夹爪最大目标误差2.94e-7m，夹爪最大回位误差2.98e-7m；另一臂保持检查通过，保存三路图像。
- v2退出码0、无[Error]/Traceback，保留上游材质等警告；运行前父入口/worker均检查物理3号卡UUID及2MiB/0%，退出后3MiB/0%。7项无GPU测试、pip check、语法、文档路径及diff检查通过。小型记录见 `experiments/pusht/interface_check_20261107/summary.json`，协议见 [环境接口检查](docs/pusht/TASK_INTERFACE_CHECK.md)。
- 本入口包含原生check_layout_stability，但不等于完整EvalEnv评测流程；额外诊断推进物理步，不调用策略服务器或完整观测/布局有效性对齐流程。稳定性标志仅用于诊断，不是课程成功标准。

下一步由主代理接通并验收EvalEnv/XPolicyLab策略通信、14维动作打包、夹爪0～1映射及原生插值；随后核验少量示范字段与时序，再决定训练。子代理继续只读整理通信日志、核对字段/路径/文档；不安装、不使用GPU、不改文件、不提交。本次没有启动策略服务、下载完整示范或训练ACT，没有改动Lift；未commit、push或发布，保留用户原有暂存项。

## 当前状态：最小资产、X5规划器与固定布局三路相机（2026-11-07）

本节优先于下方历史阶段的“未下载资产/未创建规划器/未实例化场景”描述。用户授权最多两个gpt-6-luna、low推理子代理，限定只读日志与文档核查；主代理复核结果并负责修改、GPU仿真。子代理未安装依赖、使用GPU、修改文件或提交代码。

- 资产固定数据集revision `1127bda94f172d757386945af3eb3a6754d1a4e6`，269个源文件共415,490,428字节，位于 `data/pusht/robodojo_assets/1127bda94f17/Assets/`，上游Assets为相对链接。2026-11-07重新读取所有本地文件，大小及SHA256全部匹配原manifest。不是仅有下载计划，也没有下载完整示范。详情见 [资产与规划检查](docs/pusht/ASSETS_AND_PLANNER.md)。
- `src/mlpj/pusht/assets.py`及薄入口 `scripts/pusht/data/prepare_x5_planner.py`根据官方curobo_tmp.yml生成本地路径版curobo.yml及来源校验记录；保留原模板，异内容拒绝覆盖。生成文件随检出位置变化，不能移机直接复用。
- 先前X5真实URDF完整CuroboPlanner初始化、原配置warmup与一次自身目标规划通过，证据 `outputs/pusht/checks/x5_planner_fam191ud/`：退出码0、34.77秒、PyTorch峰值约2.45GiB。它不是任务推动成功率，也不是总显存。
- 本轮发现TiledCaptureManager输出缓冲区硬编码cuda:0，补充capture_device传递，沿用显式规划设备且保留CPU物理。设备补丁现在涉及五个源码文件（新增tiled_capture_manager.py），不是改变动作、物理参数或任务指标。四项无GPU路由回归通过，pip check正常；补丁reverse检查与语法检查通过。
- 本轮实际加载官方 `push_T / arx_x5` 布局0，使用原process_config与随机化处理、单环境、dt=0.004、25Hz相机配置。直接实例化原生任务类，重置后恢复保存位姿、机器人初态和200步更新，检查头部/左腕/右腕三路640×480 RGBA。三张图像非空、有限数值，缓冲区均cuda:3；physics_device=cpu、planner_device=cuda:3，断言GPU0无PyTorch张量分配。该断言不证明所有库都未创建其他卡上下文。
- 证据 `outputs/pusht/checks/task_scene_kb88rdl4/` 含检查脚本、有效配置、三张PNG、日志、result.json、退出状态及GPU前后快照。ok:true、进程退出码0、日志无[Error]/Traceback；主代理目视确认双臂、红T和灰色目标。存在上游USD材质引用越界、MDL编译、无显示及音频警告，不能据出图宣称所有材质正确。运行前实时确认物理3号卡UUID/1MiB/0%，结束后2MiB/0%。
- 本次仅验收场景加载、一次重置和三路成像。未调用完整EvalEnv稳定性检查、未验证重复重置、受控关节动作、示范回放、WebSocket策略服务、ACT训练或闭环；尤其图中T靠近左夹爪，仍需核对保存初态与碰撞/稳定性，不能直接当合格评测初态。

下一步先验收布局恢复与稳定性、重复重置和小幅受控动作，再接策略通信与示范数据。没有升级依赖、修改Lift内容或重跑Lift；未commit、push或发布，保留用户原有暂存项。下方记录保留其当时的阶段范围。

## 最新增量：设备补丁、真实内核与客户端已检查（2026-11-04）

用户授权继续检查、配置环境。本轮没有训练或正式评测。

- 发现官方device_id未映射至AppLauncher.device；物理配置默认CPU，CuRobo却默认cuda:0。已在四个上游文件作最小设备补丁：入口对齐device_id并关闭多卡、TaskEnv分离planner_device、RobotManager传递规划器设备、CuroboPlanner设置显式CUDA设备及当前设备。不更改原物理设备、dt、频率、动作和成功标准。补丁保存env/pusht/robodojo-planner-device.patch，子模块保持原提交但工作区有本地修改。
- GPU使用前各自重新核对物理3号卡UUID/1MiB/0%。隔离两关节模型执行三组CuRobo GPU FK与反向传播，对解析公式最大位置误差8.742277657347586e-08、梯度误差0，退出码0。
- 同进程AppLauncher cuda:3、RoboDojo TaskEnv/评测环境/Push-T类导入、CuRobo显式cuda:3 FK梯度及相机三帧整合检查通过，断言GPU0未分配PyTorch张量；显式ok:true、退出码0、无[Error]日志，PNG目视正常。退出后3号卡1MiB/0%。
- XPolicyLab固定检出5d0be1499634d3176258c8f2e31f5fe816497b0a，确认包含bb9a0b5官方修复。补齐transforms3d0.4.2、msgpack1.2.4、msgpack-numpy0.4.8；14维float32动作与uint8 RGB数组官方编解码往返通过。未启动策略服务、连接WebSocket或安装ACT模型。ffmpeg已存在。
- 三项无GPU设备路由测试通过，最终pip check通过。可复现说明见docs/pusht/DEVICE_AND_CLIENT.md，新增约束在env/pusht/constraints-client.txt，证据在outputs/pusht/checks/curobo_gpu_aqnbel8c/，整合结果在integration/。
- 尚未下载官方资产或实例化X5场景；完整IK、碰撞/运动规划器、CUDA Graph与warmup未验证，不能将FK与导入成功写成Push-T成功。下一步准备最小场景资产后再验收单场景。未commit、push或发布，保留用户此前暂存内容。

## 最新增量：CuRobo安装与接口检查（2026-11-04）

用户授权继续准备CuRobo。已初始化RoboDojo指定子模块d17b54ce32cba095c0b000c4c58777075d11de0e，源码未改。复用pusht_sim安装cu12基础依赖，cuda-core0.7.0、cuda-bindings12.9.10；原有全部包版本保持不变，runtime/NVRTC仍12.6.77，没有安装系统nvcc/CMake/Ninja。首次bindings12.8缺cynvvm，失败证据保留，更新Python绑定后通过。

pip check正常。屏蔽所有CUDA设备的独立进程中，RoboDojo需要的CuRobo类及CUDA Core接口导入通过，CUDA未初始化；未创建规划器、编译GPU内核或运行仿真。本轮没有使用GPU。详细记录及复现命令见docs/pusht/CUROBO_SETUP.md，追加约束在env/pusht/constraints-curobo-cu12.txt，完整日志在outputs/pusht/checks/curobo_compat_aeghgh7c/。

后续GPU运行前必须处理设备问题：RoboDojo规划器DeviceCfg()硬编码默认cuda:0，而RobotManager没有传入self.device；仅选择Isaac GPU3不能保证CuRobo使用同一卡。下一步应处理显式设备传递并做小规模GPU内核/规划验证，再准备XPolicyLab和场景资产。X5 Push-T场景及ACT训练/闭环仍未完成。本次未commit、push或发布，保留用户先前暂存的子模块引用。

## 最新状态：Isaac Lab兼容安装已验证（2026-11-04）

本节替代下文历史“尚未安装Isaac/尚未下载RoboDojo”等当前状态描述；历史记录保留，不代表本轮重跑Lift或完成Push-T任务。

- 用户先前手动完成pusht_sim（Python3.11.17）及Isaac Sim5.1安装；wheel降到0.45.1，安装系统libGLU.so.1后，无窗口启动与相机三帧通过，用户目视确认红色方块，退出码0且GPU释放。原相机证据在outputs/pusht/checks/isaac_camera_bhKyxs/。
- RoboDojo已作为子模块加入third_party/pusht/RoboDojo，固定f57e1633ffed59761bc8e568cfe7530c4b33381c；其Isaac Lab子模块固定afca7b09d60d8beb9c1cb28b43066499940b969b。.gitmodules和RoboDojo引用为用户先前暂存内容，尚未提交。
- 用户本轮明确授权代理寻找并实践兼容方案。Lab声明Starlette0.49.1，与Isaac kernel固定FastAPI0.115.7要求的Starlette<0.46冲突。已将Lab该行改为RoboDojo脚本采用的0.45.3，并保存env/pusht/isaaclab-starlette.patch；嵌套子模块因此显示修改。父gitlink不保存补丁，复现须应用保存的补丁。
- 保留既有Isaac/torch/NumPy等全部版本，仅将setuptools83降为81。已在pusht_sim正常安装core/assets/tasks/rl/mimic五个Lab包的基础依赖，无RL或Robomimic extras。首次仅装前三包时任务导入缺h5py，失败日志保留；默认kit会加载五个扩展，补齐依赖后pip check通过。
- 本轮实际AppLauncher、任务配置导入及三帧256×256相机回归通过，显式ok:true、退出码0、日志无[Error]，PNG目视确认正常。启动前检查GPU3为1MiB/0%，UUID匹配，渲染及PhysX使用3号卡；退出后2MiB/0%。完整报告、安装前后版本、日志和失败证据在outputs/pusht/checks/dependency_compat_wptldpq6/，最终回归在runtime2/。
- 可复现步骤见docs/pusht/ISAACLAB_COMPATIBILITY.md；环境要求及版本约束在env/pusht/。约束是本机实测记录，不是空白机器可移植锁文件。
- 此节记录Isaac Lab安装阶段，当时CuRobo和XPolicyLab未初始化；后续CuRobo状态见最新增量。官方任务资产、X5 Push-T场景及ACT训练/评测仍未验收，不要把基础环境通过写成Push-T成功。
- 用户要求每次占用GPU前检查空闲。平时按“指导用户操作—用户反馈—下一步”推进；本轮环境安装与回归为用户明确授权实践，不代表后续任务自动执行。此次未commit、push或发布。

## 维护范围

本文维护 Push-T 任务的进度、环境验收、路线选择和待办；Lift 任务进度归 [HANDOFF_LIFT.md](HANDOFF_LIFT.md)。Push-T 专属更改更新本文；共享代码、目录、环境或协作规则等同时影响两项任务的更改，必须同时更新两份交接，分别记录任务影响、验证和限制。

## 目录约定

项目包名为 `mlpj`。Push-T实现归 `src/mlpj/pusht/`，入口归 `scripts/pusht/`，配置、数据、结果、实验说明和第三方依赖分别归对应顶层目录的 `pusht/`。当前已有环境兼容补丁、资产准备工具和单场景检查证据；RoboDojo、Isaac及最小资产已准备，仍无Push-T训练或评测闭环入口。Lift已归各目录的 `lift/`，配置为 `configs/lift/default.yaml`。共享工具归 `src/mlpj/common/`。

## 共享结构调整与验证（2026-11-04）

代码按 `src/mlpj/common/`、`src/mlpj/lift/`、`src/mlpj/pusht/` 组织，各任务的scripts、configs、data、outputs、experiments、third_party、docs、env及archive分开。Push-T目前仅有包和目录边界；后续仿真端和策略端分别注册mlpj，依赖环境保持独立，不能直接复用Lift的20Hz、状态动作语义和高度指标。

先前目录调整在/tmp独立验证环境完成mlpj可编辑安装，测试发现12项，11项通过、1项因缺少Lift轨迹缓存跳过；入口、根目录定位、源码快照、资源边界和链接检查通过。这些是公共结构与Lift现有代码的验证，不代表Isaac、RoboDojo、Push-T数据或闭环已验收。当前仓库没有完整示范、权重和历史outputs，本次没有迁移这些大资源。

本轮统一交接文件名为HANDOFF_LIFT.md与HANDOFF_PUSHT.md，README及AGENTS明确“任务分别维护、共同改动同时更新”。只进行文档链接、引用、命令路径和diff检查，未运行训练或仿真；未commit、push或发布。

## 共享目录补迁完成（2026-11-04）

用户授权后，本机旧位置的Lift数据、权重、历史输出、可选ACT上游和归档已分别归入lift任务目录；49项移动、1703个文件逐文件SHA256保持一致，未覆盖跟踪代码或元信息。旧缓存保留在archive/lift/cache_snapshots/20261104/。data/outputs/third_party/archive均只保留lift/pusht边界，没有把Lift资源放入Push-T目录。

现有WSL ml环境仅可编辑注册本仓库mlpj，没有安装或升级机器人/训练依赖；其他将来用于Isaac/RoboDojo的环境仍需独立注册。公共与Lift测试12项全部通过；Lift无相机物理与单初态GPU/双相机回归通过，不代表Push-T仿真、数据或训练验收。Push-T仍仅为预留目录，无新实现或实验结果。

共享补迁已解决此前记录的本机资源错位，历史冻结文件原文保留；仍未找到的原理稿需查原工作区。没有下载RoboDojo、安装Isaac或改动Push-T内容。未commit、push或发布。

## 共享目录初查补充（2026-11-04，迁移前）

d4a4bc3后当前检出核查：common/lift/pusht包、薄入口和任务预留边界清楚，初始化不导入仿真，文档相对链接通过。当前可复用的WSL ml环境尚未注册mlpj；使用临时PYTHONPATH做公共/Lift测试为11通过、1跳过，不代表Push-T环境或任务验收。

当前机器确有约3.028 GiB的Lift数据和输出，仍位于旧目录，新task资源位置基本为空；此前“当前仓库没有完整资源”的描述属于迁移当时的环境记录。旧third_party/act也仍在新任务边界之外且未被忽略。处理时只能归入lift对应位置，不得混入pusht，更不能覆盖未来RoboDojo/XPolicyLab。Push-T仍没有可运行实现或新实验结果。

本次只读取和记录，没有移动资源、注册环境、安装Isaac/RoboDojo或运行Push-T。详细核查见docs/STRUCTURE_AUDIT.md；后续共享结构或资源调整仍须同步两份任务交接。未commit、push或发布。

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

环境、固定布局与假策略单步联通，以及两条示范的CPU数据组件已检查，详见顶部最新接手状态。下一步准备并由用户执行真实ACT单批次计算检查，再接通训练入口；每次使用GPU前检查实时空闲。若仿真机器暂不可用，可先核验数据字段和 XPolicyLab 版本，Panda 路线仍可独立推进。使用独立输出目录，保留 Lift 基线与失败证据。大数据、模型和完整视频按仓库已有资源边界管理。提交与推送分别授权。

用户随后要求提交资源整理对应的修改：本地提交两份任务交接、公共结构说明及核查记录；迁移后的数据、权重、缓存、上游检出和运行产物继续忽略，不纳入Git。沿用已完成的完整性与回归验证，不重复实验；不push或发布。
