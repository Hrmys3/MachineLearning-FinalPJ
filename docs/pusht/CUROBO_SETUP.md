# 固定版本CuRobo安装与接口检查

## 来源与范围

RoboDojo f57e1633ffed59761bc8e568cfe7530c4b33381c指定的CuRobo fork提交为d17b54ce32cba095c0b000c4c58777075d11de0e。包元信息版本为0.0.post1.dev100，属于该源码提交的SCM计算结果；复现以提交号为准。本轮未修改CuRobo源码。

已完成固定源码安装、依赖一致性及屏蔽全部CUDA设备的接口导入。没有创建运动规划器、编译/执行GPU内核，也没有加载X5任务。不能将导入成功描述为GPU规划或RoboDojo任务验收。

## 兼容选择

- 复用已验收的pusht_sim环境与[Isaac Lab兼容方案](ISAACLAB_COMPATIBILITY.md)，冻结所有原有包。补装setuptools_scm10.2.1及其依赖，不重装Isaac、torch或NumPy。
- 此CuRobo默认使用CUDA Core运行时编译，未启用CUROBO_USE_PYBIND，无需先安装系统nvcc/CMake/Ninja来构建Python包。实际GPU内核编译仍需后续验证。
- 无约束cu12预演会选入较新的CUDA工具组件。本次选择cuda-core0.7.0，且不引入cuda-bindings的all额外组件。首次将绑定固定12.8.0时，cuda.core导入报缺少cuda.bindings.cynvvm；改为cuda-bindings12.9.10后接口导入通过。
- 12.9.10是Python绑定版本，不表示升级了系统CUDA工具链或驱动。现有nvidia-cuda-runtime-cu12和nvidia-cuda-nvrtc-cu12均保持12.6.77；没有安装nvidia-cuda-nvcc-cu12或cuda-toolkit。是否能执行实际内核另行验收。

## 复现安装

在已完成Isaac Lab兼容准备的Python3.11仿真环境中，从仓库根目录执行。必须使用以上固定源码提交，避免--remote更新。

```bash
git -C third_party/pusht/RoboDojo submodule update --init third_party/curobo
python -m pip install -c env/pusht/constraints-isaac51.txt \
  -c env/pusht/constraints-curobo-cu12.txt setuptools_scm
python -m pip install --no-build-isolation \
  -c env/pusht/constraints-isaac51.txt \
  -c env/pusht/constraints-curobo-cu12.txt \
  -e 'third_party/pusht/RoboDojo/third_party/curobo[cu12]'
python -m pip check
```

约束文件记录当前Linux x86_64环境实测版本，未完成空白机器复现，不是完整可移植锁文件。

## 本轮实际验证（2026-11-04）

证据目录：outputs/pusht/checks/curobo_compat_aeghgh7c/。保留安装前后版本、两份解析预演、安装报告、首轮失败与第二轮成功日志及summary.json。

pip check通过；对比本轮安装前全部发行包，既有版本无变化，新增包记录在约束中。屏蔽CUDA_VISIBLE_DEVICES后导入BatchMotionPlanner/MotionPlannerCfg、InverseKinematics/InverseKinematicsCfg、MotionPlanner、DeviceCfg/GoalToolPose/JointState/Pose/ToolPoseCriteria和curobo.runtime，以及cuda.core的LaunchConfig/Program/ProgramOptions/Device。输出CUROBO_API_IMPORT_OK、CUDA_CORE_API_IMPORT_OK、CUDA_NOT_INITIALIZED，退出码0，torch.cuda.is_initialized()仍为False。本轮未占用GPU。

## 初查发现的设备问题（已由后续补丁处理）

RoboDojo env/planner_manager/curobo_planner.py通过DeviceCfg()创建规划器设备；CuRobo该默认值硬编码cuda:0。RobotManager虽持有self.device，创建CuroboPlanner时却未传入设备。这一源码事实需要在后续场景运行前处理。仅把Isaac的渲染GPU设为3，或仅调用torch.cuda.set_device(3)，不能改变该dataclass的显式cuda:0默认值。

后续应把仿真设备显式传给规划器并验证所有组件的物理UUID，或采用经核对的统一设备映射方案；每次启动前检查GPU空闲。当前未修改该行为，也未创建会使用GPU0的规划器。XPolicyLab、官方资产和单场景验收仍未完成。本轮未commit、push或发布。

## 后续验证更新

本说明前面的“未测试GPU内核/未修改规划器”属于初查阶段。随后已完成显式设备补丁、两关节GPU FK/解析梯度对照，以及Isaac相机、RoboDojo类导入与CuRobo cuda:3同进程检查。设备补丁保留CPU物理配置，不直接将CPU传入CUDA规划器。详见[设备与客户端记录](DEVICE_AND_CLIENT.md)。后续X5真实URDF完整规划器初始化、原配置warmup与一次自身目标规划通过；证据与限制见[资产与规划检查](ASSETS_AND_PLANNER.md)。随后已完成固定官方布局0加载、一次重置及三路相机出图；布局稳定性、重复重置、动作接口与策略闭环仍需单独验收。
