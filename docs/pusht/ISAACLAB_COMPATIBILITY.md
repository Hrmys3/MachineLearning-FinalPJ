# Isaac Lab 与 Isaac Sim 5.1 的依赖兼容

适用来源：RoboDojo `f57e1633ffed59761bc8e568cfe7530c4b33381c`，其指定的 Isaac Lab fork `afca7b09d60d8beb9c1cb28b43066499940b969b`。核心包自身版本为0.54.3，不能只用README中的Isaac Lab 2.3标签替代提交号。

## 冲突与处理

Isaac Sim kernel 5.1.0.0固定FastAPI 0.115.7；FastAPI要求Starlette >=0.40,<0.46。固定Lab版本却声明Starlette ==0.49.1，无法同时满足。RoboDojo安装脚本又固定Starlette 0.45.3。当前补丁仅将Lab的该项声明改为0.45.3，与已验收的Isaac环境保持一致，不修改仿真、控制或评价代码。Lab包内未发现Starlette应用代码，但最终仍以安装检查和实际导入/渲染验证为准。

补丁保存为 `env/pusht/isaaclab-starlette.patch`。正常安装依赖，未使用--no-deps跳过Isaac Lab依赖。`env/pusht/constraints-isaac51.txt`记录本次解析版本，属于当前平台的约束，不是经过空白机器验证的完整锁文件。

## 复现方式

以下命令从仓库根目录执行，Python须属于已准备Isaac Sim 5.1且通过基础相机测试的仿真环境。先确认以上两个源码提交，不能直接对其他版本应用补丁。

```bash
git apply --check --directory=third_party/pusht/RoboDojo/third_party/IsaacLab env/pusht/isaaclab-starlette.patch
git apply --directory=third_party/pusht/RoboDojo/third_party/IsaacLab env/pusht/isaaclab-starlette.patch
python -m pip install --no-build-isolation -c env/pusht/constraints-isaac51.txt \
  "setuptools<82" \
  -e third_party/pusht/RoboDojo/third_party/IsaacLab/source/isaaclab \
  -e third_party/pusht/RoboDojo/third_party/IsaacLab/source/isaaclab_assets \
  -e third_party/pusht/RoboDojo/third_party/IsaacLab/source/isaaclab_tasks \
  -e third_party/pusht/RoboDojo/third_party/IsaacLab/source/isaaclab_rl \
  -e third_party/pusht/RoboDojo/third_party/IsaacLab/source/isaaclab_mimic
python -m pip check
```

已应用补丁时，第一条检查会失败；用以下命令确认当前源码确实包含这一补丁，不要重复应用：

```bash
git apply --reverse --check --directory=third_party/pusht/RoboDojo/third_party/IsaacLab env/pusht/isaaclab-starlette.patch
```

补丁应用会使嵌套Isaac Lab子模块工作区显示修改。父仓库的gitlink本身不会保存未提交补丁，因此重建环境必须保留并应用本仓库中的补丁文件。不要把本地修改误记成新的上游提交。

## 实际启动暴露的额外依赖

AppLauncher默认使用的isaaclab.python.kit会同时加载core/assets/tasks/rl/mimic五个本地扩展。仅安装前三个包虽然pip check通过，任务导入仍因缺少h5py失败；其依赖声明位于isaaclab_rl。最终需安装五个包的基础依赖，不启用RL或Robomimic extras，不运行训练。保持packaging 23和NumPy 1.26的约束，IPython限制<9，与RoboDojo脚本一致。

首轮失败日志保留在runtime.log。回归脚本在app.close之前保存显式结果JSON；仅有退出码0而没有成功标记，不能算通过。

## 证据与边界

本次检查目录为 `outputs/pusht/checks/dependency_compat_wptldpq6/`，包含安装前版本、约束、解析报告及安装日志。后续导入与相机回归也在该目录保存，不覆盖此前相机证据。

Isaac Lab兼容性检查不等于RoboDojo任务验收。XPolicyLab、CuRobo和官方场景资产仍需分别准备与验证；本轮不训练ACT，不运行Push-T策略闭环。

## 本次验证结果（2026-11-04）

- 两阶段安装均成功，最终pip check输出No broken requirements found。
- 对比安装前后的所有既有发行包，只有setuptools从83.0.0降为81.0.0；Isaac Sim、torch、NumPy、FastAPI、Starlette等保持原版本，其他为新增依赖。
- runtime2/check_isaaclab.py通过AppLauncher启动，实际导入DirectRLEnvCfg、ArticulationCfg和isaaclab_tasks.utils.parse_env_cfg，输出ISAACLAB_IMPORT_OK。
- 连续三帧均为256×256×4，像素范围38—227，PNG目视确认灰色地面中央红色方块，方块中心z约0.1m。出现三条FRAME_OK及CAMERA_OK，显式结果JSON为ok:true，退出码0；最终日志无[Error]行，仍有无窗口GLFW等警告。
- 启动前核对物理GPU3 UUID为GPU-0b9f2b02-f653-5a07-7c23-9c8640fa5e62，显存1MiB且利用率0%；渲染活动卡及PhysX日志均指向3号卡。退出后显存2MiB且利用率0%。

本轮未下载CuRobo/XPolicyLab/任务资产，未加载X5 Push-T场景，未训练或评测ACT。不能用该检查推断任务成功率或训练性能。未commit、push或发布。
