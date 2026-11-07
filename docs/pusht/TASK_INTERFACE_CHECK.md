# 固定布局、稳定性与原生伺服检查

该入口用于环境诊断，不连接策略服务，不执行ACT，不计算Push-T任务成功率。需要已配置的仿真环境、固定版本RoboDojo及其设备补丁、最小资产和生成的X5规划配置，见[资产与规划检查](ASSETS_AND_PLANNER.md)及[设备说明](DEVICE_AND_CLIENT.md)。先在所用Python环境注册mlpj。

从仓库根目录执行，先用nvidia-smi检查实时状态，并将以下两个占位参数替换为同一张空闲GPU的物理编号和完整UUID：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
python scripts/pusht/checks/check_task_interfaces.py \
  --output outputs/pusht/checks/task_interfaces_new \
  --device-id <GPU_INDEX> --gpu-uuid <GPU_UUID>
```

入口再次校验UUID、显存小于10MiB和利用率为0%，不接受CUDA_VISIBLE_DEVICES重映射；输出目录必须不存在。worker在启动AppLauncher前再次复核GPU，拒绝脱离父启动器直接调用和重复进入同一输出目录。检查仍无法替代共享机器的预约机制。默认超时360秒，超时结束独立worker进程组并保留证据；不能把超时视为检查通过。

默认协议固定布局0，单环境，三次重置。沿用官方process_config/process_randomization、dt=0.004和25Hz相机配置；不改摩擦、重力、碰撞或成功标准。每次重置后分阶段记录实际物体/机器人位姿，恢复保存布局，执行原生稳定性检查，初始化机器人控制状态，再更新200步，并记录250步（1秒）静置样本。

原生check_layout_stability推进300个物理步，并可能改写env.success和env.end_flag；入口记录这些标志，它们只是本次诊断中的原生布局筛查结果。该筛查允许较大漂移（4cm、全程30°及末窗口10°），所以另设更严格的静置诊断：T物体相对静置起点的最大位移不超过1mm、角差不超过1°。三次重置最终物体位姿也按1mm/1°比较，机器人原始关节差按0.001检查；臂关节单位rad、夹爪关节单位m，统一数值阈值并不代表相同物理量。严格阈值是本项目的诊断条件，不是课程或上游任务成功指标。本入口不调用EvalEnv整套布局有效性对齐、观测管理和策略重置流程；诊断阶段还推进额外静置物理步，不能称为完整复现官方评测。

稳定性通过后，对每条机械臂分别执行joint1增加0.02rad、夹爪主关节减少0.002m的目标，再返回保存的初始关节目标。每个目标保持100个物理步，使用TaskEnv.step([MetaControl])和原生RobotManager控制路径，不调用CuRobo生成运动轨迹。记录实际响应、目标误差、另一条臂的变化、回位误差及三路图像。臂目标误差/回位误差阈值为0.003rad，夹爪为0.0005m，动作幅度至少达到目标一半且方向一致；另一条臂的臂关节变化不超过0.001rad。

部署EvalEnv.take_action与这里的原生TaskEnv不是同一接口。前者在create_eval_env中定义，joint动作的夹爪标量经clip到0～1，再按当前X5范围映射为`-0.01 + 0.054*u`米，并插值执行。本入口不证明该上层转换、WebSocket通信、数据标签或示范时序正确。不要把原始夹爪米数直接输入部署归一化接口。

输出包括protocol.json、源码快照、effective-config.yaml、result.json、运行日志、进程退出状态、GPU前后快照和各阶段PNG。逐次保存结果，任何异常均保留已完成记录与计划重置数，不筛掉困难初态。即使诊断全部通过，也只能说明这个固定布局和小幅动作；不证明任意布局、碰撞安全、任务完成或ACT闭环性能。

## 本机运行记录（2026-11-07）

v1为检查脚本的ListConfig结果序列化失败，主代理停止其异常worker，退出码-9，失败日志独立保留；不是环境检查通过。修复后v2退出码0，3/3重置稳定性及静置、两组重置比较、4/4低层动作响应和回位通过。完整证据在`outputs/pusht/checks/task_interfaces_v2/`，小型结果与校验值见 [实验记录](../../experiments/pusht/interface_check_20261107/README.md)。T实际位置与官方保存布局吻合，静置及小幅动作未记录到T位置变化；没有检测接触力或穿透深度，不能据此排除接触。

脚本的内部worker父PID检查用于防止误调用，不能作为对主动伪造调用的身份认证或安全边界。正常使用必须从公开入口启动，保留父进程超时和完整证据管理。
