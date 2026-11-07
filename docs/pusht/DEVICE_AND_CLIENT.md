# Push-T设备配置与客户端环境检查

当前验收覆盖固定布局稳定性、低层动作及真实EvalEnv到独立假策略的一次动作，详见下方各阶段记录；两条示范和CPU ACT数据组件见[数据与ACT适配](DATA_AND_ACT.md)。真实ACT模型计算与完整闭环仍待执行。

## 固定源码

RoboDojo使用f57e1633ffed59761bc8e568cfe7530c4b33381c，CuRobo使用d17b54ce32cba095c0b000c4c58777075d11de0e，XPolicyLab使用5d0be1499634d3176258c8f2e31f5fe816497b0a。已通过git merge-base确认XPolicyLab包含官方要求的bb9a0b5修复提交；没有使用--remote追踪新版本，也未修改XPolicyLab源码。

## 设备问题与补丁

原入口只有device_id，而AppLauncher使用独立的device参数，二者未关联。原物理配置没有device时使用CPU；RobotManager也据此持有CPU设备，但CuRobo的DeviceCfg()默认为cuda:0。不能直接把CPU物理设备交给CUDA规划器，也不能只改渲染卡就认为规划器已对齐。

补丁env/pusht/robodojo-planner-device.patch修改五个源码文件：

- main.py在启动AppLauncher前将device_id映射到cuda:N，禁止负编号，并拒绝显式device与device_id冲突；单卡运行关闭multi_gpu。
- TiledCaptureManager的相机输出缓冲区跟随TaskEnv传入的capture_device，避免硬编码cuda:0；物理设备仍独立。
- TaskEnv将规划器设备作为独立参数，优先读取planner_device，否则取eval_cfg.device_id，不更改原有sim.device物理配置。
- RobotManager显式传递planner_device；CuroboPlanner据此创建DeviceCfg，拒绝非CUDA设备，并设置当前CUDA设备供底层编译器使用。

补丁不改控制频率、时间步长、物理参数、动作或成功判定。原物理默认CPU保持不变。未来如需GPU物理，须另行配置、记录并验证。直接调用CuroboPlanner而绕过TaskEnv时，调用方仍须显式传device；其默认cuda:0保留了原接口兼容。

从仓库根目录应用到以上固定版本：

```bash
git apply --check --directory=third_party/pusht/RoboDojo env/pusht/robodojo-planner-device.patch
git apply --directory=third_party/pusht/RoboDojo env/pusht/robodojo-planner-device.patch
```

已应用时改用reverse检查，不重复应用：

```bash
git apply --reverse --check --directory=third_party/pusht/RoboDojo env/pusht/robodojo-planner-device.patch
python -m unittest tests.pusht.test_device_configuration -v
```

测试读取实际补丁方法并用模拟规划器验证设备转发，不导入Isaac或初始化CUDA；普通克隆若尚未检出RoboDojo会跳过。补丁独立保存在主仓库，父gitlink不保存子模块未提交改动，复现时不能遗漏应用步骤。

## 客户端依赖

实际缺少的transforms3d、msgpack、msgpack-numpy已在原有约束下补齐，版本见env/pusht/constraints-client.txt；未安装策略模型、ACT训练环境或不必要的Open3D。ffmpeg可执行文件已存在。

```bash
python -m pip install -c env/pusht/constraints-isaac51.txt \
  -c env/pusht/constraints-curobo-cu12.txt \
  -r env/pusht/constraints-client.txt
python -m pip check
```

导入WsModelClient后，使用官方encode_frame/decode_frame对uint8 RGB数组和float32的14维动作作往返检查，dtype、形状和数值保持一致。该离线检查当时未启动策略服务；后续真实网络单步和少量JPEG解码见下方记录及[数据说明](DATA_AND_ACT.md)。图像解码须沿用XPolicyLab共享帮助函数，不自行插入BGR/RGB交换。

## 实际验证（2026-11-04）

完整证据：outputs/pusht/checks/curobo_gpu_aqnbel8c/。

1. 每次GPU使用前重新核对物理3号卡UUID、显存和利用率。隔离FK检查只暴露UUID对应的卡，进程内逻辑cuda:0即物理3号卡。
2. 自建两关节URDF，对三个关节配置执行CuRobo GPU FK和反向传播，与解析位置/梯度对照。最大位置误差8.742277657347586e-08，梯度最大误差0，退出码0。没有使用Panda或X5资产，不是机器人任务结果。
3. integration子目录记录无遮挡设备编号下的整合检查：AppLauncher选择cuda:3；同进程导入RoboDojo TaskEnv、规划器、评测环境及Push-T任务类；显式cuda:3运行CuRobo FK和梯度，并断言GPU0没有PyTorch张量分配。记录不证明所有可能库都不会创建其他卡上下文。
4. 同进程继续进行三帧256×256相机检查，出现ROBODOJO_IMPORT_OK、CUROBO_SAME_PROCESS_GPU3_OK、三条FRAME_OK和CAMERA_OK；显式ok:true、退出码0，日志无[Error]，PNG目视正常。退出后物理3号卡显存1MiB、利用率0%。
5. 四项无GPU设备路由回归通过（第四项于2026-11-07补充），覆盖CPU物理/CUDA规划分离、设备转发、单卡渲染、负编号及冲突参数拒绝；最终pip check通过。

后续已下载并校验固定revision的269个最小场景资产，X5真实URDF的完整规划器初始化、原配置warmup与一次自身目标规划通过，见[资产与规划检查](ASSETS_AND_PLANNER.md)。该检查不等于完整任务场景或ACT闭环。2026-11-07检查发现原生三路相机的预分配缓冲区仍硬编码cuda:0，已补充显式设备传递和无GPU回归；此前方块相机检查没有覆盖该链路，随后固定布局三路640×480相机实测缓冲区均cuda:3、退出码0；随后同布局稳定性/低层动作检查已完成，见[任务接口检查](TASK_INTERFACE_CHECK.md)；完整EvalEnv假策略单步结果见本文末节。未commit、push或发布。

## 本机策略服务回环检查（2026-11-07）

`act` Python3.10.22环境已有websockets16.1.1，能导入固定XPolicyLab的PolicyServer与WsModelClient，pip check正常。仿真环境`pusht_sim`的websockets12.0不能导入XPolicyLab服务端的`websockets.asyncio.server`；因此策略服务应使用独立兼容环境，不应为此升级Isaac仿真环境整套约束。

从仓库根目录，在已满足XPolicyLab服务端依赖的策略环境运行：

```bash
python scripts/pusht/checks/check_policy_transport.py \
  --output outputs/pusht/checks/ws_transport_new
```

检查启动本机127.0.0.1临时端口的真实PolicyServer，用确定性假策略执行握手、reset、三路RGB uint8观测传输、get_action，并按官方pack_robot_state/unpack_robot_state核对左臂6＋夹爪1＋右臂6＋夹爪1。内部为本机地址设置NO_PROXY/no_proxy直连，避免系统HTTP代理截获回环WebSocket。输出目录必须不存在。

最终`outputs/pusht/checks/ws_transport_v2/result.json`为ok:true、三路图像逐像素一致、14维打包与拆包一致、退出码0。初次直接连接受当前机器HTTP代理影响，设置回环直连后通过；这是网络环境现象，不是仿真/模型结果。小型记录见`experiments/pusht/ws_transport_20261107/summary.json`。该独立回环检查未包含EvalEnv；后续真实EvalEnv与假策略单步见末节，真实ACT权重与完整闭环尚未运行。

## 上层joint动作映射检查（2026-11-07）

`tests/pusht/test_eval_action_mapping.py`从固定RoboDojo源码读取EvalEnv的实际`validate_action_dict`、`get_action_type`、`take_action_batch`和`process_control_info`方法，用双X5假状态检查14维字段、夹爪0～1到米制的映射与10步插值。u=0映射-0.01m，u=0.5映射0.017m；默认25Hz、dt=.004使每策略动作对应10个物理步，前8步插值、后2步保持，首插值目标的1/9。该测试无GPU、无PhysX/相机，仅核对实际方法在确定性输入上的数值行为。首次测试预期算错插值分母，修正预期后通过；不能据此宣称真实仿真执行或ACT部署通过。

## 真实EvalEnv到本机假策略的单步检查（2026-11-07）

两进程分别使用已有策略Python环境与`pusht_sim`仿真环境。先在终端A启动只监听回环地址的确定性策略服务；服务端的端口写入新输出目录。终端B先实时核查待用GPU的编号、完整UUID、显存和利用率，再运行仿真检查。示例命令均在仓库根目录执行；每次换新输出目录，仿真脚本在导入Isaac前再次核查GPU必须低于10MiB且利用率为0%。

```bash
python scripts/pusht/checks/serve_dummy_policy.py \
  --output outputs/pusht/checks/eval_link_server_new

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
python scripts/pusht/checks/check_eval_link.py \
  --output outputs/pusht/checks/eval_link_sim_new \
  --device-id <GPU_INDEX> --gpu-uuid <GPU_UUID> \
  --port <PORT_FROM_SERVER_PORT_JSON>
```

仿真结束后在终端A按Ctrl+C结束服务以写出其`result.json`。当前机器实测v2：`outputs/pusht/checks/eval_link_sim_20261107_v2/result.json`为ok:true、退出码0；服务端`outputs/pusht/checks/eval_link_server_20261107_v2/result.json`记录reset 1次、动作请求1次、三路640×480×3 uint8相机数据均有限。仿真端通过真实EvalEnv读取观测、经真实WsModelClient发送、接收14维joint动作并调用`take_action`；采样间隔10个物理步，左臂首关节由约0增加至0.008826rad，目标增量0.01rad。GPU3结束后为5MiB、0%。日志未见`[Error]`或Traceback。第一轮v1虽实际完成动作，但结果写在`app.close()`之后而未落盘，相机摘要还误读为`colors`；已修正并用v2重新验收，不把v1计为通过。

此检查只覆盖布局0、一次假策略动作，不能代表完整600步评测、任务成功率、ACT权重、真实模型归一化或示范时间对齐。v2后脚本另加显式三路相机形状/类型和动作字段断言；这些新断言基于v2两端记录核对，尚未单独重跑。小型记录见`experiments/pusht/eval_link_20261107/summary.json`。
