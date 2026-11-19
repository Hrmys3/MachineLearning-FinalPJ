# 真实ACT与EvalEnv单动作联通

## 运行入口

策略端`scripts/pusht/checks/serve_act_policy.py`使用固定XPolicyLab真实Model和PolicyServer，读取本机诊断checkpoint与统计量；仿真端`scripts/pusht/checks/check_act_eval_link.py`加载官方push_T/arx_x5布局0，通过原生EvalEnv发送一次观测并执行一次返回动作。保留原假策略入口，互不替代。

两个终端均从仓库根目录执行。终端A使用已满足ACT/WebSocket服务端依赖的Python，终端B使用已满足Isaac/RoboDojo依赖的Python。先实时检查GPU，分别选两张空闲卡；独立脚本会再次守卫UUID、显存低于10MiB及利用率0。输出目录须不存在。

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
# 终端A：只监听127.0.0.1，启动后从port.json读取端口
python scripts/pusht/checks/serve_act_policy.py \
  --checkpoint-dir outputs/pusht/act_smoke_20261119_v1 \
  --gpu-uuid <POLICY_GPU_UUID> --output outputs/pusht/checks/act_link_server_new

# 终端B：使用另一张空闲卡，保留物理CUDA/Vulkan编号，勿设置CUDA_VISIBLE_DEVICES
python scripts/pusht/checks/check_act_eval_link.py \
  --device-id <SIM_GPU_INDEX> --gpu-uuid <SIM_GPU_UUID> \
  --port <PORT_FROM_SERVER_PORT_JSON> --output outputs/pusht/checks/act_link_client_new
```

仿真进程结束后，在终端A按Ctrl+C关闭服务并写出服务结果，再检查两张卡释放。服务仅接受一次动作推理，重复get_action会被拒绝；不支持直接作为正式批量评测服务。运行期间的状态、图像、预测均只写入本机输出目录。

两端均显式禁用cudnn/matmul TF32，策略端关闭时间集成并从首个预测动作开始。仿真保持CPU物理、显式CUDA规划/相机、dt=.004、25Hz、原生夹爪0到1裁剪与行程映射、关节插值。没有额外动作缩放、臂目标裁剪或专家接管。

## 2026-11-19证据

完整产物在`outputs/pusht/checks/act_eval_20261119_v1/`，小型记录见[summary.json](../../experiments/pusht/act_eval_link_20261119/summary.json)。运行代码基础6538def，两个检查脚本当时未提交，实际执行源码、启动命令和超时/收尾监督源码保存在该目录。模型last SHA256为`a00e0b85b1c054f5ac7bdc254263e754503d0a04bfea84cfde32eddb1b0cb726`，统计量仅来自episode0。

- 策略服务复用Python3.10.22、torch2.4.1+cu121、torchvision0.19.1+cu121、websockets16.1.1；无需安装或升级。训练产物来自torch2.7.0+cu126环境，本次实际加载和GPU推理通过，但未做两个PyTorch版本的逐值数值对照。
- 策略使用物理GPU6、仿真使用GPU3，启动前两卡及各进程内部均重新检查空闲。两个进程退出码均0、各自及总审计ok:true，结束两卡均2MiB/0%。没有结束或操作其他用户进程。
- reset、update_obs、get_action各1次。服务端收到的三路480×640×3 uint8图像逐像素SHA256与客户端发送相同，状态字段逐值相同；服务端预测与客户端收到的四个动作字段逐值相同，维度6+1+6+1。
- 实际调用EvalEnv.take_action一次，记录原生控制队列长度10和sim_step调用10次，物理dt=.004，总计0.04秒。左右臂最大实测关节变化分别0.0184140、0.1997251rad；右臂目标最大变化1.8889253rad，单个控制周期没有到达该目标，不能把通信通过写成目标精确跟踪。
- 左/右夹爪原始输出0.9994833、0.00469235，本次均未触发0到1裁剪；米制主关节目标0.04397210、-0.00974661与捕获的原生末控制目标一致。负米制值在本机原生行程范围内，不是归一化夹爪越界。此前离线出现的负归一化预测没有在本次在线帧触发，因此仍未在线覆盖越界裁剪情形。
- T物体位置和姿态保持记录值，位移0；头部前后PNG目视确认右臂移动且场景正常。保存三路执行前/后共6张PNG、真实机器人/物体状态、预测、夹爪映射、原生插值首末项和有效配置。未采接触力，不能据位移0断言没有接触。
- 服务端单次推理计时约0.488秒，客户端get_action往返约0.491秒，包含首请求开销；不是持续推理吞吐或25Hz实时性证据。服务端PyTorch峰值约641.95MiB，不是总显存。日志无[Error]/Traceback，保留已有无显示、材质引用和预训练参数弃用警告。

## 验收边界与下一步

这是20次更新诊断模型在官方布局0的单动作链路检查，不是示范初态回放、600步评测或推动成功率。两次训练示范与布局0不能直接视为匹配初态。下一步应明确初态/动作执行协议，并在授权后增加多次观测—推理—执行的短闭环；关闭时间集成时，上游适配器会缓存动作块并按内部计数取后续动作，不能把每次get_action都称为重新推理。正式train20/50、valid20/heldout20和完整成功率仍待完成。
