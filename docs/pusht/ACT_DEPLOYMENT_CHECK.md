# ACT训练端与部署端离线数值对照

## 运行方法与范围

入口为`scripts/pusht/checks/check_act_deployment.py`。在满足Push-T模型环境要求、已注册mlpj且有对应训练产物的环境中，从仓库根目录运行；每次使用新的输出目录和实时空闲GPU：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
python scripts/pusht/checks/check_act_deployment.py \
  --checkpoint-dir outputs/pusht/act_smoke_20261119_v1 \
  --gpu-uuid <GPU_UUID> \
  --output outputs/pusht/checks/act_deployment_new
```

默认episode0的12、179、358帧，时间集成关闭，每帧重置适配器计数；对应直接预测50步动作块的第一项。路径A使用项目Dataset、ACTPolicy及训练统计量；路径B独立从HDF5构建singular字段/color图像的运行时观测，调用固定XPolicyLab的真实`policy.ACT.model.Model`及其update_obs/get_action。两者加载同一policy_last.ckpt。

脚本验证checkpoint SHA256、统计量重算、数据SHA256、训练源码SHA256及XPolicyLab版本；比较模型输入、完整(1,50,14)预测块、反归一化和左6/夹爪1/右6/夹爪1拆包。保存protocol.json、源码、逐帧NPZ、result.json和GPU前置快照。当前入口要求当前数据组件源码与该次训练冻结值相符；有意修改预处理后应先另建训练/验证协议，不能略过校验继续引用旧结果。

严格对照在检查进程内显式设置cudnn.allow_tf32=False和cuda.matmul.allow_tf32=False，使用rtol=1e-5、atol=1e-6；这不是对所有其他进程或正式训练环境的全局修改。每次GPU运行前内部再查UUID、显存低于10MiB且利用率为0%。

## 2026-11-19实测

代码基础为6538def，新增检查入口当时未提交；模型来自20次更新诊断，last SHA256为`a00e0b85b1c054f5ac7bdc254263e754503d0a04bfea84cfde32eddb1b0cb726`，统计量SHA256为`7332e72e3c8e3bbca408a276940adfac7928dbe1d4d84a2ed3ec16e8032f007b`。最终证据`outputs/pusht/checks/act_deployment_20261119_v3/`，小型记录见[实验摘要](../../experiments/pusht/act_deployment_20261119/summary.json)。

| 对照项 | 三帧中的最大绝对差 |
| --- | ---: |
| 模型图像输入 | 0，逐值相等 |
| 标准化状态输入 | 0，逐值相等 |
| 完整50步标准化动作块 | 7.1525574e-7 |
| 反归一化后的首个14维动作 | 1.1920929e-7 |
| 相同内存布局重算的动作块 | 0 |

模型权重逐值一致，三帧均通过原严格阈值，ok:true、退出码0。PyTorch峰值962.80MiB，脚本报告5.46秒；GPU3运行前1MiB/0%，结束后1MiB/0%。没有启动Isaac、网络服务或完整策略回合。

## 保留的失败与精度边界

首轮v1在默认计算设置下超出严格阈值，退出码1，未计为通过。v2补充完整三帧、权重和输入逐值对照、实际图像strides与同布局重算：输入和权重均相同，项目张量strides为(2764800,921600,307200,640,1)，上游原生布局为(2764800,921600,1,1920,3)；同布局重算差为0。默认设置下完整预测块最大差约2.2519e-4，首动作最大差7.7486e-5，v2仍按原严格阈值失败、退出码1。随后只在检查进程禁用TF32，使用原阈值重跑v3通过；没有通过放宽阈值抹去失败。

因此结论是：预处理、权重、反归一化与拆包一致；显式FP32下两条原生路径通过严格比较。未显式禁用TF32的设置仍有已测量的布局相关浮点差异，后续实际模型服务需要冻结同样精度约定后再验收；不能把v3结果描述成默认设置逐值相同。

## 模型输出与后续工作

v3在t=179的右夹爪原始输出为-0.0009475872，超出0到1；其余两个被检查帧的左右夹爪均在范围内。检查保留原始预测，不对输出裁剪，也不把该越界算成拆包错误。EvalEnv原生动作路径的夹爪裁剪须在后续真实模型联动时确认并记录，不能据本次检查声称已执行该命令或证明策略有效。

该模型只训练20次更新，本检查不是准确率、推动成功率或泛化评测；未验证时间集成、多次连续get_action的整块消费、真实WebSocket服务进程、物理动作、初态恢复或完整闭环。下一步先准备真实模型服务的相同精度/动作配置，再在明确授权后验收WebSocket到EvalEnv的单动作和短闭环；正式train20/50与保留集尚未运行。
