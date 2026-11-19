# Push-T 示范与 ACT 数据适配

## 数据来源与下载

官方仓库为 [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)，本项目固定数据revision `1127bda94f172d757386945af3eb3a6754d1a4e6`。`data/RoboDojo/push_T/arx_x5/data/`有100条HDF5，元数据查询总大小为27,904,738,093字节；这不是本机已下载量。原始数据、图像与缓存不进入普通Git。

在满足huggingface_hub要求的Python环境中，从仓库根目录按需下载：

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
for episode_id in [0, 1]:
    print(hf_hub_download(
        repo_id="RoboDojo-Benchmark/RoboDojo", repo_type="dataset",
        revision="1127bda94f172d757386945af3eb3a6754d1a4e6",
        filename=f"data/RoboDojo/push_T/arx_x5/data/episode_{episode_id:07d}.hdf5",
        local_dir="data/pusht/robodojo_demos/1127bda94f17",
    ))
PY
```

本机已有的最小仿真资产在`data/pusht/robodojo_assets/1127bda94f17/Assets/`，包含269个官方源文件和2个本机生成规划配置；55个评测布局JSON是初态配置，不是训练示范。

## 已完成检查及证据范围

以下为2026-11-07用户按指导执行并反馈的结果；文档更新时只复核源码和必要的CPU检查，不把终端反馈写成新的GPU实验。

| episode | 字节数 | 时间点数 | SHA256 |
| --- | ---: | ---: | --- |
| 0 | 296260537 | 359 | `bd4907829ffd6f26c16f099233224282e3f52284023ca7ec11af41a46d623966` |
| 1 | 432310016 | 512 | `3984f2223c4ac57946dd9cbf38c8457ebd9c5f80267583c9b406d1e410bd2a3f` |

两条文件大小和官方SHA256相符。第0条另确认HDF5格式、data_format_version=v1.0，以及指令“Push the T-shaped block to align it precisely with the gray T-shaped pad.”；两条frequency均为25Hz，joint状态和动作分别为(T,14)，全部有限。第0条还逐项检查了state/action全部关节及位姿字段的有限性；第1条尚未同样检查全部位姿字段。

两条joint数据在所有可比较位置均满足`action[:-1] == state[1:]`，逐值最大差为0。第0/1条末动作与末状态最大差分别约1.8672e-5、1.0431e-5；末动作缺少文件内下一帧可供比较，不能据变化小认定为填充。当前组件保留末动作，截断动作块之后的位置才标记为padding。这个数值关系不证明动作等于原始控制命令，也不独立证明图像与状态的物理时间完全同步；示范物理回放待做。

第0条三路相机在0、179、358帧均经官方`decode_image_bit`解码为480×640×3 uint8；用户生成`outputs/pusht/checks/demo0_contact_sheet.png`并反馈图片正常。第1条三路首末帧解码正常且有变化，尚无该条图像目视反馈。未检查两条示范的每一帧，也未核验其余98条示范内容。

## 固定划分

唯一配置为[configs/pusht/splits_v1.json](../../configs/pusht/splits_v1.json)。固定种子20261107，0、1号用于接口调试，显式放入训练池；其余2—99号打乱后取58条补足训练池，随后20条验证、20条保留测试。`train20 == train50[:20]`、`train50 == train60[:50]`；train60、valid20、heldout20互斥并覆盖0—99。已检查过的0、1号不得改称独立测试数据。

当前只下载0、1号，尚不能计算正式train20或train50统计量。未来训练入口应读取显式清单，按实际预算计算统计量；不能把全部训练池、验证或测试帧混入较小预算。保留集在冻结模型与协议后使用，正式划分不可随模型成败调整。

## 上游 ACT 审计

固定XPolicyLab为`5d0be1499634d3176258c8f2e31f5fe816497b0a`。原`policy/ACT/utils.py`在`__getitem__`内设`is_sim=None`，因此从`action[max(0,t-1):]`开始取块；对于已检查数据，t>0时首标签等于当前state[t]。用户在第0条t=12、第1条t=125实测旧首标签与当前状态差为0，同索引动作与当前状态最大差约0.13201344和0.15263975。

原`load_data`随机80/20划分后，仍以全部num_episodes计算统计量；`get_norm_stats`将短轨迹末帧重复到最长轨迹再计算均值和样本标准差。这会混入验证数据并改变末态权重。原转换器还按连续编号0到num_episodes-1读取，不匹配本项目非连续的预算清单。上游源码保留，项目组件尚未接入原train.sh/imitate_episodes.py，不能直接用其默认训练入口宣称遵循本项目协议。

## 项目内的数据契约

实现为[src/mlpj/pusht/act_data.py](../../src/mlpj/pusht/act_data.py)和[act_dataset.py](../../src/mlpj/pusht/act_dataset.py)。

- 14维顺序：左臂6、左夹爪1、右臂6、右夹爪1；夹爪使用文件内归一化值。与Lift的9维末端状态和7维增量动作无关。
- 图像顺序：`cam_head`、`cam_right_wrist`、`cam_left_wrist`；使用固定XPolicyLab官方解码，不额外换色。每次只解码一个时间点三路图像，返回float32 RGB、[0,1]、(3,3,480,640)。ACTPolicy内部仍会进行ImageNet标准化。
- 输入图像/state[t]与action[t:]配对，不再额外移位。动作块长度由horizon指定，检查使用50；没有确定正式每次执行长度或完整评测协议。
- `compute_norm_stats(episodes, train_ids)`仅拼接指定编号的真实帧，以float64计算，样本标准差ddof=1，下限1e-2，输出float32的qpos/action mean/std。各真实帧等权，不做末帧重复填充。
- `slice_action_chunk`返回float32动作和布尔is_pad；True为填充。组合样本归一化后将填充动作置零，必须在损失中使用掩码。
- `PushTACTDataset`按传入episode列表定义长度；训练采样每次在所选episode中随机取时间点。`get_at(id,t)`用于确定性检查；`FixedPushTACTDataset`按显式(episode_id,timestep)列表取样。短训练诊断固定验证帧，并在fork_rng内重设随机种子、验证后恢复随机状态。正式验证协议仍需单独冻结。
- 当前函数要求调用方传入统计量和编号，尚未自动加载/校验固定划分或核对所有HDF5校验值；正式入口必须承担这些检查。

用户已完成合成数据上的动作索引/掩码和训练范围统计检查；真实两条轨迹读取、图像、组合样本、反归一化和CPU Dataset/DataLoader检查通过。t=12动作块有50项有效；第0条t=358有1项有效、49项padding。batch=2输出为images(2,3,3,480,640)、qpos(2,14)、actions(2,50,14)、is_pad(2,50)，前三者float32，掩码bool，均在CPU。

从仓库根目录使用图像接口时，先按README注册mlpj，并让固定XPolicyLab源码可导入：

```bash
export PYTHONPATH="third_party/pusht/RoboDojo/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from mlpj.pusht.act_data import load_joint_episodes, compute_norm_stats
from mlpj.pusht.act_dataset import PushTACTDataset
base = "data/pusht/robodojo_demos/1127bda94f17/data/RoboDojo/push_T/arx_x5/data"
episodes = load_joint_episodes(base, [0, 1])
stats = compute_norm_stats(episodes, train_ids=[0, 1])  # 仅两条诊断统计量
sample = PushTACTDataset(base, episodes, [0, 1], stats).get_at(0, 12)
print([(tuple(x.shape), str(x.dtype), str(x.device)) for x in sample])
PY
```

## 接手顺序与任务隔离

用户要求常规推进采用“教用户操作—用户反馈—再给下一步”，未经明确授权不代为执行数据处理、训练或仿真；本次文档更新与Git提交已获明确授权。每次使用GPU前重新检查实时UUID、显存和利用率，历史快照不可复用。

用户已执行`scripts/pusht/checks/check_act_batch.py`，证据`outputs/pusht/checks/act_batch_ukd37nn7/`为ok:true：单样本、三路原尺寸图像、50步动作块，完成一次优化器更新和(1,50,14)推理；参数变化约1e-5，PyTorch峰值1513.5MiB。使用0/1两条诊断统计量，没有保存checkpoint；记录中的config.batch_size=2是未使用的解析器默认值，实际batch size为1。此样本没有padding，因此不受下面损失修正影响。

`src/mlpj/pusht/act_training.py`的`valid_act_losses`将固定上游ACT的masked L1从包含填充位置的分母换算为有效元素平均；必须只转换一次，掩码长度必须匹配模型动作块。合成CPU检查中，50个/1个有效动作的相同误差均得到L1=2，填充梯度为0。

`scripts/pusht/train/train_act_smoke.py`现已创建，默认20次更新、batch size 1、有效元素平均L1、KL权重10、梯度裁剪1、无AMP。仅episode 0参与训练/统计量，episode 1是诊断验证，仍属于正式训练池。验证使用固定帧与fork_rng种子1000；保存parameters.yaml、源码、逐次指标、best/last模型及统计量，并计划检查最后模型重载的推理一致性。best允许在第0次更新被选中；该目录没有用于断点续训的优化器状态。完整GPU流程尚未执行，目前仅语法及相关CPU组件通过，不能称checkpoint重载已验收。运行命令见根README。

下一步实时检查GPU后执行短训练诊断，核对有限损失、更新次数、checkpoint与统计量重载，再准备正式预算数据、训练入口和完整闭环。尚无正式Push-T训练权重、成功率或保留集结果。

Push-T不导入Lift代码；独立维护机器人、25Hz控制、相机、14维动作、环境依赖与实验资源。mlpj包壳、common、根文档和Git历史仍共享。`pusht`分支隔离开发历史，不能隔离本机Python环境、资源文件或GPU，细则见[结构说明](../REPOSITORY_LAYOUT.md)。
