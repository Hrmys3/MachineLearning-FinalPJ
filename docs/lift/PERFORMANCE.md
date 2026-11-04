# 训练与闭环评测性能对照

所有命令从仓库根目录执行，先使用已有CUDA项目环境。不升级依赖，不覆盖已有模型、实验目录或测速目录。测速时不要同时运行训练、正式评测或另一组基准。

## 先做什么

先测相同任务的基线，再逐项开启优化。模型、种子、示范预算、batch、更新次数、初态编号、预测/执行长度、20Hz、100步和成功标准保持一致。降低分辨率、缩短评测、改变物理参数不是本工具的工程提速方案。

自动对照入口为 `scripts/lift/tools/benchmark_speed.py`。每种方案每次重复有独立YAML、日志和输出目录；顺序交替，不同时运行不同方案。默认共同设置PyTorch CPU线程为1，可通过 `--torch-threads` 修改；这不是对原脚本默认线程数的直接速度比较。文件缓存、GPU温度和其他程序仍可能影响结果，至少重复3次看中位数。

### 训练：缓存与数据预取

```bash
ACT_PRETRAINED=0 python scripts/lift/tools/benchmark_speed.py \
  --kind train --variants baseline,cache,workers \
  --updates 200 --repeats 3 --output outputs/lift/speed_train_v1
```

`ACT_PRETRAINED=0` 让所有方案使用相同的无ImageNet初始化，适合流程测速；要研究正式模型效果，另用相同预训练设置和完整训练预算。工具每次仍验证完整验证集并保存checkpoint，总时间包含这些开销。每个训练输出会占用一份模型资源。

- `baseline`：无缓存、worker0、FP32、无编译。
- `cache`：预处理图像缓存到内存，其余保持基线；同时缓存训练与验证数据，缓存构建成本计入总时间。
- `workers`：2个CPU数据worker、固定页内存、异步传输、保留worker；这是数据供应组合，不应将收益单独归因于某个开关。
- `amp`：仅切换FP16混合精度。
- `fast`：预处理缓存＋FP16。
- `compile`：只启用torch.compile，编译成本计入总时间。

先验证缓存，再另测混合精度：

```bash
ACT_PRETRAINED=0 python scripts/lift/tools/benchmark_speed.py \
  --kind train --variants baseline,amp --updates 200 --repeats 3 \
  --output outputs/lift/speed_train_amp_v1
```

AMP使用autocast和GradScaler，KL指数及损失归约保留FP32。FP16溢出时GradScaler可能跳过更新，实际成功/跳过次数在performance.json中；不能把少做了更新当作等价训练加速。检查验证L1、loss曲线，再以同一闭环初态集验证训练结果，不根据短训练直接宣布效果不变。BF16可在配置中选择，但须硬件支持并另行验证。

### 评测：模型复用与并行

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/lift/tools/benchmark_speed.py \
  --kind eval --policy act_hold --variants baseline,reuse,inference,parallel \
  --episodes 10 --repeats 3 --no-video --output outputs/lift/speed_eval_v1
```

所有方案共同关闭视频，仅排除保存/编码，仍逐步生成相同的双相机观测、执行全部100步。要测包含动画的完整流程，去掉 `--no-video`，使用新的输出目录。

- `baseline`：每轮新进程、新模型、新环境，FP32。
- `reuse`：一个worker依次处理最多5轮，复用模型，每轮仍创建并关闭环境，然后重启worker。初态恢复与规则保持目标每轮重新初始化。
- `inference`：仅将no_grad切换为inference_mode。
- `parallel`：同时2个worker，每轮仍加载模型。提高批量吞吐不等于降低单轮延迟，可能争抢GPU或内存。
- `fast`：2个worker、每worker最多5轮、inference_mode，仍为FP32。
- `amp`：仅切换FP16，必须检查逐步动作和成功率。
- `compile`：模型编译并复用最多5轮，编译耗时计入总时间；不是仅改变一个开关的对照，比较时同时参照reuse。
- `no_video`：仅关闭保存视频与自动总览；属于输出开销对照，不是模型前向加速。

`--episodes` 只选择原初态集的前N个用于测速，不重建或缩小冻结集文件。排查阶段可先用2—5个；正式验证用 `--episodes 100`。组内发生Python异常会记录并继续，底层崩溃时父进程保留已完成轮次，未完成轮次仍计入分母，并返回失败。多轮worker日志为worker_XXXX.log；每轮独立进程时仍为episode_XXXX.log。

## 如何判断加速成立

```bash
python -m json.tool outputs/lift/speed_eval_v1/comparison.json
```

重点看：

1. `all_runs_completed` 必须为真；异常或缺少性能结果时不生成可宣传的速度比。
2. `wall_speedup` 是基线总时间中位数除以该方案总时间中位数。大于1更快，小于1更慢；包含启动、缓存构建、验证和输出成本。
3. 训练看 `steady_samples_per_second`（排除配置的预热更新），也看包含初始化和验证的总时间。预热更新仍正常训练；更新数不足预热要求时稳定速度为null。
4. 评测看 `quality_by_repeat`：初态集合相同、配对成功指标变化列表、动作最大绝对差及方块高度最大差。FP32工程优化优先要求动作/轨迹一致或差异可解释；仅成功率相同不足以证明行为等价。
5. AMP/编译改变浮点计算后可能放大为轨迹差异。需要完整100初态、多训练种子等进一步验证，不能只看小样本速度。

`runs.json` 保存每次命令、返回码、外层实际耗时及性能结果。训练验证L1会受到浮点非确定性影响，不要求不同GPU训练逐位相等；缓存输入本身通过逐元素相等测试。编译接口已接入但属于实验选项，未承诺本机能提速，首次编译及算子兼容性须另测。

## 阶段计时与硬件采样

配置 `performance`：

| 选项 | 作用 |
| --- | --- |
| timing_enabled | CPU墙钟阶段统计，默认开启 |
| cuda_events | 额外CUDA事件统计，默认关闭，最后统一同步读取 |
| hardware_monitor_enabled | 后台采样CPU/内存/GPU，默认关闭 |
| hardware_sample_seconds | 采样周期，默认1秒 |
| warmup_updates | 稳定训练吞吐排除的前若干更新，默认20 |
| profiler_enabled / profiler_active_steps | 训练短窗口PyTorch Profiler，默认关闭 |
| torch_threads | CPU计算线程，默认null沿用环境 |

给自动对照加 `--diagnostics` 可在全部方案共同开启硬件采样和CUDA事件。采样与Profiler也有开销，定位后关闭Profiler，再以相同计时设置做公平对照。

训练输出performance.json包含数据准备、数据等待、传输、前向损失、反向更新、验证和保存时间，以及稳定/总体吞吐、实际更新次数、显存与进程内存峰值。评测每轮含模型加载、环境建立、ACT前向、env.step、GIF编码和总耗时；批量根目录performance.json含整批时间、worker时间、总览编码和完成episode吞吐。

GPU阶段的wall时间可能只是异步提交或包含等待，不能当成纯GPU计算时间；详细设备时间在cuda_stages。阶段之间可能重叠，不能直接相加当总时间。env_step_including_render包含物理、控制、相机观测，尚未进一步拆开。CUDA事件统计不测OpenGL渲染。显存峰值是PyTorch分配，不含OpenGL；进程内存峰值也不含其他worker。

hardware.csv记录系统整体及各CPU核心忙碌比例、可用内存、已用swap，以及各GPU利用率、显存、温度和功耗。CPU整体低不能排除单核瓶颈；查看逐核列。已用swap不等于正在频繁换页，必要时另看vmstat。WSL不支持的GPU指标可能为N/A，工具错误保存在hardware.errors.json，不将缺少硬件监控支持当作模型失败。

也可单独采样：

```bash
python scripts/common/tools/monitor_hardware.py --output outputs/lift/hardware_probe_v1.csv --seconds 60
```

不写入GPU频率、功率或风扇设置；不新增外部Python依赖。硬件采样自动接入训练及批量评测，单轮入口需单独运行监控命令。

## 直接启用配置

训练可选择 `train.cache_mode: raw_ram/processed_ram`，或设置num_workers/pin_memory/persistent_workers/prefetch_factor/non_blocking_transfer；当前禁止RAM缓存与多worker同时开启，避免内存复制。推理可设置eval.episodes_per_worker、num_workers、inference_mode、precision、compile_model。

video.save_rollout关闭动画保存；frame_stride只减少保存帧且保留原动画时长，不改变模型观测；auto_make_grids控制自动总览。CSV、result.json和失败状态始终保存。未实现环境复用、减少相机观测频率或异步视频编码，不应在配置中假设它们已可用。

## 已保存的定量对照

10个固定开发初态、每方案3次重复且完整保存GIF的实际分析见 [推理速度对照](../../experiments/lift/speed_eval10_v1/REPORT.md)。其中包含耗时中位数/范围、配对轨迹核对、初始化与并行开销的解释；不要与此前2初态smoke或关闭视频的测速混合比较。
