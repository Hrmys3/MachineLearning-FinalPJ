# Push-T最小资产与X5规划器检查

## 固定来源与本机验证

代码版本沿用[设备与客户端记录](DEVICE_AND_CLIENT.md)。资产来自Hugging Face数据集`RoboDojo-Benchmark/RoboDojo`，revision固定为`1127bda94f172d757386945af3eb3a6754d1a4e6`，与代码提交号独立。

最小资产放在`data/pusht/robodojo_assets/1127bda94f17/Assets/`；上游`third_party/pusht/RoboDojo/Assets`以相对符号链接指向该目录。下载包括X5机器人、Rigid/t、Geometry/t_cushion与camera_stand、Simple_Room_nolight、material_0122与material_0564、brown_photostudio_02_4k.hdr及arx_x5/0下的push_T布局。共269个源文件、415,490,428字节，不含训练示范或模型。

首次下载记录位于`outputs/pusht/checks/assets_preflight_he7vl08y/`：inventory.json、download-plan.json、manifest.json记录允许列表、revision、大小、官方LFS SHA256及本地SHA256；最终manifest为verified:true。2026-11-07重新读取全部269个本地源文件，大小及SHA256全部匹配。清单不含随后生成的规划配置。资产和完整检查日志不进入普通Git，本说明不代表资源已发布。

## 生成本地X5规划路径

官方资产提供`Robots/x5/curobo_tmp.yml`，代码读取`curobo.yml`。在已注册mlpj、安装PyYAML并准备上述资产的环境中，从仓库根目录执行：

```bash
python scripts/pusht/data/prepare_x5_planner.py \
  --assets-dir data/pusht/robodojo_assets/1127bda94f17/Assets
```

工具将模板中的`${ASSETS_PATH}`替换为所给Assets目录的父目录，检查URDF及资产根目录存在，生成curobo.yml及curobo.generated.json。保留原模板，不改关节、碰撞或控制参数。相同内容可重复执行，不同内容拒绝覆盖；生成文件包含本机解析后的绝对路径，移动仓库后需检查并重新生成，不能盲目复用。

## X5规划器实际检查

历史证据在`outputs/pusht/checks/x5_planner_fam191ud/`。check_planner.py使用官方X5 URDF及生成配置，保持dt=0.004、CUDA Graph和原规划器warmup，完成初始化及默认关节态到自身目标的一次plan_cspace。result.json为ok:true、warmup_completed:true、self_target_plan_success:true；进程退出码0，耗时34.77秒，PyTorch峰值分配2,634,544,640字节（约2.45GiB），不等于总显存。

该次运行通过UUID只暴露当时空闲的物理3号卡，因此规划器逻辑cuda:0指向物理3号卡；退出后显存1MiB。重跑前必须检查实时空闲和UUID，历史检查脚本含本机绝对配置路径，不应原样用于其他检出。

该结果只证明真实X5运动规划器能初始化、warmup并完成上述自目标规划。未在该检查中加载Isaac的X5 USD任务场景，也没有推动物体、示范回放、策略服务、ACT训练或闭环成功率。

## 固定布局加载与三路相机（2026-11-07）

证据在`outputs/pusht/checks/task_scene_kb88rdl4/`，保留check_scene.py、effective-config.yaml、三张PNG、result.json、runtime.log、process-status.json及GPU前后快照。直接加载原生push_T类、官方arx_x5配置和布局0，单环境、dt=0.004、相机25Hz；一次重置、恢复保存位姿和机器人初态后更新200步并捕获三路图像。未连接策略服务器。

头部、左腕、右腕图像均为480×640×4、有限数值且非空；主代理目视看到双臂、红T和灰色目标。CPU物理、cuda:3规划与三路输出缓冲区通过，GPU0无PyTorch张量分配；这不涵盖其他库的所有GPU上下文。运行前物理3号卡1MiB/0%，退出码0，结束后2MiB/0%。日志无[Error]/Traceback，但有上游USD材质引用越界、MDL编译和无显示/音频警告，不能据此宣称材质完全正确。

本检查没有执行完整EvalEnv的布局稳定性流程，也没有比较重复重置或下发关节动作。图中T接近左夹爪，后续必须核对保存布局与实际位姿/碰撞；不能把一次出图当成初态稳定性、动作接口或任务成功验收。

后续同一布局的稳定性、三次重置和四项低层伺服检查已完成，范围、失败记录与限制见 [环境接口检查](TASK_INTERFACE_CHECK.md)。这仍不代表完整EvalEnv策略接口或任务成功率已验收。
