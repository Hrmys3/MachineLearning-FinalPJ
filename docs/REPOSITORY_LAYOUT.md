# 仓库结构

本项目采用 `src/mlpj/` Python包与按任务分类的资源目录。根README提供运行方法；先在所选环境中注册源码包，再从仓库根目录执行脚本。

| 目录 | 用途 |
| --- | --- |
| `src/mlpj/common/` | 根目录定位、共享计时与硬件监控实现 |
| `src/mlpj/lift/` | Lift配置、数据、训练、评测、专家与检查实现 |
| `src/mlpj/pusht/` | Push-T资产准备、场景检查和ACT数据组件 |
| `scripts/lift/` | Lift入口，内部按checks/data/train/eval/experts/tools分类 |
| `scripts/pusht/` | Push-T环境、场景及策略通信检查入口 |
| `scripts/common/tools/` | 共享硬件监控入口 |
| `tests/{common,lift,pusht}/` | 自动化测试；相机和真实模型检查仍在scripts |
| `configs/lift/default.yaml` | Lift统一配置；Push-T配置归configs/pusht |
| `data/{lift,pusht}/` | 下载数据、缓存和实际初态；大资源默认忽略 |
| `outputs/{lift,pusht}/` | 模型、冻结参数、日志、CSV和动画；默认忽略 |
| `experiments/{lift,pusht}/` | 固定划分说明、小型清单、指标及分析，纳入Git |
| `third_party/lift/act_lift/` | 已跟踪的ACT适配副本，包含原许可证 |
| `third_party/lift/act/` | 可选上游副本，默认忽略 |
| `third_party/pusht/` | 固定RoboDojo子模块及其XPolicyLab、IsaacLab、CuRobo依赖 |
| `docs/{lift,pusht}/` | 各任务说明；当前性能说明在docs/lift/PERFORMANCE.md |
| `env/{lift,pusht}/` | 各任务环境要求及依赖记录 |
| `archive/{lift,pusht}/` | 历史辅助材料，不是运行必经入口 |

可复用实现位于对应的`mlpj`模块；Lift运行器和Push-T部分环境/通信诊断保留脚本实现。原有文件名继续使用，旧目录不保留兼容入口。共享包和任务包的初始化不加载Isaac或MuJoCo；不要为复用文件名而混用动作、相机和成功判定语义。源码包的根目录定位面向仓库检出与可编辑安装。

Lift批量入口调用 `scripts/lift/eval/eval_act_one.py`，冻结本次 `parameters.yaml`，并捕获入口及实际评测模块、配置和共享工具的源码。保持协议、失败分母和20Hz控制约定保持不变。测试运行：

```bash
python -m unittest discover -s tests -v
```

本次按任务调整现有目录，未重跑历史实验。历史实验JSON中的路径字段及说明直接改为新路径；来源提交、校验值、指标及源码快照原文保留，快照中的路径表示当时实现。代码结构迁移时的验证环境只有小型元信息和实验记录；随后已将当前机器旧目录中的数据、缓存、权重与历史outputs补迁到对应lift目录，并逐文件校验内容一致。其他机器的资源仍需按新目录放置，不能将本机验证扩大为所有机器均已迁移。详细范围见 [目录与资源核查](STRUCTURE_AUDIT.md)。

普通克隆包含代码、适配ACT、Lift划分/统计量和小型实验记录，不包含示范、缓存、权重、完整日志和视频。`.gitkeep`只保留未实现任务或运行资源的目录边界。第三方README保留上游内容，项目运行以根README为准。

## 任务隔离与分支

Lift继续使用自己的MuJoCo/robosuite、适配ACT、配置和数据；Push-T使用独立的RoboDojo/Isaac、XPolicyLab和14维数据接口，不导入Lift任务实现。共享面为mlpj包壳、common、根文档与Git历史。当前Push-T开发在`pusht`分支，创建点为`0fdbaa9f2b5283f8a4239ea1aba74a92741c5c74`；`main`未合入后续Push-T改动。分支隔离代码历史，不隔离同一检出的未跟踪数据、环境或GPU。

Push-T固定划分在`configs/pusht/splits_v1.json`；原始HDF5、资产、缓存和运行产物继续留在忽略目录。RoboDojo与嵌套IsaacLab的本地补丁保存在`env/pusht/`，Git子模块引用本身不包含未提交补丁；按环境文档应用后子模块显示修改属于预期状态。
