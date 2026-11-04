# 仓库结构

本项目采用 `src/mlpj/` Python包与按任务分类的资源目录。根README提供运行方法；先在所选环境中注册源码包，再从仓库根目录执行脚本。

| 目录 | 用途 |
| --- | --- |
| `src/mlpj/common/` | 根目录定位、共享计时与硬件监控实现 |
| `src/mlpj/lift/` | Lift配置、数据、训练、评测、专家与检查实现 |
| `src/mlpj/pusht/` | Push-T包边界，目前无任务实现 |
| `scripts/lift/` | Lift入口，内部按checks/data/train/eval/experts/tools分类 |
| `scripts/pusht/` | Push-T入口预留 |
| `scripts/common/tools/` | 共享硬件监控入口 |
| `tests/{common,lift,pusht}/` | 自动化测试；相机和真实模型检查仍在scripts |
| `configs/lift/default.yaml` | Lift统一配置；Push-T配置归configs/pusht |
| `data/{lift,pusht}/` | 下载数据、缓存和实际初态；大资源默认忽略 |
| `outputs/{lift,pusht}/` | 模型、冻结参数、日志、CSV和动画；默认忽略 |
| `experiments/{lift,pusht}/` | 固定划分说明、小型清单、指标及分析，纳入Git |
| `third_party/lift/act_lift/` | 已跟踪的ACT适配副本，包含原许可证 |
| `third_party/lift/act/` | 可选上游副本，默认忽略 |
| `third_party/pusht/` | RoboDojo/XPolicyLab依赖预留，尚未下载 |
| `docs/{lift,pusht}/` | 各任务说明；当前性能说明在docs/lift/PERFORMANCE.md |
| `env/{lift,pusht}/` | 各任务环境要求及依赖记录 |
| `archive/{lift,pusht}/` | 历史辅助材料，不是运行必经入口 |

`scripts/`只保留薄入口，实际实现位于对应的`mlpj`模块。原有文件名继续使用，旧目录不保留兼容入口。共享包和任务包的初始化不加载Isaac或MuJoCo；不要为复用文件名而混用动作、相机和成功判定语义。源码包的根目录定位面向仓库检出与可编辑安装。

Lift批量入口调用 `scripts/lift/eval/eval_act_one.py`，冻结本次 `parameters.yaml`，并捕获入口及实际评测模块、配置和共享工具的源码。保持协议、失败分母和20Hz控制约定保持不变。测试运行：

```bash
python -m unittest discover -s tests -v
```

本次按任务调整现有目录，未重跑历史实验。历史实验JSON中的路径字段及说明直接改为新路径；来源提交、校验值、指标及源码快照原文保留，快照中的路径表示当时实现。迁移时本机只有小型数据元信息和实验记录，没有示范缓存、权重及完整outputs。其他机器上的大资源需放入新的任务目录，不能据此声称已迁移或重新验证。

普通克隆包含代码、适配ACT、Lift划分/统计量和小型实验记录，不包含示范、缓存、权重、完整日志和视频。`.gitkeep`只保留未实现任务或运行资源的目录边界。第三方README保留上游内容，项目运行以根README为准。
