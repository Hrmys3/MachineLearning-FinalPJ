# 固定随机开发初态集

配置入口：`configs/lift.yaml`；生成入口：`scripts/data/prepare_random_lift_states.py`。

`manifest.json` 是本次实际生成的清单，记录数量100、主种子20261103、版本、校验值和生成代码。数据文件位于 `data/lift_act/random_dev100_v1.hdf5`，包含固定模型XML、100个完整初态、各初态采样种子、方块与末端位置，不纳入普通Git且未发布。

本集在一个固定几何模型上使用原生reset采样，目的为参数开发对照，不能作为反复调参后的独立最终测试集。无训练示范轨迹被复制为这100个初态；各初态未按策略成功与否筛选。已逐条检查有限性、唯一性、初始不成功，以及重建XML后的状态/方块位置准确恢复。

重新生成需准备源资源和相同版本环境；已有文件不会被覆盖。修改数量或种子时，使用新的配置副本和数据集路径。跨机器重建XML可能包含不同资源路径，重新生成文件的字节校验值可能变化，必须保留当次清单。跨机器和空白环境复现尚未验收。

运行100轮开发评测（输出必须为新目录）：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/run_act_validation.py --policy act_hold --output outputs/act_hold_random100_v1
```

运行旧20个验证示范初态：

```bash
MUJOCO_GL=glfw ACT_PRETRAINED=0 python scripts/eval/run_act_validation.py --initial-states valid --policy act_hold --output outputs/act_hold_valid_v2
```
