# ACT 20次更新诊断

代码来源`6538def`，两条官方示范固定数据revision；具体参数、源码和SHA256保存在`outputs/pusht/act_smoke_20261119_v1/`，小型摘要见[summary.json](summary.json)。episode0训练/统计，episode1仅诊断验证，二者仍属于正式训练池。

执行前实时核对空闲GPU，将完整UUID填入占位；使用新的输出目录，脚本再次检查显存低于10MiB且利用率0：

```bash
python scripts/pusht/train/train_act_smoke.py \
  --gpu-uuid <GPU_UUID> --output outputs/pusht/act_smoke_new \
  --updates 20 --val-every 10
```

本次20次更新无缺失，评估位于0/10/20，模型与统计量重载通过；退出码0。诊断验证总损失下降主要来自KL，L1最终略升；不能推广为正式验证成绩或闭环成功率。Pytorch峰值1782MiB，脚本计时10.85秒，GPU退出后1MiB/0%。模型、完整日志和数据保持Git忽略边界，未上传或发布。
