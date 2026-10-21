# PSTID v0 独立模型

已恢复为最初从 `models/PROPHET` 复制出的独立实现，仅类名、包名和模型名不同。空间码本、时间原型、Query残差及原始逐步预测头与PROPHET一致，不包含未来轨迹Delta或共享预测头。

```bash
python train.py --model PSTID --data cq --gpu 2
```

默认输出到 `results/cq/PSTID_<时间戳>/`。配置记录 `model_variant=pstid_fork_v0`，现有PROPHET消融汇总会跳过独立PSTID实验，保留对历史PSTID名称的兼容。

```python
from models.PSTID import PSTID
```

v1/v2训练结果仍保留在原目录；其轨迹检查点与v0结构不同，不能直接用v0加载。此次回退没有修改已有实验记录，也没有启动训练。
