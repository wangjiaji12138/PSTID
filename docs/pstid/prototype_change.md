# 当前状态调制的未来原型变化

上一版只调整未来原型 logits，CQ/seed=88 的 60 轮验证仅改善约 0.06%，不足以默认替换有效基线。本实验转向直接建模预测变化量。

保留历史主干、共享时间嵌入、全部未来时间步以及现有未来预测修正。新增：

```
history = 原历史窗口隐藏特征 + 历史末步原型增强
context = history 或 Encoder(history)           # raw / encoded
q = tanh(Linear(context))                        # B, 1, N, rank
change_h = GELU(Linear(proto_future_h - proto_current))
extra_correction_h = Linear(concat(change_h, change_h * q))
prediction_h = 原版预测_h + extra_correction_h
```

- 差值描述未来时刻相对当前时刻的原型变化，而不是未来的绝对身份。
- 共同的空间原型在差值中抵消；节点身份及整个历史窗口状态仍通过 q 调制修正。
- rank 最多 16；默认训练维度新增 3120 个参数。所有未来步批量计算，历史 query 只算一次，不增加逐步 Python 循环。
- 最后一个线性层零初始化，新增层不消耗基础层的初始化随机流；同种子初始化时与原版预测一致。
- 初始第一步训练先更新新 readout，其后的梯度进入 query/delta 投影。

开关：`--use_proto_change 1`；默认 0，验证有明确收益后才考虑默认替换。`--proto_change_context raw` 使用主干 MLP 前状态，`encoded` 使用 MLP 后状态。`--use_time_interaction` 是之前的 logits 实验，本实验对照关闭它。

验证集对照，不读取测试集指标：

```bash
python -m experiments.compare_pstid_interaction --mechanism change --proto_change_context raw --gpu 1 --epochs 60 --seed 88 --output results/pstid_change_60_seed88.json
python -m experiments.compare_pstid_interaction --mechanism change --proto_change_context raw --gpu 3 --epochs 60 --seed 42 --output results/pstid_change_60_seed42.json
```

比较项是同种子、相同初始化和训练日程下的最佳验证 MAE，同时报告耗时，不能只选择某一个较好的 epoch 或种子宣称稳定改善。

## raw 状态对照结果

CQ、60 轮、相同学习率日程，最佳验证 MAE：

| 种子 | 独立原型基线 | 状态调制的原型差值 | 相对改善 |
| --- | --- | --- | --- |
| 88 | 0.5820817 | 0.5789289 | 0.542% |
| 42 | 0.5838979 | 0.5807068 | 0.547% |

每轮耗时中位数增加约 14.5%–16.0%。这是两个种子的验证集结果，未使用测试集，仍不能视为明显突破。完整记录见 `prototype_change_seed88.json`、`prototype_change_seed42.json`。

## encoded 状态对照结果

进一步将 query 的输入换为主干 MLP 提炼后的状态；其他参数数量和训练日程不变。

| 种子 | 独立原型基线 | encoded 差值修正 | 相对改善 | 每轮中位耗时（基线 / 新版） |
| --- | --- | --- | --- | --- |
| 88 | 0.5820817 | 0.5785546 | 0.606% | 1.2674 / 1.3911 s |
| 42 | 0.5838979 | 0.5801087 | 0.649% | 1.1657 / 1.2574 s |

两种子均优于 raw，但收益仍有限，不宣称大幅提升或测试集收益。训练进程并行运行于不同 GPU，耗时仅作为本次运行开销参考。默认仍关闭新分支；如试用本轮较优方案：

```bash
python train.py --model PSTID --data cq --use_time_interaction 0 --use_proto_change 1 --proto_change_context encoded
```

完整验证轨迹见 `prototype_change_encoded_seed88.json`、`prototype_change_encoded_seed42.json`。本轮共比较了两个设计（raw、encoded），不是在独立测试集上确认的泛化收益。

## 当前默认配置

按用户要求，时间交互 `use_time_interaction=1` 和原型差值修正 `use_proto_change=1` 均默认开启，`proto_change_context=encoded`。命令行、直接构造模型和 `from_args` 缺省值一致；上文默认关闭的说明为实验阶段策略，已由此配置替代。需要复现旧版或做消融时显式传入对应开关为 0。上述实验结果为分别启用各分支的对照，不代表两者同时开启的实测结果。
