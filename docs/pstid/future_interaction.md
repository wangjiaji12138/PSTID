# 历史末步与未来时间的原型交互

历史流量仍由整个窗口编码，历史时间身份只取末步。未来 H 个时间步全部保留。

令 c0 为历史末步的 tid/diw 拼接嵌入，ch 为未来第 h 步的时间嵌入，l0、lh 为原分配器 logits，a0 = softmax(l0 / temperature)。新增未来 logits：

```
relation_h = concat(c0, ch - c0, c0 * ch)
adjustment_h = Linear(relation_h) + a0 @ prototype_transition
future_weights_h = softmax((lh + adjustment_h) / temperature)
```

`prototype_transition` 是 K×K 的可学习 logits 偏置矩阵，不是归一化的转移概率矩阵。当前原型分布同时包含历史流量信息（use_his=1 时），因此未来分配可以依赖当前交通状态和目标日历时间。历史分配、主干回归头及未来修正输出保持原结构。

新增两组权重均零初始化。加载上一版共同参数时，初始预测函数保持一致；交互模块初始化不改变其他层的随机流。默认训练配置 C=32、K=8 时增加 832 个参数。新增分配计算按未来步批量执行，没有逐步 Python 循环。

开关：`--use_time_interaction 1`（实验性交互）或 `--use_time_interaction 0`（默认，上一版独立分配）。`from_args` 对不含此字段的旧配置默认关闭交互，避免重建旧模型时新增参数。若主动将旧权重加载到启用交互的模型，应检查 `strict=False` 的 missing_keys 仅为 `calendar_interaction.weight` 和 `prototype_transition`；不要忽略其他不匹配。直接续用旧优化器状态不适用于新增参数。

验证集对照（不使用测试集，不覆盖已有训练产物）：

```bash
python -m experiments.compare_pstid_interaction --gpu 1 --epochs 20 --seed 88
```

脚本使用相同种子、相同基础参数初始化、相同训练/验证划分与训练配置。报告默认写入 `results/pstid_interaction_comparison.json`。单次短程实验只能提供方向性证据，完整训练和多种子结果才可判断稳定收益。

## 首轮实测结论

CQ，seed=88，60 轮，使用学习率衰减后的最佳验证 MAE：

| 模型 | 最佳验证 MAE | 每轮耗时中位数（排除首轮） |
| --- | --- | --- |
| 独立分配 | 0.5820817 | 1.1081 s |
| 交互分配 | 0.5817334 | 1.2031 s |

相对 MAE 改善仅 0.0598%，耗时增加约 8.6%。这是单种子、单数据集实验，不能证明稳定收益，也未达到明显提升的目标。因此默认关闭交互，保留实验开关供后续改进。没有使用测试集选型。完整轨迹见 `interaction_validation.json`。
