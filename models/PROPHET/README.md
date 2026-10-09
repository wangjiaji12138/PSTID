# PROPHET 模型组件

## 阅读顺序

1. `model.py`：`PROPHET.__init__`、`_get_embeddings`、`forward`、`from_args`。
2. `prototype.py`：`ProtoModule` 如何调用空间和时间组件，并融合特征。
3. `spatial.py` / `temporal.py`：两种原型分配算法。
4. `graph.py` / `layers.py`：图初始化与残差 MLP。

`PROPHET.py` 仅负责旧路径兼容。模型内部使用相对导入，外部推荐：

```python
from models.PROPHET.model import PROPHET
# 通过统一工厂构造实验模型：
from models.registry import build_model
```

## 计算过程

```text
X: (B, T_in, N, 3)
  ├── 历史数值 → Conv2d 时间序列嵌入
  ├── 节点身份嵌入
  ├── 最后历史时刻的 TID / DIW 嵌入
  └── 邻接矩阵谱初始化的可学习图嵌入（use_adj）
                  │
                  ├── 拼接为 hidden
                  └── ProtoModule
                        ├── SpatialCodebook
                        ├── TemporalBasis
                        └── 原型增强 + Query 残差 → 输出投影
                                      │
                  hidden + 原型增强 ──┘
                           │
                       残差 MLP
                           │
                       回归 Conv2d
                           │
                   (B, T_out, N, 1)
```

空间码本将当前批次的时间序列均值与图嵌入（或节点嵌入）拼接并投影。归一化 Query 与原型的余弦相似度经温度 softmax 后，对全部原型加权求和；同一节点的空间分配在当前批次内共享。

时间基将历史表示与最后时刻的 TID/DIW 嵌入拼接。MLP 预测原型分配，另一投影网络产生 Query 特征，原型加权结果再广播到内部时间维度。当前实现并非逐未来时刻单独分配原型。

`ProtoModule` 汇总已启用分支的增强特征与 Query 残差，再投影到 `hidden_dim`，与主干嵌入相加。模块拆分没有改变参数注册顺序、张量布局或 state_dict 键名。

## 消融和训练接口

| 参数 | 作用 |
| --- | --- |
| `use_proto` | 是否启用原型模块 |
| `use_spatio` | 是否启用空间码本 |
| `use_temporal` | 是否启用时间基 |
| `use_adj` | 是否使用图嵌入；关闭后空间码本使用节点嵌入 |

这些参数与训练 CLI 的同名开关对应。原有四种单项消融与完整模型均有数值回归基准。

训练调用 `model({'X': X})`，只返回预测张量。原型融合直接调用 `ProtoModule.forward`，空间和时间组件只返回融合所需的增强特征与 Query 残差。模型不采集注意力统计、时间分组或嵌入缓存。

扩展原型算法时优先修改对应码本组件；修改参数映射时只改 `PROPHET.from_args`。若改变已注册参数名，应同时考虑旧 checkpoint 的加载和迁移。
