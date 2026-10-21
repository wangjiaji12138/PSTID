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

`ProtoModule` 汇总已启用分支的增强特征，仅在 `use_proto=1` 时叠加 Query 残差，再投影到 `hidden_dim`，与主干嵌入相加。模块拆分没有改变参数注册顺序、张量布局或 state_dict 键名。

## 消融和训练接口

| 参数 | 作用 |
| --- | --- |
| `use_proto` | 1：启用原型模块；0：去掉整个原型模块，embeddings 直接输入 MLP |
| `use_spatio` | 是否启用空间码本 |
| `use_temporal` | 是否启用时间基 |
| `use_adj` | 是否使用图嵌入；关闭后空间码本使用节点嵌入 |
| `use_his` | 原型输入是否包含历史序列嵌入，默认开启 |

这些参数与训练 CLI 的同名开关对应。完整模型及未改变含义的单项消融保留原有数值回归基准。

`--use_proto 0`（w/o P）不创建 `ProtoModule`，不执行空间/时间码本、Query 投影或融合输出投影。时间序列、TID、DIW、节点及可选的邻接嵌入拼接后，直接输入 MLP encoder 和回归层。`use_adj` 仍控制邻接嵌入是否参与拼接；`use_spatio`、`use_temporal`、`use_his` 在此模式下不影响前向计算。

训练配置及检查点保存 `proto_ablation="direct_embeddings_mlp"` 标记，汇总显示为 `w/o P`。汇总仅纳入带此标记的 `use_proto=0` 实验，避免混用旧版 Linear→Softmax→Linear 等实现的指标；旧实验需要重新训练。`use_proto=1` 的参数结构与计算保持不变。

`--use_his 0`（Python 中为 `use_his=False`）时，时间原型的分配器与 Query 投影只输入 `tid_emb + diw_emb`；空间原型只输入 `adj_mx_emb`，若同时设置 `--use_adj 0` 则使用 `node_emb`。相应输入层维度随之缩小，主干仍使用历史序列嵌入。批量消融和结果汇总将此配置标记为 `-His`。旧配置未指定 `use_his` 时默认开启，保留原有模型行为。

训练调用 `model({'X': X})`，只返回预测张量。原型融合直接调用 `ProtoModule.forward`，空间和时间组件只返回融合所需的增强特征与 Query 残差。模型不采集注意力统计、时间分组或嵌入缓存。

扩展原型算法时优先修改对应码本组件；修改参数映射时只改 `PROPHET.from_args`。若改变已注册参数名，应同时考虑旧 checkpoint 的加载和迁移。
