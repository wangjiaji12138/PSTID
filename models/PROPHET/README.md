# PROPHET 模型组件

PROPHET 已同步当前 PSTID 的未来时间原型、时间交互和原型变化残差实现，保留独立的模型类及注册入口。

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
                       回归 Conv2d + 未来原型读出 + 原型变化残差
                           │
                   (B, T_out, N, 1)
```

空间码本将当前批次的时间序列均值与图嵌入（或节点嵌入）拼接并投影。归一化 Query 与原型的余弦相似度经温度 softmax 后，对全部原型加权求和；同一节点的空间分配在当前批次内共享。

时间基将窗口历史表示分别与历史末步、每个未来步的 TID/DIW 嵌入拼接，共享 MLP 分配器与原型表。未来日程仅从历史末步和每日有效时间槽推算，不读取预测标签。时间交互分支将历史与未来的日历关系、历史原型转移偏好加入未来分配 logits。

`ProtoModule` 汇总已启用分支的增强特征，叠加 Query 残差，再投影到 `hidden_dim`。历史增强与窗口嵌入相加后进入单个时间位置的 MLP encoder 和多步回归头，避免重复计算历史窗口表示。未来各步增强经 `future_readout` 修正对应预测；`proto_change_readout` 再结合当前状态和未来减历史的原型差进行非线性修正。两个修正读出均零初始化。

## 消融和训练接口

| 参数 | 作用 |
| --- | --- |
| `use_proto` | 1：启用原型模块；0：去掉整个原型模块，embeddings 直接输入 MLP |
| `use_spatio` | 是否启用空间码本 |
| `use_temporal` | 是否启用时间基 |
| `use_adj` | 是否使用图嵌入；关闭后空间码本使用节点嵌入 |
| `use_his` | 原型输入是否包含历史序列嵌入，默认开启 |
| `use_time_interaction` | 历史与未来时间交互及原型转移偏好，默认开启 |
| `use_proto_change` | 状态调制的原型变化残差，默认开启 |
| `proto_change_context` | 残差使用 `encoded`（默认）或 `raw` 窗口状态 |

这些参数与训练 CLI 的同名开关对应。关闭 `use_proto` 或 `use_temporal` 时，不创建未来读出及原型变化残差分支。

`--use_proto 0`（w/o P）不创建 `ProtoModule`，不执行空间/时间码本、Query 投影或融合输出投影。时间序列、TID、DIW、节点及可选的邻接嵌入拼接后，直接输入 MLP encoder 和回归层。`use_adj` 仍控制邻接嵌入是否参与拼接；`use_spatio`、`use_temporal`、`use_his` 在此模式下不影响前向计算。

训练配置及检查点保存 `proto_ablation="direct_embeddings_mlp"` 标记，汇总显示为 `w/o P`。汇总仅纳入带此标记的 `use_proto=0` 实验，避免混用旧版 Linear→Softmax→Linear 等实现的指标；旧实验需要重新训练。

`--use_his 0`（Python 中为 `use_his=False`）时，时间原型的分配器与 Query 投影只输入 `tid_emb + diw_emb`；空间原型只输入 `adj_mx_emb`，若同时设置 `--use_adj 0` 则使用 `node_emb`。相应输入层维度随之缩小，主干仍使用历史序列嵌入。批量消融和结果汇总将此配置标记为 `-His`。旧配置未指定 `use_his` 时默认开启，保留原有模型行为。

训练调用 `model({'X': X})`，只返回预测张量。原型融合直接调用 `ProtoModule.forward`，空间和时间组件只返回融合所需的增强特征与 Query 残差。模型不采集注意力统计、时间分组或嵌入缓存。

扩展原型算法时优先修改对应码本组件；修改参数映射时只改 `PROPHET.from_args`。若改变已注册参数名，应同时考虑旧 checkpoint 的加载和迁移。

## 运行与检查点

```bash
python train.py --model PROPHET --data nyc_taxi --gpu 2
```

训练入口从训练集推断每日有效时间槽并保存为 `prophet_time_slots`，用于支持非全天采样及跨日推算。直接调用构造函数时可传入 `time_slots`；`from_args` 优先读取 `prophet_time_slots`，也接受已有 PSTID 配置中的 `pstid_time_slots`，均未提供时按全天处理。

新实验记录 `model_variant=prophet_future_change_v1`；关闭变化残差后，按时间交互开关分别记录 `prophet_future_interaction_v1` 或 `prophet_future_independent_v1`。

相同配置下，当前 PSTID 和 PROPHET 的 `state_dict` 键名及形状一致，可严格加载当前 PSTID 权重。旧版 PROPHET 的完整模型检查点缺少新增分支参数，不能直接严格加载到新结构；复现旧实验需使用原版本代码。即使关闭两个新增开关，逐未来步的原型读出仍然存在，并不恢复旧版架构。
