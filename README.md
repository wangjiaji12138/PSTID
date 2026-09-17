# PSTID: 原型时空身份网络
     
基于原型学习的时空图预测模型，用于城市出行需求预测等时空预测任务。

## 项目概述

PSTID 项目实现了一套完整的时空图预测框架，支持多种模型架构：

### 基线模型
- **HA**: 历史平均法（最简单的统计基线）
- **LSTM**: 长短期记忆网络（深度学习基线）

### 时空图神经网络
- **PSTID**: 原型时空身份网络（STID + 原型模块）
- **STID**: 时空身份网络
- **STGCN**: 时空图卷积网络
- **AGCRN**: 自适应图卷积循环网络
- **GWNet**: 图小波神经网络
- **MegaCRN**: 记忆增强图卷积循环网络
- **DCRNN**: 扩散卷积循环神经网络
- **MTGNN**: 多时间图神经网络
- **STAEformer**: 时空自编码器Transformer
- **STDN**: 时空动态网络
- **STNorm**: 时空归一化网络
- **STSSL**: 自监督时空网络
- **STSSDL**: 半监督深度学习网络
- **PDFormer**: 流行方向感知的Transformer

## 项目结构

```
PSTID/
├── models/                     # 模型定义
│   ├── HA/                    # 历史平均基线
│   │   └── HA.py
│   ├── LSTM/                  # LSTM 基线
│   │   └── LSTM.py
│   ├── PSTID/                 # PSTID 模型（重点模型）
│   │   └── PSTID.py
│   ├── STID/                  # STID 模型
│   ├── STGCN/                 # STGCN 模型
│   ├── AGCRN/                 # AGCRN 模型
│   ├── GWNET/                 # GWNet 模型
│   ├── MegaCRN/               # MegaCRN 模型
│   ├── DCRNN/                 # DCRNN 模型
│   ├── MTGNN/                 # MTGNN 模型
│   ├── STAEformer/            # STAEformer 模型
│   ├── STDN/                  # STDN 模型
│   ├── STNorm/                # STNorm 模型
│   ├── STSSL/                 # STSSL 模型
│   ├── STSSDL/                # STSSDL 模型
│   ├── PDFormer/              # PDFormer 模型
│   ├── base.py                # 基类模型
│   └── model_builder.py       # 模型构建器
├── utils/                     # 工具函数
│   ├── data_utils.py          # 数据加载工具
│   ├── metrics.py             # 评估指标
│   └── log_record.py          # 日志记录
├── data/                      # 数据目录
│   └── processed/             # 处理后的数据集
├── results/                   # 实验结果
├── train.py                   # 训练脚本
├── predict.py                 # 预测脚本
├── gen_dataset.py             # 数据生成脚本
├── abl.py                     # 消融实验脚本
├── requirements.txt          # 依赖包
└── README.md
```

---

## PSTID 模型架构

### 1. 模型概述

**PSTID (Prototype-based Spatio-Temporal Identity Network)** 是基于 STID 的改进模型，在输入嵌入和 MLP 层之间引入了**原型模块 (ProtoModule)**，通过原型机制增强时空表示，减少噪声并提升模型可解释性。

### 2. 整体架构

```
Input (B, T, N, 3)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    STID Embedding Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ Time Series Emb │  │  Time-in-Day    │  │   Spatial  │ │
│  │ (Conv2d)        │  │  Embedding      │  │  Embedding │ │
│  └────────┬────────┘  └────────┬────────┘  └─────┬──────┘ │
│           │                     │                  │        │
│           └──────────┬─────────┴──────────────────┘      │
│                      │                                     │
│                      ▼                                     │
│              ┌───────────────┐                             │
│              │ Concat & Reshape │                           │
│              └───────┬───────┘                             │
│                      │                                     │
└──────────────────────┼─────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ProtoModule (原型模块)                    │
│  ┌─────────────────────┐     ┌─────────────────────┐      │
│  │  Spatial Codebook    │     │  Temporal Basis     │      │
│  │  (空间码本)          │     │  (时间基)           │      │
│  │                     │     │                     │      │
│  │  • prototypes (N_s)│     │  • prototypes (N_t)│      │
│  │  • spatio_emb_proj │     │  • MLP Allocator   │      │
│  │  • attention        │     │  • attention       │      │
│  └──────────┬──────────┘     └──────────┬──────────┘      │
│             │                            │                  │
│             └───────────┬────────────────┘                  │
│                         ▼                                   │
│              ┌───────────────────┐                          │
│              │   Fusion & Add    │                          │
│              │   proto_enhanced  │                          │
│              └─────────┬─────────┘                          │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       MLP Layers                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MultiLayerPerceptron × num_block                    │   │
│  │  (Conv2d → ReLU → Dropout → Conv2d → Add)           │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Regression Layer (输出层)                   │
│              Conv2d: (hidden_dim → output_window)           │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
              Output Prediction (B, output_window, N, 1)
```

### 3. 核心组件详解

#### 3.1 STID Embedding Layer

```python
# 时间序列嵌入
time_series_emb = Conv2d(in_channels=input_window, out_channels=time_series_emb_dim)

# 时间嵌入 (Time-in-Day)
time_in_day_emb = Embedding(time_of_day_size, temp_dim_tid)

# 星期嵌入 (Day-in-Week)
day_in_week_emb = Embedding(day_of_week_size, temp_dim_diw)

# 空间嵌入 (节点嵌入)
node_emb = Embedding(num_nodes, spatial_emb_dim)
```

**输入特征**：
- `X[..., 0]`: 数值特征（订单数/流量等）
- `X[..., 1]`: 一天内时间 (归一化 0-1)
- `X[..., 2]`: 星期几 (归一化 0-1)

#### 3.2 ProtoModule (原型模块)

原型模块是 PSTID 的核心创新，包含两个子模块：

##### 3.2.1 Spatial Codebook (空间码本)

**功能**：编码静态/半静态的节点身份信息

**结构**：
```
输入: time_series_emb (B,T,N,D_ts) + spatio_emb (N,D_spatio)
  │
  ├──► 时间序列均值: time_series_emb.mean(dim=[0,1]) → (N, D_ts)
  │                    │
  │                    └──► Linear(D_ts → D_proto) → time_series_proj
  │
  ├──► 空间嵌入投影: spatio_emb → Linear(D_spatio → D_proto) → spatio_emb_proj
  │
  ├──► 特征融合: spatio_emb_proj + time_series_proj → x_in (N, D_proto)
  │
  ├──► Query投影: x_in → Linear(D_proto → D_proto) → proj_feat
  │
  ├──► 注意力计算:
  │      • 归一化: prototypes / proj_feat
  │      • 点积相似度 / temperature
  │      • Top-K 选择
  │      • Softmax 归一化
  │
  └──► 原型增强: Top-K weights × prototypes → proto_enhanced (N, D_proto)
```

**关键机制**：
- **正交初始化**: `nn.init.orthogonal_(prototypes)` 促进原型多样性
- **Top-K 注意力**: 只选择最相关的 K 个原型，减少噪声
- **Idx Dropout**: 训练时随机丢弃部分原型，防止过拟合

##### 3.2.2 Temporal Basis (时间基)

**功能**：编码动态的时序模式，每个时间步使用不同的原型组合

**结构**：
```
输入: time_series_emb (B,T,N,D_ts) + tid_emb (B,T,N,D_tid) + diw_emb (B,T,N,D_diw)
  │
  ├──► 提取最后时间步: time_series_emb[:, -1], tid_emb[:, -1], diw_emb[:, -1]
  │
  ├──► 特征拼接: [ts_last; tid_last; diw_last] → (B*N, D_in)
  │
  ├──► MLP Allocator:
  │      Linear(D_in → D_in*2) → GELU → Linear(D_in*2 → n_temporal) → base_attn
  │
  ├──► Softmax / temperature: 归一化为概率分布
  │
  └──► 原型增强: base_attn @ prototypes → proto_enhanced (B*N, D_proto)
```

**关键机制**：
- **MLP 直接预测**: 不使用点积注意力，而是通过 MLP 直接预测原型分配权重
- **动态分配**: 每个样本/时间步可以有完全不同的原型组合
- **时间感知**: 结合了 TID 和 DIW 嵌入的时间信息

#### 3.3 Fusion (融合)

```python
# 空间 + 时间 原型增强融合
if use_spatio and use_temporal:
    proto_enhanced = sp_proto_enhanced + tp_proto_enhanced
elif use_spatio:
    proto_enhanced = sp_proto_enhanced
else:
    proto_enhanced = tp_proto_enhanced

# 残差连接
hidden = concat(embeddings) + proto_enhanced
```

### 4. 数据流

```
Batch Input: X (B, T, N, 3)
     │
     ▼
Embedding: emb (B, T, N, hidden_dim)
     │
     ▼
ProtoModule:
     ├──► SpatialCodebook → sp_proto (N_s, D)
     ├──► TemporalBasis → tp_proto (N_t, D)
     └──► Fusion → proto_enhanced (B, T, N, D)
     │
     ▼ (residual add)
Hidden: hidden (B, T, N, hidden_dim)
     │
     ▼
MLP Layers: encoder (B, hidden_dim, N, T)
     │
     ▼
Regression: output (B, output_window, N, 1)
```

### 5. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_window` | 24 | 输入序列长度 |
| `output_window` | 24 | 预测序列长度 |
| `num_layers` | 3 | MLP 层数 |
| `input_embedding_dim` | 32 | 输入嵌入维度 |
| `spatial_emb_dim` | 16 | 空间嵌入维度 |
| `temp_dim_tid` | 16 | 时间嵌入维度 |
| `temp_dim_diw` | 16 | 星期嵌入维度 |
| `num_spatial_prototypes` | 4-16 | 空间原型数量 |
| `num_temporal_prototypes` | 16-1326 | 时间原型数量 |
| `proto_temperature` | 0.5 | 原型温度参数 |
| `spatial_idx_dropout` | 0.1 | 空间原型丢弃率 |

### 6. 消融开关

| 参数 | 说明 |
|------|------|
| `--use_proto` | 是否使用原型模块 |
| `--use_spatio` | 是否使用空间码本 |
| `--use_temporal` | 是否使用时间基 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 训练基线模型 (HA, LSTM)
python train.py --model HA --data nyc_taxi
python train.py --model LSTM --data nyc_taxi --epochs 50

# 训练 PSTID 模型
python train.py --model PSTID --data nyc_taxi --epochs 100 --batch_size 64

# 训练 PSTID 模型
python train.py --model PSTID --data nyc_taxi --epochs 100 --batch_size 64

# 训练其他模型
python train.py --model STID --data nyc_taxi
```

### 3. 消融实验

```bash
# 完整 PSTID
python train.py --model PSTID --use_proto 1 --use_spatio 1 --use_temporal 1

# 无原型
python train.py --model PSTID --use_proto 0

# 无空间原型
python train.py --model PSTID --use_proto 1 --use_spatio 0

# 无时间原型
python train.py --model PSTID --use_proto 1 --use_temporal 0
```

## 数据格式

### NPZ 文件

每个 npz 文件包含:
- `X`: 输入特征，shape `(num_samples, seq_len, num_nodes, 3)`
- `y`: 预测目标，shape `(num_samples, horizon, num_nodes, 1)`

### 邻接矩阵

`adj.pkl` 包含 `(num_nodes, num_nodes)` 的 numpy 数组，表示基于地理距离的邻接权重。

## 评估指标

- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **SMAPE**: 对称平均绝对百分比误差

## 可视化

PSTID 支持丰富的原型可视化：

- **原型分布**: t-SNE 降维可视化
- **Query-Prototype 对比**: 查询与原型的匹配关系
- **原型使用率**: 各原型的使用频率统计
- **时间步级使用率**: Temporal Prototype 随时间的变化

---

## 引用

```bibtex
@article{pstid2024,
  title={PSTID: 原型时空身份网络},
  author={},
  year={2024}
}
```
