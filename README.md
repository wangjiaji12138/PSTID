# PROPHET：原型时空身份网络

用于城市出行需求等时空数据预测。项目包含数据准备、模型训练、验证测试、checkpoint 保存和实验结果汇总。

## 目录与职责

```text
PROPHET/
├── models/
│   ├── registry.py             # 唯一模型列表、按需导入、模型参数适配
│   ├── base.py                 # 模型基类
│   ├── PROPHET/
│   │   ├── model.py            # 主网络、嵌入、前向传播、from_args
│   │   ├── spatial.py          # 空间码本 SpatialCodebook
│   │   ├── temporal.py         # 时间基 TemporalBasis
│   │   ├── prototype.py        # 原型特征融合 ProtoModule
│   │   ├── graph.py            # 邻接矩阵谱嵌入初始化
│   │   ├── layers.py           # 残差 MLP
│   │   └── PROPHET.py            # 旧模型导入路径兼容层
│   └── <baseline>/             # STID、GRU、AGCRN 等基线
├── data_pipeline/
│   ├── dataset.py              # all.npz/adj.pkl 读取、Dataset、Scaler
│   ├── loaders.py              # DataLoader 与随机种子
│   ├── normalization.py        # 数值通道归一化
│   ├── grid.py                 # 原始订单 CSV → 网格时空数据
│   └── npz.py                  # 原始 NPZ → 预测滑动窗口
├── training/
│   ├── config.py               # 命令行参数
│   ├── runner.py               # 内部实验流程，由 train.py 调用
│   ├── engine.py               # 单轮训练、评估、早停
│   ├── checkpoint.py           # checkpoint、指标与配置保存
│   └── reproducibility.py      # 随机种子与确定性设置
├── experiments/ablation.py     # 消融结果汇总
├── utils/                     # 指标、日志和项目路径
├── tests/                     # 数值回归与训练集成测试
├── data/                      # 原始数据与预处理产物
├── results/                   # 训练结果
└── train.py                    # 训练启动入口
```

详细设计见 [架构与迁移说明](docs/architecture.md)；模型计算过程见 [PROPHET 组件说明](models/PROPHET/README.md)。

## 使用

在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

预处理两类数据：

```bash
python -m data_pipeline.grid --city cq
python -m data_pipeline.npz NYC_TAXI
```

训练与消融：

```bash
python train.py --model PROPHET --data cq --epochs 100 --gpu 0
python train.py --model PROPHET --data cq --use_proto 0 --gpu 0
python train.py --model PROPHET --data cq --use_spatio 0 --gpu 0
python train.py --model PROPHET --data cq --use_temporal 0 --gpu 0
python train.py --model PROPHET --data cq --use_adj 0 --gpu 0
python train.py --model PROPHET --data cq --use_his 0 --gpu 0
python train.py --model STID --data cq --gpu 0
```

CPU 运行用 `--device cpu`；`--results_dir` 可指定实验结果根目录。训练结束后保存模型、配置、日志及指标，不生成图片或采集原型诊断数据。

消融结果分析：

```bash
python -m experiments.ablation
```

训练统一通过 `python train.py` 启动，`training/runner.py` 仅实现内部实验流程。根目录的批量实验 shell 脚本也使用 `train.py`。`python gen_dataset.py`、`python prepare_dataset.py` 和 `python abl.py` 仍可使用。绘图入口、示例曲线生成脚本和 `--no_viz` 参数已删除。

## 数据与模型接口

每个数据集位于 `data/processed/<数据集>/`，至少包含：

- `all.npz`：`X` 为 `(样本数, 输入窗口, 节点数, 3)`，`y` 为 `(样本数, 预测窗口, 节点数, 1)`。
- `adj.pkl`：形状为 `(节点数, 节点数)` 的邻接矩阵。

三个输入通道依次为数值、归一化的日内时间、归一化的星期。默认按样本顺序划分训练/验证/测试集，比例为 `0.6/0.2/0.2`。Dataset 返回原始值，由归一化函数处理数值通道，时间特征保持原样。

模型统一通过 `build_model(name, args, num_nodes, adj_mx, device)` 创建；PROPHET 使用 `model({'X': X})`，返回 `(B, 预测窗口, N, 1)`。

当前支持：HA、GRU、STGCN、AGCRN、GWNET、MegaCRN、STNorm、MTGNN、STAEformer、STDN、STID、PROPHET、DCRNN、STSSL、STSSDL、HimNet。名称不区分大小写；以 `models/registry.py` 为准。

## 验证

```bash
python -m unittest discover -s tests -v
```

覆盖原始 PROPHET 五种配置的输出、参数键与梯度基准，旧模型导入路径，数据窗口与归一化，前向无诊断缓存，以及禁止绘图库导入情况下的 CPU 短训练和 checkpoint 指标复现。测试使用临时目录，不写入真实数据集或实验结果。
