# 模型配置 registry（官方来源与本地适配）

核验日期：2026-11-03。`models/official_configs.json` 覆盖 `MODEL_LIST` 全部 17 项；其中 13 项有作者官方仓库，4 项为通用基线或本地模型。每项保存固定 commit 链接、选用的数据集 preset、官方配置内容/脚本字面量、映射后的模型及训练参数，以及适配限制。Python 脚本的 `official_config` 是静态提取的默认值和字面量，不是执行结果；README 命令或训练函数覆盖以映射及 notes 为准。

## 使用方式与固定基线策略

```bash
python train.py --model STAEformer --data cq --gpu 0 --seed 42
python train.py --model AGCRN --data hz
python train.py --model PSTID --learning_rate 0.001 --input_embedding_dim 64
```

除 PSTID、PROPHET 外，所有基线的模型超参数及非统一训练设置均写死在 `models/registry.py`。各 `_build_*` 使用固定构造参数；`BASELINE_CONFIGS` 保存与这些参数对应的配置记录及固定训练设置。`official_configs.json` 为官方来源和映射记录，修改 JSON 不会改变基线运行参数。

基线的显式 CLI 或调用方超参数会被 registry 固定值覆盖，包括层数、隐藏维度、dropout、学习率、weight decay、梯度裁剪、优化器及调度。实验 config 保存实际生效值，`hyperparameter_policy=fixed_baseline`。直接 `build_model()` 也执行相同策略。旧 checkpoint 如果采用了不同结构，不会通过传入历史超参数自动恢复，需使用对应旧代码。

**统一训练设置：** 所有模型的 `batch_size` 和最大训练轮数均由训练入口设置，默认分别为 64 和 100。内部字段统一为 `args.max_epoch`；`--max_epoch` 为标准命令行名称，`--max_epochs`、`--epochs`、`--epoch` 为兼容别名。这两项已从基线固定表移除，命令行值不会被 registry 覆盖；早停仍可能在轮数上限之前结束训练。例：`python train.py --model AGCRN --max_epoch 200 --batch_size 32`。

PSTID、PROPHET 保留命令行/调用方参数优先的调参方式，`hyperparameter_policy=tunable`。

数据集、数据路径、输入/输出窗口、采样间隔、随机种子、GPU/设备、结果路径仍为实验设置，可以指定；时间嵌入槽数由采样间隔计算。基线特征布局固定为当前训练数据格式（3 输入通道，1 预测通道；部分模型内部只取需求通道）。

`MODEL_REGISTRY` 仍为统一模型工厂入口。`MODEL_CONFIG_REGISTRY`、`get_model_config()` 查询官方来源记录，基线实际固定设置查看 `BASELINE_CONFIGS` 及对应 `_build_*`。

## 默认 preset 与训练设置

CQ/HZ 等自有数据集没有这些仓库的官方 preset。本表明确选择一个参考 preset 迁移，不能称为官方为 CQ/HZ 推荐的最优配置。STAEformer、HimNet 的其他官方数据集配置也保存在 JSON 中，当前不会自动按 `--data` 切换。

下表的 batch/epochs 是来源配置的参考值，实际训练统一使用命令行设置（未指定时为 64/100）。

| 模型 | 参考 preset | LR | batch（来源） | epochs（来源） | weight decay | 调度 | patience |
|---|---|---:|---:|---:|---:|---|---:|
| AGCRN | PEMSD4 | 0.003 | 64 | 100 | 0 | none | 15 |
| GWNET | METR-LA README command | 0.001 | 64 | 100 | 0.0001 | none | 0 |
| MTGNN | METR-LA multi-step | 0.001 | 64 | 100 | 0.0001 | none | 0 |
| STAEFORMER | PEMS04 | 0.001 | 16 | 300 | 0.0005 | multistep [15, 30, 50] ×0.1 | 20 |
| HIMNET | PEMS04 | 0.001 | 16 | 200 | 0.0001 | multistep [30, 50] ×0.1 | 20 |
| MEGACRN | METRLA | 0.01 | 64 | 200 | 0 | multistep [50, 100] ×0.1 | 20 |
| STDN | PEMSD4 | 0.001 | 64 | 200 | 0 | step /10 ×0.9 | 10 |
| DCRNN | METR-LA | 0.01 | 64 | 100 | 0 | multistep [20, 30, 40, 50] ×0.1 | 50 |
| STSSL | NYCTaxi | 0.001 | 32 | 100 | 0 | none | 15 |
| STSSDL | METRLA argparse defaults | 0.01 | 64 | 200 | 0 | multistep [50, 100] ×0.1 | 30 |
| STGCN | PeMSD7(M) | 0.001 | 50 | 50 | 0 | step /5 ×0.7 | 0 |
| STNORM | electricity / Wavenet | 0.0001 | 8 | 1000 | 0 | none | 50 |
| STID | PEMS04 | 0.002 | 64 | 100 | 0.0001 | multistep [1, 50, 80] ×0.5 | 0 |
| HA | local | 0.001 | 64 | 100 | 1e-05 | multistep [20, 40, 50, 60, 70] ×0.1 | 15 |
| GRU | local | 0.001 | 64 | 100 | 1e-05 | multistep [20, 40, 50, 60, 70] ×0.1 | 15 |
| PROPHET | local | 0.001 | 64 | 100 | 1e-05 | multistep [20, 40, 50, 60, 70] ×0.1 | 15 |
| PSTID | local | 0.001 | 64 | 100 | 1e-05 | multistep [20, 40, 50, 60, 70] ×0.1 | 15 |

HA 不使用优化器，表中本地训练字段仅为接口兼容。训练设置还包含 Adam epsilon、梯度裁剪、优化器类型、MSE/MAE；详见 JSON。STGCN 使用 RMSprop；其余可训练模型使用 Adam。DCRNN 的最低 LR、2000 步 scheduled sampling 已接入训练路径。

## 数据与复现边界

保留本项目数据划分、输入/输出窗口（默认 24/24）、节点数、特征布局、随机种子和标准化。时间嵌入大小从 `time_intervals` 计算，默认 1800 秒对应每天 48 个槽；STDN 的时间片对应 30 分钟，而不是照搬官方 5 分钟。解析结果和 `registry_provenance` 写入实验 config/checkpoint。

这次修改的是配置及其必要训练接线，不是重写所有模型。所有官方项均标记为 `official_adapted`，不能据此声称已精确复现官方实验。特别注意：

- STGCN 本地网络为简化结构，无法表达官方 bottleneck / Chebyshev 网络；输出尾轴保留历史时间并由共享训练端平均。采用 MSE/RMSprop 参数仍不等同于 TensorFlow 官方训练。
- MegaCRN 本地 forward 使用随机未来日历占位，不使用 curriculum；共享训练端没有官方 memory 辅助损失。registry 保留官方设置并标注未生效部分。
- MTGNN 官方逐步增加预测长度的 curriculum 尚未接入。
- STSSL 官方 DWA 动态权重未接入；本地保持固定辅助损失权重，输入历史和输出头也不同。
- STSSDL 使用本项目已有训练集历史锚点版本，继续关闭 curriculum（官方开启）；其余尺寸、损失系数和训练参数来自官方。
- STNorm 官方仓库当前提供 electricity 配置；每 10 epoch 验证、连续 5 次无提升早停在本地以每 epoch 验证、patience=50 近似；scaler 和选 checkpoint 指标也不同。
- GWNET 已取消按窗口自动扩展层数，默认保持官方每 block 2 层；长窗口取最后一个时间位置的预测。旧 checkpoint 若含 3 层，需要使用对应旧代码加载。STDN 使用邻接矩阵的位置编码要求至少 32 个节点。
- 官方多种 masked MAE 忽略零值；当前共享 MAE 保留零需求。保留本地零值语义，不能将此处损失称为官方 masked MAE。

## 官方来源与逐模型适配说明

### AGCRN

状态：`official_adapted`；参考：PEMSD4。

- [model/PEMSD4_AGCRN.conf](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/PEMSD4_AGCRN.conf)
- [model/Run.py](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/Run.py)

- Official grad_norm and lr_decay are disabled. Dataset seed/splits are retained locally.

### GWNET

状态：`official_adapted`；参考：METR-LA README command。

- [train.py](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/train.py)
- [model.py](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/model.py)
- [engine.py](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/engine.py)
- [README.md](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/README.md)

- Flags follow README training command, not bare argparse defaults. Local encoder consumes demand only (1 vs official 2 channels).
- Dataset node count, windows, channels, time resolution, seed, splits and preprocessing remain local. Shared MAE does not apply upstream missing-zero masks. See this document for reproduction limits.

### MTGNN

状态：`official_adapted`；参考：METR-LA multi-step。

- [train_multi_step.py](https://github.com/nnzhan/MTGNN/blob/f811746fa7022ebf336f9ecd2434af5f365ecbf6/train_multi_step.py)
- [trainer.py](https://github.com/nnzhan/MTGNN/blob/f811746fa7022ebf336f9ecd2434af5f365ecbf6/trainer.py)

- Official horizon curriculum (cl=True, step_size1=2500) is not implemented by the local engine. Local input has 3 vs official 2 channels.

### STAEFORMER

状态：`official_adapted`；参考：PEMS04。

- [model/STAEformer.yaml](https://github.com/XDZhelheim/STAEformer/blob/fc49d39b2f1a8e3cf37b6289d7240680e1690f3f/model/STAEformer.yaml)
- [model/train.py](https://github.com/XDZhelheim/STAEformer/blob/fc49d39b2f1a8e3cf37b6289d7240680e1690f3f/model/train.py)

- PEMS04 is the explicit transfer preset for local city datasets; other official dataset presets are preserved in official_config.

### HIMNET

状态：`official_adapted`；参考：PEMS04。

- [model/HimNet.yaml](https://github.com/XDZhelheim/HimNet/blob/2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5/model/HimNet.yaml)
- [scripts/train.py](https://github.com/XDZhelheim/HimNet/blob/2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5/scripts/train.py)

- Calendar sizes and future covariates follow the local dataset.

### MEGACRN

状态：`official_adapted`；参考：METRLA。

- [model/traintest_MegaCRN.py](https://github.com/deepkashiwa20/MegaCRN/blob/5158190948de7e8628f84780a429eb9189f5908f/model/traintest_MegaCRN.py)
- [model/MegaCRN.py](https://github.com/deepkashiwa20/MegaCRN/blob/5158190948de7e8628f84780a429eb9189f5908f/model/MegaCRN.py)

- Local forward ignores curriculum and generates random calendar placeholders (official ycov_dim=1; local requires 2). Shared engine does not use official memory auxiliary losses. This is not an official reproduction.

### STDN

状态：`official_adapted`；参考：PEMSD4。

- [conf/PEMSD4_1dim_12.conf](https://github.com/roarer008/STDN/blob/e24a60fa9485fb9805df9c171d67ab60840d757d/conf/PEMSD4_1dim_12.conf)
- [train.py](https://github.com/roarer008/STDN/blob/e24a60fa9485fb9805df9c171d67ab60840d757d/train.py)

- model_dim=K*d. time_slice_size follows local sampling minutes.
- Dataset node count, windows, channels, time resolution, seed, splits and preprocessing remain local. Shared MAE does not apply upstream missing-zero masks. See this document for reproduction limits.

### DCRNN

状态：`official_adapted`；参考：METR-LA。

- [data/model/dcrnn_la.yaml](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/model/dcrnn_la.yaml)
- [model/dcrnn_supervisor.py](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/model/dcrnn_supervisor.py)

- Official TensorFlow model uses 2 input channels; local implementation uses demand only.

### STSSL

状态：`official_adapted`；参考：NYCTaxi。

- [configs/NYCTaxi.yaml](https://github.com/Echo-Ji/ST-SSL/blob/b3f3dfc48d12524c207cc89f4e242cb8aac081c0/configs/NYCTaxi.yaml)
- [model/models.py](https://github.com/Echo-Ji/ST-SSL/blob/b3f3dfc48d12524c207cc89f4e242cb8aac081c0/model/models.py)
- [model/trainer.py](https://github.com/Echo-Ji/ST-SSL/blob/b3f3dfc48d12524c207cc89f4e242cb8aac081c0/model/trainer.py)

- Official input_length=35 (noncontiguous history), output=1 step with inflow/outflow; local uses contiguous history and single-channel multi-step head. Official DWA (temp=2) is not implemented; local fixed auxiliary weights remain.

### STSSDL

状态：`official_adapted`；参考：METRLA argparse defaults。

- [model_STSSDL/train_STSSDL.py](https://github.com/Jimmy-7664/ST-SSDL/blob/a98331d3a45a081778f3abb608897902a60e301d/model_STSSDL/train_STSSDL.py)
- [model_STSSDL/STSSDL.py](https://github.com/Jimmy-7664/ST-SSDL/blob/a98331d3a45a081778f3abb608897902a60e301d/model_STSSDL/STSSDL.py)

- Official use_curriculum_learning=True; local weekly-flow-anchor variant deliberately disables it. Local history fitted on training inputs only; contrastive implementation may differ from official triplet.

### STGCN

状态：`official_adapted`；参考：PeMSD7(M)。

- [main.py](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/4ab75ab69af354d6a0ab86aefa082d389b187127/main.py)
- [models/trainer.py](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/4ab75ab69af354d6a0ab86aefa082d389b187127/models/trainer.py)

- Official bottleneck blocks [[1,32,64],[64,32,128]], Ks=Kt=3 cannot be represented by local simplified constant-width first-order graph convolution. Official loss is tf.nn.l2_loss on standardized single-step predictions; local uses demand-unit mean squared error. RMSProp uses PyTorch semantics; this is not exact TensorFlow reproduction.
- Dataset node count, windows, channels, time resolution, seed, splits and preprocessing remain local. Shared MAE does not apply upstream missing-zero masks. See this document for reproduction limits.

### STNORM

状态：`official_adapted`；参考：electricity / Wavenet。

- [main.py](https://github.com/JLDeng/ST-Norm/blob/ae228bd93e33355f6b84deec43e447b0b062dad4/main.py)
- [models/Wavenet.py](https://github.com/JLDeng/ST-Norm/blob/ae228bd93e33355f6b84deec43e447b0b062dad4/models/Wavenet.py)
- [README.md](https://github.com/JLDeng/ST-Norm/blob/ae228bd93e33355f6b84deec43e447b0b062dad4/README.md)

- Official script trains electricity, not a traffic-specific preset. Official validation every 10 epochs, stop after 5 checks; local validation every epoch with patience=50 is an approximation. Official MinMax scaling and RMSE checkpoint selection differ from local standardization and MAE selection. Wavenet defines dropout=0.2 but never applies it; effective local dropout=0.

### STID

状态：`official_adapted`；参考：PEMS04。

- [stid/PEMS04.py](https://github.com/GestaltCogTeam/STID/blob/e8b313bc591bdd0101a1619962c9b503e75127c0/stid/PEMS04.py)

- Official time_of_day_size=288 is adapted to local sampling.

### HA

状态：`local`；参考：local。


- Historical average baseline; no unique official repository or trainable parameters.

### GRU

状态：`local`；参考：local。


- Generic GRU baseline; no unique official forecasting hyperparameter preset.

### PROPHET

状态：`local`；参考：local。


- Local research model; retain project defaults, not claimed to be upstream official.

### PSTID

状态：`local`；参考：local。


- Local research model; retain project defaults, not claimed to be upstream official.

