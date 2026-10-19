# 训练架构与迁移说明

## 依赖方向

```text
train.py（训练启动入口）
       │
       ▼
training.runner ──→ training.engine ──→ models
       │                    │
       ├─────────→ data_pipeline
       └─────────→ training.checkpoint

experiments.ablation ──→ 已保存的配置和测试指标
```

训练通过 `python train.py` 启动，由它调用 `training.runner.main` 组织实验。模型代码只负责预测计算。训练引擎负责损失、优化、验证、早停和测试。通用指标和日志放在 `utils`，数据逻辑集中在 `data_pipeline`。

## 原文件迁移

| 原文件 | 实际实现 | 兼容方式 |
| --- | --- | --- |
| `models/PROPHET/PROPHET.py` | `models/PROPHET/model.py` 及各组件 | 导出模型、计算组件和图初始化函数 |
| `model_builder.py` | `models/registry.py` | 保留 `build_model`、`MODEL_LIST` |
| `train.py` | `training/` 与 `data_pipeline/` | 训练启动入口，并保留主要训练函数导出 |
| `gen_dataset.py` | `data_pipeline/grid.py` | 保留类和 CLI |
| `prepare_dataset.py` | `data_pipeline/npz.py` | 保留函数、配置和 CLI |
| `utils/data_utils.py` | `data_pipeline/dataset.py` | 保留三个公共对象 |
| `abl.py` | `experiments/ablation.py` | 保留报告函数和 CLI |

新增代码应导入实际实现模块；训练命令统一使用 `train.py`，其余兼容入口用于旧脚本和调用方。基线模型的目录及算法保持原样。

绘图模块、数据采集钩子、原型使用率缓存及对应命令已移除。`PROPHET.forward` 只接收 batch，`ProtoModule.forward` 只返回融合特征；不再提供诊断对象或额外注意力导出。注意力仍作为原型加权计算的中间结果参与训练。训练日志保留损失、学习率和评估指标。

## 模型扩展

1. 在 `models/<模型名>/` 实现模型，明确输入和输出形状。
2. 在 `models/registry.py` 增加参数适配函数，并注册到 `MODEL_REGISTRY`。在适配函数内导入模型，避免其他模型的依赖影响当前实验。
3. 需要新的命令行参数时修改 `training/config.py`。PROPHET 的参数映射由 `PROPHET.from_args` 统一维护。
4. 标准模型遵守 `forward({'X': X})` 接口。现有 STSSL/STSSDL 的自定义损失和预测分支由训练引擎保留。
5. 添加模型前向、输出形状和 checkpoint 测试。

`MODEL_LIST` 由注册表派生，训练 CLI 和模型工厂共用。

## 路径和产物

`utils/paths.py` 是项目根路径的唯一来源。默认数据和结果路径不会随源码迁移而改变；显式传入的相对路径按调用方当前工作目录解释。

每次训练保存：

```text
results/<数据集>/<模型>_<时间戳>/
├── checkpoint/best_model.pth
├── log/
├── training_history.json
├── test_results.json
└── config.json
```

`--results_dir` 指定上述 `results` 根目录。HA 没有可训练参数，沿用直接评估路径，不生成模型 checkpoint。

checkpoint 字段及 PROPHET 的参数名保持不变，例如 `proto_module.spatial_codebook.prototypes`、`encoder.0.fc1.weight`。删除的诊断缓存不是已注册参数，因此项目使用的 `model_state_dict` 不需要迁移键名。

## 回归依据

`tests/fixtures/prophet_reference.json` 来自原始 PROPHET 实现，记录固定 CPU 输入下完整模型和四种消融配置的输出、参数键及输入嵌入层梯度。测试还验证训练不依赖绘图库，前向传播不创建诊断缓存，以及重新加载 checkpoint 后能复现测试指标。

数据切分比例、滑动窗口策略、标准配置的模型计算和损失定义沿用原实现。CPU 短训练验证不等于完整数据集的精度重新评测。
