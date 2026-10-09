# 导出 checkpoint 原始参数

在项目根目录运行（仅依赖 PyTorch 和 Python 标准库）：

```bash
python get_emb.py --checkpoint results/cq/PROPHET_YYYYMMDD_HHMMSS/checkpoint/best_model.pth
python get_emb.py --checkpoint /path/to/best_model.pth --output /path/to/embeddings.json
```

默认输出到 checkpoint 同目录的 `best_model.emb.json`。指定输出路径时自动创建父目录，成功后替换该路径已有的 JSON。脚本使用 CPU 直接读取保存的权重，无需实例化模型、加载数据集或执行前向传播。

## JSON 结构

- `tid_emb`：时间字符串（如 `08:00:00`） → 完整向量。
- `diw_emb`：星期字符串（如 `星期一`） → 完整向量。
- `node_emb`、`adj_mx_emb`：从 0 开始的节点编号字符串 → 完整向量，顺序与 checkpoint 一致。
- `prototype_parameters`：checkpoint 原始参数名 → 原始数值数组，包含时空原型矩阵及 `proto_module` 内的全部参数。

所有字段直接位于 JSON 顶层。输出不含路径、形状、类型、说明、索引表或重复的向量视图。原型参数保留原始维度，标量保存为数字；不降维、不展平、不归一化、不取均值，也不进行小数位舍入或浮点类型转换。

示例读取：

```python
import json

with open("best_model.emb.json", encoding="utf-8") as f:
    exported = json.load(f)

print(exported["tid_emb"]["08:00:00"])
print(exported["diw_emb"]["星期一"])
print(exported["node_emb"]["0"])
print(exported["prototype_parameters"]["proto_module.spatial_codebook.prototypes"])
```

消融模型未保存的邻接嵌入不输出，未保存的原型参数也不输出。没有原型模块时 `prototype_parameters` 为空字典。三个必需嵌入 `tid_emb`、`diw_emb`、`node_emb` 缺失时会报错，不生成不完整的导出文件。

## 时间和星期标签

日内时间标签根据嵌入行数均分一天，表示时间槽的起点，不作时区转换。例如 48 行对应每槽 30 分钟，标签为 `00:00:00`、`00:30:00` 等。

`cq/hz/sh/jl/yt` 按当前数据管线使用日历星期，行 0 为星期一。`nyc_taxi/chi_taxi/nyc_bike` 从处理后数据起点按七天循环编码，没有保存真实星期起点；这些数据集、未知数据集及纯 state_dict 默认使用 `周期日索引_0` 等标签。确认实际星期起点后，可用 `--weekday-zero 4` 指定行 0 是星期五（0=周一，6=周日）。此参数只补充标签，不改动权重。

每天同一钟点共享同一个日内嵌入，同一星期共享同一个星期嵌入。时空原型的行号是学习到的原型编号，不对应固定钟点或节点。

支持 `model_state_dict`、`state_dict`/`model` 字典包装及直接保存的 state_dict，也兼容 `module.`、`model.`、`_orig_mod.` 前缀和嵌入的 `.weight` 形式。使用 `weights_only=True` 加载；包含 NaN/Infinity 的目标参数会报错。
