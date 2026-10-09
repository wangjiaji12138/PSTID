"""从 PyTorch checkpoint 导出完整嵌入和时空原型参数（CPU，无需数据集）。

用法：python get_emb.py --checkpoint path/to/best_model.pth -o embeddings.json
JSON 直接按时间、星期和节点编号保存完整向量，原型参数按原始参数名保存数组。
"""

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import pickle
import tempfile

import torch


PARAMETER_KEYS = {
    "tid_emb": ("time_in_day_emb", "tid_emb"),
    "diw_emb": ("day_in_week_emb", "diw_emb"),
    "node_emb": ("node_emb",),
    "adj_mx_emb": ("adj_mx_emb",),
}
REQUIRED = ("tid_emb", "diw_emb", "node_emb")
WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def clock_label(seconds):
    """时间槽边界；一天结束保留为 24:00:00。"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    second_label = f"{seconds:09.6f}".rstrip("0").rstrip(".")
    return f"{int(hours):02d}:{int(minutes):02d}:{second_label}"


def add_time_labels(parameters, checkpoint, weekday_zero=None):
    """将嵌入矩阵改为标签到完整向量的映射，不重复保存数值。"""
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    dataset = str(config.get("data", "")).lower()
    tid, diw = parameters["tid_emb"], parameters["diw_emb"]

    if weekday_zero is not None and weekday_zero not in range(7):
        raise ValueError("weekday_zero 必须为 0（星期一）到 6（星期日）。")
    if weekday_zero is None and dataset in {"cq", "hz", "sh", "jl", "yt"}:
        weekday_zero = 0
    if len(diw) != 7 and weekday_zero is not None:
        raise ValueError("diw_emb 不是 7 行，无法标注星期。")

    parameters["tid_emb"] = {
        clock_label(index * 86400 / len(tid)): values
        for index, values in enumerate(tid)
    }
    parameters["diw_emb"] = {
        WEEKDAYS[(weekday_zero + index) % 7]
        if weekday_zero is not None else f"周期日索引_{index}": values
        for index, values in enumerate(diw)
    }
    for name in ("node_emb", "adj_mx_emb"):
        if name in parameters:
            parameters[name] = {
                str(index): values for index, values in enumerate(parameters[name])
            }


def normalize_key(key):
    """兼容 DataParallel、模型包装及 torch.compile 的参数前缀。"""
    parts = key.split(".")
    while parts and parts[0] in {"module", "model", "_orig_mod"}:
        parts.pop(0)
    return ".".join(parts)


def extract_state_dict(checkpoint):
    """支持项目 checkpoint、常见包装和直接保存的 state_dict。"""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint 必须是字典或 state_dict。")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), Mapping):
            return checkpoint[key], key
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, "<root>"
    raise ValueError("未找到 model_state_dict/state_dict/model，也不是纯张量 state_dict。")


def tensor_values(key, tensor):
    """不转换浮点精度、不归一化、不展平、不截断。"""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"参数 {key} 不是 Tensor。")
    value = tensor.detach().cpu()
    if value.layout != torch.strided or value.is_quantized or value.is_complex():
        raise ValueError(f"参数 {key} 必须是普通实数稠密张量。")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"参数 {key} 含 NaN/Infinity，无法无损写入标准 JSON。")
    return value.tolist()


def export_embeddings(checkpoint_path, output_path=None, weekday_zero=None):
    """导出参数，返回实际输出路径和 JSON 内容。"""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else checkpoint_path.with_suffix(".emb.json")
    )
    if output_path == checkpoint_path or (
        output_path.exists() and output_path.samefile(checkpoint_path)
    ):
        raise ValueError("JSON 输出路径不能与 checkpoint 相同。")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict, _ = extract_state_dict(checkpoint)
    indexed = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("state_dict 的键必须是字符串。")
        normalized = normalize_key(key)
        if normalized in indexed:
            raise ValueError(f"去除包装前缀后出现重复参数名：{normalized}")
        indexed[normalized] = (key, tensor)

    parameters = {}
    missing = []
    for name, aliases in PARAMETER_KEYS.items():
        candidates = [
            key for alias in aliases for key in (alias, alias + ".weight")
            if key in indexed
        ]
        if len(candidates) > 1:
            raise ValueError(f"{name} 对应多个参数，无法唯一选择：{candidates}")
        if not candidates:
            missing.append(name)
            continue
        key = candidates[0]
        original, tensor = indexed[key]
        values = tensor_values(original, tensor)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError(f"{name} 必须是非空二维嵌入矩阵，才能标注行对应关系。")
        parameters[name] = values

    required_missing = [name for name in REQUIRED if name in missing]
    if required_missing:
        raise ValueError("缺少必需的嵌入参数：" + ", ".join(required_missing))

    add_time_labels(parameters, checkpoint, weekday_zero)

    # 时空原型及其模块参数统一按 checkpoint 原始参数名保存，每个参数只保存一次。
    payload = {
        **parameters,
        "prototype_parameters": {
            original: tensor_values(original, tensor)
            for key, (original, tensor) in indexed.items()
            if key.startswith("proto_module.")
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "-c", required=True, type=Path,
                        help="PyTorch checkpoint 或 state_dict 文件")
    parser.add_argument("--output", "-o", type=Path,
                        help="输出 JSON；默认在 checkpoint 旁生成 <文件名>.emb.json")
    parser.add_argument("--weekday-zero", type=int, choices=range(7),
                        help="指定星期嵌入索引 0 对应的实际星期：0=周一，6=周日；默认按数据集编码判断")
    args = parser.parse_args()
    try:
        output_path, _ = export_embeddings(args.checkpoint, args.output, args.weekday_zero)
    except (OSError, ValueError, TypeError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        parser.exit(1, f"导出失败：{exc}\n")
    print(f"已导出：{output_path}")



if __name__ == "__main__":
    main()
