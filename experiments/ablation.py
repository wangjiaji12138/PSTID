"""
PROPHET Ablation Analysis Script
==============================
自动读取 results/ 目录下各数据集的 PROPHET（原名 PSTID）消融实验结果，
根据 config.json 中的 use_proto / use_spatio / use_temporal / use_adj / use_his 标记
自动识别实验类型，计算对比指标，并覆盖写入 abl_prophet.md。

每个数据集的完整单项消融包含以下 6 种配置（未列出的开关均为 True）：
  1. Full          (proto=T, spatio=T, temporal=T, adj=T)
  2. w/o P         (proto=F；去掉整个原型模块，embeddings 直接输入 MLP)
  3. -Spatio       (proto=T, spatio=F, temporal=T, adj=T)
  4. -Temporal     (proto=T, spatio=T, temporal=F, adj=T)
  5. -Adj          (proto=T, spatio=T, temporal=T, adj=F)
  6. -His          (his=F)
"""

import json
import os
from pathlib import Path
from utils.paths import PROJECT_ROOT

# ========== 配置 ==========
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_MD = PROJECT_ROOT / "abl_prophet.md"

# 实验标记 → 人类可读名称
ABL_LABELS = {
    "full":          "Full",
    "minus_proto":   "w/o P",
    "minus_spatio":  "-Spatio",
    "minus_temporal":"-Temporal",
    "minus_adj":     "-Adj",
    "minus_his":     "-His",
}

# 指标列表
METRICS = ["MAE", "RMSE", "MAPE", "SMAPE", "R2"]

# 数据集列表（按需调整顺序）
DATASETS = ["chi_taxi", "cq", "hz", "nyc_taxi", "nyc_bike"]

# 报告使用当前名称；实验目录兼容更名前后的名称。
MODEL_NAME = "PSTID"
MODEL_ALIASES = (MODEL_NAME, "PSTID")

# ========== 辅助函数 ==========

def identify_abl_type(config: dict) -> str | None:
    """根据 config.json 判断消融类型"""
    proto     = config.get("use_proto", True)
    spatio    = config.get("use_spatio", True)
    temporal  = config.get("use_temporal", True)
    adj       = config.get("use_adj", True)
    his       = config.get("use_his", True)

    # 优先按最关键的 proto/spatio/temporal 消融来分类
    if not proto:
        return "minus_proto"
    if not spatio:
        return "minus_spatio"
    if not temporal:
        return "minus_temporal"
    # proto/spatio/temporal 全启用，按 adj 区分
    if not adj:
        return "minus_adj"
    if not his:
        return "minus_his"
    # 所有组件都启用 → Full；旧配置缺少 use_his 时默认启用
    if proto and spatio and temporal and adj and his:
        return "full"
    return None


def load_results(dataset: str) -> dict[str, dict]:
    """
    加载指定数据集下所有 PROPHET / PSTID 实验的结果。
    返回: {abl_type: {"config": ..., "metrics": ..., "exp_dir": ...}, ...}
    """
    base = RESULTS_ROOT / dataset
    results = {}

    prefixes = tuple(f"{name}_" for name in MODEL_ALIASES)
    # 按时间戳排序，同类消融保留最新结果，不受模型更名影响。
    for exp_dir in sorted(base.iterdir(), key=lambda path: (path.name.partition("_")[2], path.name)):
        if not exp_dir.is_dir() or not exp_dir.name.startswith(prefixes):
            continue

        config_path = exp_dir / "config.json"
        metrics_path = exp_dir / "test_results.json"

        if not config_path.exists() or not metrics_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)
        # Historical PSTID names remain aliases; the independent fork does not.
        if config.get("model_variant", "").startswith("pstid_fork_"):
            continue
        if not config.get("use_proto", True) and config.get("proto_ablation") != "direct_embeddings_mlp":
            print(f"  [警告] {dataset}/{exp_dir.name} 缺少当前直接输入 MLP 的实验标记，"
                  "不纳入 w/o P 对照，请重新训练 --use_proto 0")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        abl_type = identify_abl_type(config)
        if abl_type is None:
            print(f"  [警告] {dataset}/{exp_dir.name} 无法识别消融类型，跳过")
            continue

        results[abl_type] = {
            "config": config,
            "metrics": metrics,
            "exp_dir": exp_dir.name,
        }

    return results


def pct(old: float, new: float) -> float:
    """计算相对变化百分比"""
    if old == 0:
        return 0.0
    return (new - old) / old * 100


# ========== 主流程 ==========

def main():
    all_data = {}

    for dataset in DATASETS:
        results = load_results(dataset)
        all_data[dataset] = results

        n = len(results)
        print(f"[{dataset}] 加载了 {n} 个实验: {list(results.keys())}")

        # 检查完整模型和各单项消融是否齐全
        expected = set(ABL_LABELS)
        found = set(results.keys())
        missing = expected - found
        if missing:
            print(f"  [警告] 缺少以下消融配置: {missing}")
        if found - expected:
            print(f"  [警告] 发现未知配置: {found - expected}")

    # ========== 构建 Markdown ==========

    lines = []
    lines.append(f"# {MODEL_NAME} 消融实验结果\n")
    lines.append("## 创新点说明\n")
    lines.append("PROPHET 在 STID 的基础上，通过原型模块增强输入嵌入：")
    lines.append("- **核心思想**：STID 的 TID (Time-In-Day) 和 DIW (Day-In-Week) 嵌入以及空间节点嵌入可能产生噪声")
    lines.append("- **解决方案**：通过码本模块，将节点嵌入归类成几个空间码本信息，将时间嵌入归类成几个时间码本信息")
    lines.append("- **预期效果**：极大减少噪声并提升模型的可解释性\n")

    lines.append("## 实验设置\n")
    lines.append("- **数据集**：`chi_taxi`、`cq`、`hz`、`jl`、`nyc_taxi`、`sh`、`yt`")
    lines.append(f"- **模型**：{MODEL_NAME}")
    lines.append("- **Baseline**：Full（启用 Proto + Spatio + Temporal 全部码本）\n")

    lines.append("## 消融实验配置\n")
    lines.append("| 标记 | 含义 | 验证假设 |")
    lines.append("|------|------|---------|")
    lines.append("| Full | 全部组件启用 | 基准性能 |")
    lines.append("| w/o P | 去掉整个原型模块，拼接后的 embeddings 直接输入 MLP | 整个原型模块的作用 |")
    lines.append("| -Spatio | 关闭空间码本 | 空间码本的作用 |")
    lines.append("| -Temporal | 关闭时间码本 | 时间码本的作用 |")
    lines.append("| -Adj | 关闭 adj 矩阵衍生的空间嵌入 | 图结构信息的作用 |")
    lines.append("| -His | 时间、空间原型输入均去掉历史序列嵌入 | 历史序列对原型分配和 Query 特征的作用 |")
    lines.append("")

    lines.append("## 绝对指标对比\n")
    lines.append("| 数据集 | 实验 | MAE | RMSE | MAPE | SMAPE | R² |")
    lines.append("|--------|------|-----:|-----:|-----:|------:|----:|")

    for dataset in DATASETS:
        res = all_data[dataset]
        for i, (key, label) in enumerate(ABL_LABELS.items()):
            if key not in res:
                continue
            m = res[key]["metrics"]
            sep = "**" if i == 0 else ""
            lines.append(
                f"| {sep}{dataset}{sep} | {label} "
                f"| {m['MAE']:.4f} | {m['RMSE']:.4f} "
                f"| {m['MAPE']:.2f} | {m['SMAPE']:.2f} | {m['R2']:.4f} |"
            )
    lines.append("")

    # ---- 1. 相对 Full 的性能变化 ----
    lines.append("## 相对 Full 的性能变化（%）\n")
    lines.append(
        "| 数据集 | 实验 | ΔMAE(%) | ΔRMSE(%) | ΔMAPE(%) | ΔSMAPE(%) | ΔR²(%) |"
    )
    lines.append(
        "|--------|------|--------:|----------:|----------:|----------:|--------:|"
    )

    for dataset in DATASETS:
        res = all_data[dataset]
        full = res.get("full", {}).get("metrics")
        if not full:
            continue

        for key, label in ABL_LABELS.items():
            if key == "full":
                continue  # 跳过 Full 自身
            m = res.get(key, {}).get("metrics")
            if not m:
                continue

            d_mae  = (m["MAE"] - full["MAE"]) / full["MAE"] * 100
            d_rmse = (m["RMSE"] - full["RMSE"]) / full["RMSE"] * 100
            d_mape = (m["MAPE"] - full["MAPE"]) / full["MAPE"] * 100
            d_smap = (m["SMAPE"] - full["SMAPE"]) / full["SMAPE"] * 100
            d_r2   = (m["R2"] - full["R2"]) / abs(full["R2"]) * 100

            lines.append(
                f"| {dataset} | {label} "
                f"| {d_mae:>+0.2f} | {d_rmse:>+0.2f} "
                f"| {d_mape:>+0.2f} | {d_smap:>+0.2f} | {d_r2:>+0.2f} |"
            )

    lines.append("")

    # ---- 3. 整个原型模块与单组件消融的变化对比（ΔMAE）----
    lines.append("## 整个原型模块与单组件消融的变化对比（ΔMAE）\n")
    lines.append(
        "| 数据集 | ΔSpatio | ΔTemporal | ΔAdj | S+T+A 简单加和 | ΔP | "
        "P/(S+T+A) 比值 |"
    )
    lines.append(
        "|--------|--------:|----------:|------:|---------------:|-------:|"
        "------------------:|"
    )

    for dataset in DATASETS:
        res = all_data[dataset]
        full = res.get("full", {}).get("metrics")
        minus_s = res.get("minus_spatio", {}).get("metrics")
        minus_t = res.get("minus_temporal", {}).get("metrics")
        minus_a = res.get("minus_adj", {}).get("metrics")
        minus_p = res.get("minus_proto", {}).get("metrics")

        if not all([full, minus_s, minus_t, minus_a, minus_p]):
            lines.append(f"| {dataset} | — | — | — | — | — | — |")
            continue

        d_s = minus_s["MAE"] - full["MAE"]
        d_t = minus_t["MAE"] - full["MAE"]
        d_a = minus_a["MAE"] - full["MAE"]
        d_p = minus_p["MAE"] - full["MAE"]
        s_t_a = d_s + d_t + d_a
        ratio = d_p / s_t_a if s_t_a != 0 else 0.0

        lines.append(
            f"| {dataset} "
            f"| {d_s:>+0.4f} "
            f"| {d_t:>+0.4f} "
            f"| {d_a:>+0.4f} "
            f"| {s_t_a:>+0.4f} "
            f"| {d_p:>+0.4f} "
            f"| **{ratio:.2f}×** |"
        )
    lines.append("")

    # ---- 4. 结论分析 ----
    lines.append("## 结论与分析\n")
    lines.append("### 假设验证\n")
    lines.append("1. **原型模块效果**：通过比较 Full 与 w/o P 的性能差异，检验整个原型模块的收益")
    lines.append("2. **空间码本作用**：通过比较 Full 与 -Spatio 的性能差异，验证空间码本对节点嵌入的归类效果")
    lines.append("3. **时间码本作用**：通过比较 Full 与 -Temporal 的性能差异，验证时间码本对时间嵌入的归类效果")
    lines.append("4. **图结构信息作用**：通过比较 Full 与 -Adj 的性能差异，验证 adj 矩阵衍生的空间嵌入是否有效")
    lines.append("5. **历史序列作用**：通过比较 Full 与 -His 的性能差异，验证原型输入中历史序列嵌入的贡献\n")

    lines.append("### 预期结果\n")
    lines.append("- **Full 应优于 STID**：原型模块应能减少原始嵌入中的噪声")
    lines.append("- **Full 与 w/o P 对照**：由重新训练的指标判断原型增强的效果")
    lines.append("- **Spatio + Temporal 应接近 Full**：两个码本的联合效果应接近完整的原型模块\n")

    lines.append("说明：\n")
    lines.append("- MAE/RMSE/MAPE/SMAPE 的正值表示消融后性能下降（变差）\n")
    lines.append("- R² 的负值表示消融后拟合质量变差\n")
    lines.append("- P/(S+T+A) 仅作描述性比较：去掉整个原型模块与关闭单个组件是不同操作，不能将此比值直接解释为贡献加和性")
    lines.append("- 表中 w/o P 指 embeddings 直接输入 MLP；仅纳入带 direct_embeddings_mlp 标记的实验，旧版结果需重新训练")

    # ========== 写入文件 ==========
    content = "\n".join(lines)
    OUTPUT_MD.write_text(content, encoding="utf-8")
    print(f"\n已写入: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
