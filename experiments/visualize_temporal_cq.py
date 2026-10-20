"""Plot June 8/15/22 Wednesday demand above each day's 10:00 allocations.

Run from the repository root: python -m experiments.visualize_temporal_cq
"""
import argparse
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pstid-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline.dataset import StandardScaler
from data_pipeline.normalization import normalize_input
from models.PROPHET.model import PROPHET
from utils.paths import PROJECT_ROOT


DEFAULT_CHECKPOINT = PROJECT_ROOT / "results_42/cq/PROPHET_20261010_045203/checkpoint/best_model.pth"
COLORS = ["#4C78A8", "#F2A541", "#59A89C", "#B279A2", "#7F8C8D", "#E26D5A", "#9DAD4E", "#76B7D4"]


def load_samples(data_dir, dates):
    meta = json.loads((data_dir / "metadata.json").read_text())
    start, end = pd.Timestamp(meta["time_range"]["start"]), pd.Timestamp(meta["time_range"]["end"])
    timeline = pd.date_range(start, end, freq=meta["time_size"])
    # The CQ pipeline retains the same daily operating hours throughout the data.
    timeline = timeline[(timeline.time >= start.time()) & (timeline.time <= end.time())]
    if len(timeline) != meta["num_timestamps"]:
        raise ValueError("Metadata does not reconstruct the processed time axis.")
    positions = timeline.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Requested timestamps are outside the processed operating hours.")
    window = meta["input_window"]
    indices = positions - window + 1
    if (indices < 0).any() or (indices >= meta["total_samples"]).any():
        raise ValueError("Requested dates are outside the available input windows.")
    with np.load(data_dir / "all.npz") as data:
        all_x = data["X"]
        if len(all_x) != meta["total_samples"]:
            raise ValueError("Sample count differs from metadata.")
        x = all_x[indices].copy()
    histories = [timeline[i:i + window] for i in indices]
    expected = np.array([[(t.hour + t.minute / 60) / 24, t.dayofweek / 7]
                         for history in histories for t in history]).reshape(len(dates), window, 1, 2)
    np.testing.assert_allclose(x[..., 1:], np.broadcast_to(expected, x[..., 1:].shape), atol=1e-7)
    if not all(d.dayofweek == 2 for d in dates):
        raise ValueError("All selected dates must be Wednesdays.")
    return meta, x, indices, histories


def allocations(checkpoint, x):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    if config["data"].lower() != "cq":
        raise ValueError("This visualization requires a CQ checkpoint.")
    model = PROPHET.from_args(config, x.shape[2], None, torch.device("cpu")).eval()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    basis = model.proto_module.temporal_basis
    if basis is None or not basis.use_proto or not basis.use_his:
        raise ValueError("Use a full model with temporal prototypes and historical input enabled.")
    scaler = StandardScaler(ck["scaler_mean"], ck["scaler_std"])
    with torch.inference_mode():
        _, h, tid, diw, _, _ = model._get_embeddings({"X": normalize_input(torch.from_numpy(x), scaler)})
        # Calendar embeddings agree across dates at the same clock time.
        for clock_time in np.unique(x[:, -1, 0, 1]):
            mask = torch.from_numpy(x[:, -1, 0, 1] == clock_time)
            group = tid[mask]
            torch.testing.assert_close(group, group[:1, :1, :1].expand_as(group), rtol=0, atol=0)
        torch.testing.assert_close(diw, diw[:1, :1, :1].expand_as(diw), rtol=0, atol=0)
        inputs = torch.cat([h[:, -1], tid[:, -1], diw[:, -1]], dim=-1)
        alpha = torch.softmax(basis.allocator(inputs.reshape(-1, inputs.shape[-1])) / basis.temp, -1)
        alpha = alpha.reshape(x.shape[0], x.shape[2], -1)
        # Check against the actual temporal forward path, including its temperature.
        enhanced, _ = basis(h, tid, diw)
        torch.testing.assert_close(enhanced[:, 0], alpha @ basis.prototypes)
        if not torch.isfinite(alpha).all() or (alpha < 0).any():
            raise ValueError("Invalid allocation probabilities.")
        torch.testing.assert_close(alpha.sum(-1), torch.ones(alpha.shape[:2]))
    return ck, alpha.numpy(), float(basis.temp.detach())


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def export_wednesday_columns(output, paper_output, days, hours, demand, alpha,
                             node, histories, indices, checkpoint, seed, temperature):
    """Full observed days above allocations computed from history ending at 10:00."""
    selected_slot = np.flatnonzero(np.isclose(hours, 10.))
    if len(selected_slot) != 1:
        raise ValueError("Expected exactly one 10:00 slot.")
    slot = int(selected_slot[0])
    curves, weights = demand[:, :, node], alpha[:, slot, node]
    ymax = max(10., float(np.ceil(curves.max() * 1.1 / 10) * 10))
    daily_rows, allocation_rows = [], []
    for day_index, day in enumerate(days):
        for clock_index, hour in enumerate(hours):
            daily_rows.append({"date": str(day.date()), "node_id": node,
                               "hour": float(hour), "demand": float(curves[day_index, clock_index])})
        sample = day_index * len(hours) + slot
        w = weights[day_index]
        allocation_rows.append({
            "date": str(day.date()), "time": "10:00", "node_id": node,
            "sample_index": int(indices[sample]),
            "history_start": str(histories[sample][0]), "history_end": str(histories[sample][-1]),
            "dominant_prototype": int(w.argmax()), "dominant_weight": float(w.max()),
            **{f"alpha_P{k+1}": float(v) for k, v in enumerate(w)},
        })
    manifest = {
        "selected_node": node, "dates": [str(d.date()) for d in days],
        "allocation_time": "10:00", "hours": hours.tolist(), "slots_per_day": len(hours),
        "demand_y_limits": [0, ymax], "daily_totals": curves.sum(axis=1).tolist(),
        "checkpoint": str(checkpoint.resolve()), "seed": seed, "temperature": temperature,
        "selection_rule": "User-specified dates and 10:00; retained node 44 unless --node overrides it. No allocation-based screening.",
        "layout": "2 rows x 3 columns; full Wednesday demand above its 10:00 allocation",
        "history_window": len(histories[0]), "split": "train",
        "semantics": "Top panels show Wednesday observations only, including slots after 10:00 for context. Bottom allocations use only 24 valid historical slots ending at 10:00, including the previous day; no future demand is used.",
        "checks": ["strict checkpoint load", "historical timestamps and calendar features",
                   "same-clock calendar embeddings", "probabilities sum to one",
                   "alpha @ prototypes equals temporal forward"],
    }
    daily, allocations_frame = pd.DataFrame(daily_rows), pd.DataFrame(allocation_rows)
    for folder in {output, paper_output}:
        folder.mkdir(parents=True, exist_ok=True)
        daily.to_csv(folder / "wednesday_columns_demand.csv", index=False)
        allocations_frame.to_csv(folder / "wednesday_columns_allocations.csv", index=False)
        (folder / "wednesday_columns_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    colors = ["#4C78A8", "#66AB90", "#DB795E"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), layout="constrained")
    for col, day in enumerate(days):
        ax = axes[0, col]
        ax.plot(hours, curves[col], color=colors[col], lw=1.8, marker="o", ms=3)
        ax.fill_between(hours, curves[col], color=colors[col], alpha=.18)
        ax.axvline(10, color=".4", ls="--", lw=1)
        ax.scatter([10], [curves[col, slot]], color=colors[col], edgecolor="black", s=36, zorder=3)
        ax.set(xlim=(hours[0], hours[-1]), ylim=(0, ymax),
               xticks=[8, 10, 14, 18, 20.5], xticklabels=["08:00", "10:00", "14:00", "18:00", "20:30"],
               xlabel="Wednesday time", title=f"{day:%Y-%m-%d} | Wednesday")
        ax.tick_params(axis="x", labelsize=8)
        if col == 0:
            ax.set_ylabel("Orders / 30 min")
        w = weights[col]
        ax = axes[1, col]
        bar_colors = [colors[col] if k == w.argmax() else "#c8ced5" for k in range(len(w))]
        ax.bar(np.arange(len(w)), w, color=bar_colors, width=.68)
        for k, value in enumerate(w):
            if value >= .05:
                ax.text(k, value + .025, f"{value:.1%}", ha="center", fontsize=9)
        ax.set(ylim=(0, 1), xticks=np.arange(len(w)), xticklabels=[f"T{k}" for k in range(len(w))],
               xlabel="Temporal prototype", title=f"10:00 | Dominant T{w.argmax()} ({w.max():.1%})")
        if col == 0:
            ax.set_ylabel("Allocation weight")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"CQ node {node} | Full Wednesday demand and 10:00 prototype allocation", fontsize=15)
    for folder in {output, paper_output}:
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"cq_wednesday_1000_comparison_preview.{ext}", dpi=200)
    plt.close(fig)
    readme = f"""# 三个周三的全天需求与上午 10:00 原型分配

节点 {node}；列顺序为 2022-06-08、2022-06-15、2022-06-22。
上排展示各周三 08:00–20:30 的全部 26 个有效观测槽，采用统一纵轴和同色填充。
下排展示同列日期 10:00 的原始分配，突出主导原型，权重纵轴统一为 0–1。
需求图的竖虚线标记 10:00。分配只使用截至 10:00 的 24 步历史，可能包含前一天；
上排 10:00 之后的观测仅提供全天背景，不是该时刻的模型输入。

![预览](cq_wednesday_1000_comparison_preview.png)

`wednesday_columns_demand.csv`、`wednesday_columns_allocations.csv`、
`wednesday_columns_manifest.json` 为可移植绘图输入，已同步到 `{paper_output}`。
MATLAB 运行 `figs/temporal_cq/state_comparison.m`，输出论文样式版本。

复现：`python -m experiments.visualize_temporal_cq`。
日期和时刻由用户指定；这次不按原型差异筛选日期或钟点。
历史案例搜索可通过 `--state-case-search` 单独运行。
"""
    (output / "README.md").write_text(readme)
    print(allocations_frame[["date", "time", "dominant_prototype", "dominant_weight"]].to_string(index=False))
    print(f"Saved 2 x 3 figure and portable inputs to {output} and {paper_output}")


def rank_state_cases(x, alpha, days, hours, min_mean=5., min_mae=5., min_confidence=.6):
    """Rank illustrative cases, explicitly screening BOTH inputs and outputs.

    Cases keep the node, weekday and clock fixed. Output screening makes this
    an explanatory case search, not evidence of population-wide association.
    """
    n_days, n_slots, n_nodes, _ = alpha.shape
    history = x[..., 0].reshape(n_days, n_slots, x.shape[1], n_nodes)
    records = []
    for slot, hour in enumerate(hours):
        for node in range(n_nodes):
            for a, b in combinations(range(n_days), 2):
                ha, hb = history[a, slot, :, node].astype(float), history[b, slot, :, node].astype(float)
                wa, wb = alpha[a, slot, node], alpha[b, slot, node]
                ma, mb = ha.mean(), hb.mean()
                mae = np.abs(ha - hb).mean()
                confidence = min(float(wa.max()), float(wb.max()))
                changed = wa.argmax() != wb.argmax()
                input_ok = min(ma, mb) >= min_mean and mae >= min_mae
                records.append({
                    "node_id": node, "clock_index": slot,
                    "time": f"{int(hour):02d}:{int(hour % 1 * 60):02d}",
                    "day_a": a, "day_b": b,
                    "date_a": str(days[a].date()), "date_b": str(days[b].date()),
                    "history_mean_a": ma, "history_mean_b": mb,
                    "history_mae": mae,
                    "history_relative_mae": mae / ((ma + mb) / 2 + 1),
                    "allocation_tv": float(np.abs(wa - wb).sum() / 2),
                    "dominant_a": int(wa.argmax()), "dominant_b": int(wb.argmax()),
                    "min_dominant_weight": confidence,
                    "demand_eligible": input_ok,
                    "illustration_eligible": input_ok and changed and confidence >= min_confidence,
                })
    return pd.DataFrame(records).sort_values(
        ["illustration_eligible", "history_relative_mae", "node_id", "clock_index", "day_a", "day_b"],
        ascending=[False, False, True, True, True, True], kind="stable").reset_index(drop=True)


def export_state_case(output, paper_output, ranking, case, x, alpha, days, hours,
                      histories, indices, checkpoint, seed, temperature, thresholds):
    """Export the two actual model inputs and their full probability vectors."""
    node, slot = int(case.node_id), int(case.clock_index)
    day_indices = [int(case.day_a), int(case.day_b)]
    day_indices.sort(key=lambda d: float(x[d * len(hours) + slot, :, node, 0].mean()))
    records = []
    for label, day in zip(["Lower historical demand", "Higher historical demand"], day_indices):
        index = day * len(hours) + slot
        h, w = x[index, :, node, 0], alpha[day, slot, node]
        records.append({
            "state": label, "date": str(days[day].date()), "time": case.time,
            "node_id": node, "sample_index": int(indices[index]),
            "history_start": str(histories[index][0]), "history_end": str(histories[index][-1]),
            "history_mean": float(h.mean()), "history_peak": float(h.max()),
            "dominant_prototype": int(w.argmax()),
            **{f"history_lag{len(h)-1-i}": float(v) for i, v in enumerate(h)},
            **{f"alpha_P{k+1}": float(v) for k, v in enumerate(w)},
        })
    frame = pd.DataFrame(records)
    tick_positions = [0, 6, 12, 18, 23]
    times = histories[day_indices[0] * len(hours) + slot]
    tick_labels = [("Tue " if t.dayofweek == 1 else "Wed ") + t.strftime("%H:%M")
                   for t in times[tick_positions]]
    boundary = next(i for i, t in enumerate(times) if t.date() == times[-1].date())
    selected = ranking[ranking.node_id == node]
    manifest = {
        "selected_node": node, "selected_time": case.time,
        "dates": frame.date.tolist(), "history_window": x.shape[1],
        "history_tick_indices": [i + 1 for i in tick_positions],
        "history_tick_labels": tick_labels,
        "current_day_start_index": boundary + 1,
        "checkpoint": str(checkpoint.resolve()), "seed": seed, "temperature": temperature,
        "thresholds": thresholds,
        "selection_rule": "Same-node, same-clock Wednesday pairs. Require both history means >= min_mean, history MAE >= min_mae, different dominant prototypes and both dominant weights >= min_confidence. Rank eligible cases by MAE / (pair mean + 1), descending; ties by node, clock, dates.",
        "selection_uses_allocations": True,
        "scope": "Illustrative case selected using both history contrast and allocation switch; not an unbiased association test or globally identified prototype semantics.",
        "candidate_pairs": len(ranking),
        "eligible_pairs": int(ranking.illustration_eligible.sum()),
        "node_eligible_pairs": int(selected.illustration_eligible.sum()),
        "history_mae": float(case.history_mae),
        "allocation_tv": float(case.allocation_tv),
        "split": "train",
        "checks": ["strict checkpoint load", "full history timestamps and calendar features",
                   "same clock and weekday embeddings", "probabilities sum to one",
                   "alpha @ prototypes equals temporal forward"],
    }
    nodes = ranking.groupby("node_id").agg(
        demand_eligible_pairs=("demand_eligible", "sum"),
        illustration_eligible_pairs=("illustration_eligible", "sum"),
        mean_allocation_tv=("allocation_tv", "mean"))
    best = ranking[ranking.illustration_eligible].groupby("node_id").history_relative_mae.max()
    nodes["best_eligible_history_score"] = best
    for folder in {output, paper_output}:
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "state_case_inputs.csv", index=False)
        ranking.to_csv(folder / "state_case_candidates.csv", index=False)
        nodes.to_csv(folder / "state_case_nodes.csv")
        (folder / "state_case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    # Separate states, common axes: the input curve and its output share a row.
    colors = ["#4C78A8", "#DB795E"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), gridspec_kw={"width_ratios": [1.5, 1]},
                             layout="constrained")
    hcols = [f"history_lag{k}" for k in range(x.shape[1]-1, -1, -1)]
    history_values = frame[hcols].to_numpy()
    ymax = np.ceil(history_values.max() * 1.1 / 10) * 10
    lags = np.arange(1 - x.shape[1], 1)
    for row, record in frame.iterrows():
        h = history_values[row]
        w = record[[f"alpha_P{k}" for k in range(1, 9)]].to_numpy(float)
        ax = axes[row, 0]
        ax.plot(lags, h, color=colors[row], lw=1.8, marker="o", ms=3)
        ax.fill_between(lags, h, color=colors[row], alpha=.18)
        if boundary > 0:
            ax.axvline(lags[boundary] - .5, color=".6", ls=":", lw=1)
        ax.axhline(record.history_mean, color=colors[row], ls="--", alpha=.75, lw=1)
        ax.set(xlim=(lags[0], 0), ylim=(0, ymax), ylabel="Orders / 30 min",
               xticks=lags[tick_positions], xticklabels=tick_labels,
               title=f"{record.state} | {record.date}\nMean {record.history_mean:.1f} / peak {record.history_peak:.0f}")
        ax.tick_params(axis="x", labelsize=8)
        ax = axes[row, 1]
        bar_colors = [colors[row] if k == w.argmax() else "#c8ced5" for k in range(8)]
        ax.bar(np.arange(8), w, color=bar_colors, width=.68)
        for k, value in enumerate(w):
            if value >= .01:
                ax.text(k, value + .025, f"{value:.1%}", ha="center", fontsize=9)
        ax.set(ylim=(0, 1), xticks=np.arange(8), xticklabels=[f"T{k}" for k in range(8)],
               ylabel="Allocation weight", title=f"Dominant prototype: T{w.argmax()} ({w.max():.1%})")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("24 valid historical slots (overnight gap omitted)")
    axes[1, 1].set_xlabel("Temporal prototype")
    fig.suptitle(f"CQ node {node} | Same weekday and clock: Wednesday {case.time}", fontsize=16)
    for folder in {output, paper_output}:
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"cq_demand_state_comparison_preview.{ext}", dpi=200)
    plt.close(fig)
    return manifest


def plot_day_comparison(output, days, hours, demand, alpha, node, seed):
    """Use exactly the same demand limits/ticks and probability scale for all days."""
    curves = demand[:, :, node]
    weights = alpha[:, :, node, :]
    ymax = max(10, float(np.ceil(curves.max() * 1.1 / 10) * 10))
    yticks = np.arange(0, ymax + 1, 10)
    day_colors = ["#377EB8", "#E36A47", "#329D83"]
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("CQ · 三个周三的全天订单与时间原型分配", x=.065, y=.965,
                 ha="left", fontsize=21, weight="bold")
    fig.text(.065, .915, f"2022 年 6 月 15 / 22 / 29 日 · 08:00–20:00 · 固定节点 {node} · PROPHET seed={seed}",
             fontsize=12, color="#52606D")
    gs = fig.add_gridspec(3, 2, left=.065, right=.92, top=.82, bottom=.15,
                         width_ratios=[1.05, 1.35], hspace=.45, wspace=.24)
    curve_axes, heat_axes = [], []
    for i, day in enumerate(days):
        ax = fig.add_subplot(gs[i, 0], sharey=curve_axes[0] if curve_axes else None)
        curve_axes.append(ax)
        ax.plot(hours, curves[i], color=day_colors[i], lw=2, marker="o", markersize=3)
        ax.fill_between(hours, curves[i], color=day_colors[i], alpha=.10)
        ax.set(xlim=(8, 20), ylim=(0, ymax), xticks=np.arange(8, 21, 2), yticks=yticks)
        ax.set_xticklabels([f"{h:02d}:00" for h in range(8, 21, 2)])
        ax.set_ylabel("订单量（单 / 30 分钟）", fontsize=10)
        ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{day:%m-%d}  周三   |   所示时间槽合计 {curves[i].sum():.0f} 单",
                     loc="left", fontsize=12, pad=10, color=day_colors[i], weight="bold")
        ax = fig.add_subplot(gs[i, 1]); heat_axes.append(ax)
        im = ax.imshow(weights[i].T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                       extent=[7.75, 20.25, .5, weights.shape[-1] + .5], interpolation="nearest")
        ax.set_xticks(np.arange(8, 21, 2), [f"{h:02d}:00" for h in range(8, 21, 2)])
        ax.set_yticks(np.arange(1, weights.shape[-1] + 1), [f"P{k+1}" for k in range(weights.shape[-1])])
        ax.set_title(f"{day:%m-%d}  各时刻的分配权重 α", loc="left", fontsize=12, pad=10)
    for ax in [curve_axes[-1], heat_axes[-1]]:
        ax.set_xlabel("周三当天时刻")
    cax = fig.add_axes([.94, .15, .013, .67])
    fig.colorbar(im, cax=cax, label="分配权重（统一 0–1）")
    fig.text(.065, .065, f"三幅订单曲线共用 y 轴范围 0–{ymax:g}，刻度间隔 10 单；右侧三幅热图共用权重色标 0–1。", fontsize=10, color="#52606D")
    fig.text(.065, .03, "左侧仅展示周三当天订单；右侧为各时刻真实模型分配，模型仍使用截至该时刻的 24 步历史输入。", fontsize=10, color="#52606D")
    for ax in curve_axes:
        np.testing.assert_array_equal(ax.get_ylim(), [0, ymax])
        np.testing.assert_array_equal(ax.get_yticks(), yticks)
    save_figure(fig, output, "cq_wednesday_day_comparison")

    # Overlay makes differences at the same time directly visible.
    fig, ax = plt.subplots(figsize=(11, 5.7))
    for i, day in enumerate(days):
        ax.plot(hours, curves[i], color=day_colors[i], lw=2, marker="o", markersize=4,
                label=f"{day:%m-%d}（{curves[i].sum():.0f} 单）")
    ax.set(xlim=(8, 20), ylim=(0, ymax), xticks=np.arange(8, 21), yticks=yticks,
           ylabel="订单量（单 / 30 分钟）", xlabel="周三当天时刻")
    ax.set_xticklabels([f"{h:02d}:00" for h in range(8, 21)])
    ax.set_title(f"CQ · 节点 {node} · 三个周三全天订单对比（2022 年）", fontsize=16, pad=18, weight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=2)
    save_figure(fig, output, "cq_wednesday_day_overlay")
    return ymax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--paper-output", type=Path, default=PROJECT_ROOT / "figs/temporal_cq")
    parser.add_argument("--node", type=int, default=None, help="Default: node 44 for Wednesday columns; all nodes for case search.")
    parser.add_argument("--state-case-search", action="store_true", help="Run the earlier automatic paired-state search instead of the three Wednesday columns.")
    parser.add_argument("--min-history-mean", type=float, default=5.)
    parser.add_argument("--min-history-mae", type=float, default=5.)
    parser.add_argument("--min-dominant-weight", type=float, default=.6)
    args = parser.parse_args()
    if args.output is None:
        args.output = PROJECT_ROOT / "可视化结果" / ("时间基CQ_需求状态对比" if args.state_case_search else "时间基CQ_三个周三十点")
    if args.min_history_mean < 0 or args.min_history_mae < 0 or not 0 <= args.min_dominant_weight <= 1:
        parser.error("History thresholds must be nonnegative; dominant weight must be in [0, 1].")
    torch.set_num_threads(2)
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "pdf.fonttype": 42})
    days = pd.DatetimeIndex(["2022-06-15", "2022-06-22", "2022-06-29"] if args.state_case_search
                            else ["2022-06-08", "2022-06-15", "2022-06-22"])
    hours = np.arange(8, 20.01 if args.state_case_search else 20.51, .5)
    timestamps = pd.DatetimeIndex([day + pd.Timedelta(hours=float(hour)) for day in days for hour in hours])
    meta, x, indices, histories = load_samples(PROJECT_ROOT / "data/processed/cq", timestamps)
    ck, raw_alpha, temperature = allocations(args.checkpoint, x)
    if args.node is not None and not 0 <= args.node < x.shape[2]:
        raise ValueError("Node ID is outside the processed dataset.")
    # The final demand observation belongs to the plotted Wednesday time slot;
    # never plot the rest of the historical input as that day's observations.
    demand = x[:, -1, :, 0].reshape(len(days), len(hours), x.shape[2])
    alpha = raw_alpha.reshape(len(days), len(hours), x.shape[2], -1)
    if not args.state_case_search:
        node = 44 if args.node is None else args.node
        export_wednesday_columns(args.output, args.paper_output, days, hours, demand, alpha,
                                 node, histories, indices, args.checkpoint, ck["config"]["seed"], temperature)
        return
    ranking = rank_state_cases(x, alpha, days, hours, args.min_history_mean,
                               args.min_history_mae, args.min_dominant_weight)
    eligible = ranking[ranking.illustration_eligible]
    if args.node is not None:
        eligible = eligible[eligible.node_id == args.node]
    if eligible.empty:
        args.output.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(args.output / "state_case_candidates.csv", index=False)
        raise ValueError("No case meets the declared state-contrast criteria. Candidate table saved; inspect it before changing thresholds.")
    case = eligible.iloc[0]
    node = int(case.node_id)
    state_manifest = export_state_case(
        args.output, args.paper_output, ranking, case, x, alpha, days, hours,
        histories, indices, args.checkpoint, ck["config"]["seed"], temperature,
        {"min_mean": args.min_history_mean, "min_mae": args.min_history_mae,
         "min_confidence": args.min_dominant_weight})
    assert all(t.dayofweek == 2 and 8 <= t.hour + t.minute / 60 <= 20 for t in timestamps)
    np.testing.assert_allclose(x[:, -1, 0, 1], np.tile(hours / 24, len(days)), atol=1e-7)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, timestamp in enumerate(timestamps):
        for n in range(x.shape[2]):
            rows.append({"timestamp": str(timestamp), "date": str(timestamp.date()), "time": timestamp.strftime("%H:%M"),
                         "sample_index": int(indices[i]), "node_id": n, "demand": float(x[i, -1, n, 0]),
                         "history_start": str(histories[i][0]), "history_end": str(histories[i][-1]),
                         "dominant_prototype": int(raw_alpha[i, n].argmax()) + 1,
                         **{f"alpha_P{k+1}": float(v) for k, v in enumerate(raw_alpha[i, n])}})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_nodes_day_allocations.csv", index=False)
    selected = frame[frame.node_id == node]
    selected.to_csv(args.output / "selected_node_day_allocations.csv", index=False)
    pd.DataFrame(demand[:, :, node].T, index=[f"{int(h):02d}:{int(h % 1 * 60):02d}" for h in hours],
                 columns=[f"{day:%Y-%m-%d}" for day in days]).to_csv(args.output / "three_wednesdays_demand.csv", index_label="time")
    np.savez_compressed(args.output / "day_allocations.npz", timestamps=timestamps.to_numpy(dtype="datetime64[s]"),
                        hours=hours, sample_indices=indices, day_demand=demand, alpha=alpha, selected_node=node)
    ymax = plot_day_comparison(args.output, days, hours, demand, alpha, node, ck["config"]["seed"])
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "seed": ck["config"]["seed"],
                "temperature": temperature, "selected_node": node, "node_position": meta["node_positions"][node],
                "node_selection": "ranked illustrative state contrast" if args.node is None else "state contrast within requested node",
                "state_case": state_manifest,
                "dates": [str(d.date()) for d in days], "clock_range": ["08:00", "20:00"],
                "slots_per_day": len(hours), "slot_semantics": "30-minute left-labeled bins, including the 20:00 bin",
                "demand_y_limits": [0, ymax], "demand_y_tick_interval": 10,
                "daily_displayed_slot_totals": demand[:, :, node].sum(1).tolist(),
                "allocation_shape": list(alpha.shape), "split": "train",
                "checks": ["strict checkpoint load", "all historical time features match timestamps",
                           "calendar embeddings match at the same clock time across dates", "probabilities sum to one",
                           "alpha @ prototypes matches temporal forward", "plotted demand uses Wednesday observations only",
                           "all three demand axes have identical limits and ticks"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    totals = demand[:, :, node].sum(1)
    table = "\n".join(f"| {day:%Y-%m-%d} | {total:.0f} |" for day, total in zip(days, totals))
    readme = f"""# CQ：不同历史需求状态与时间原型分配

![同一节点和时刻的需求状态对比](cq_demand_state_comparison_preview.png)

主图固定节点 {node}、星期三、{case.time}，逐行展示两个完整历史输入及其真实分配。
历史含 24 个有效时间槽，可能跨越前一天；竖虚线标出当天开始，横虚线表示历史均值。
两行需求纵轴相同，权重纵轴均为 0–1。T0–T7 对应模型原型矩阵第 0–7 行。

在 {len(ranking)} 个同节点、同钟点的日期配对中，先要求两段历史均值均不低于
{args.min_history_mean:g} 单/槽、历史 MAE 不低于 {args.min_history_mae:g} 单/槽；
再要求主导原型不同，且两者的主导权重均不低于 {args.min_dominant_weight:.0%}。
符合条件后按历史 MAE / (两段历史的平均需求 + 1) 降序选择。
全体共有 {int(ranking.illustration_eligible.sum())} 个配对满足条件。
此规则明确使用了需求与分配两方面的信息，属于解释性案例筛选，不是总体统计检验。
“较低/较高历史需求”仅描述这两个输入，不将 T0 等原型定义为全局固定的需求等级。

- `state_case_inputs.csv`：主图使用的两段原始历史和全部权重。
- `state_case_candidates.csv`：所有节点和时刻的候选指标，包括未通过条件的样本。
- `state_case_nodes.csv`：逐节点候选数、符合条件数与最佳历史差异得分。
- `state_case_manifest.json`：阈值、选例规则与检查记录。
- 主图的便携数据与预览同时导出到 `{args.paper_output}`，可运行 `state_comparison.m`。

以下全天图保留三个日期，提供所选节点的上下文：

![全天订单与时间分配](cq_wednesday_day_comparison.png)

![三天订单叠加对比](cq_wednesday_day_overlay.png)

仅使用 2022-06-15、2022-06-22、2022-06-29，选中节点 {node}，使用 seed={ck['config']['seed']} 模型。
左侧需求曲线只取周三当天各时刻的真实订单，所有曲线 y 轴范围统一为 0–{ymax:g}，刻度间隔 10 单。
横轴 08:00–20:00，间隔 30 分钟，共 25 个标记点（含 20:00）。数据槽以左端点命名，因此 20:00 点表示 20:00–20:30 的订单。

右侧是每个周三在每个时刻的真实时间原型分配，三幅热图统一使用 0–1 色标。
每个时刻按原模型使用截至该时刻的 24 个有效时间槽作为历史输入，早晨时刻的内部历史会包含前一天；这些前一天的订单不出现在需求曲线上。
P1–P8 对应模型原型矩阵的第 0–7 行，权重未截断或重新归一化。
这些日期属于训练时段，结果用于模型解释，不代表活动的因果效应。

| 日期 | 所展示的 25 个时间槽订单合计 |
| --- | ---: |
{table}

模型：`{args.checkpoint.resolve()}`，温度 {temperature:.6f}。
来源：`data/processed/cq/all.npz`，使用 checkpoint 自带归一化参数并以 eval 模式提取权重。
验证了时间戳、同一钟点的日历嵌入一致性、概率和、实际时间分支输出，以及三幅曲线的 y 轴范围和刻度一致性。

- `three_wednesdays_demand.csv`：25 行 × 三天订单曲线。
- `selected_node_day_allocations.csv`：所选节点 75 个时刻的需求与完整权重。
- `all_nodes_day_allocations.csv`：全部 137 个节点的数据。
- `day_allocations.npz`：需求形状为 (3, 25, 137)，权重为 (3, 25, 137, 8)。
- PNG 用于预览；同名 PDF 用于排版。

复现：
```bash
python -m experiments.visualize_temporal_cq
```
默认搜索全部节点；可用 `--checkpoint PATH --node 44 --output DIR --paper-output DIR`
调整模型、限定节点和输出目录。阈值通过 `--min-history-mean`、`--min-history-mae`、
`--min-dominant-weight` 显式控制，无符合条件的案例时报告失败，不自动降低门槛。
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved figures and data to {args.output}")


if __name__ == "__main__":
    main()
