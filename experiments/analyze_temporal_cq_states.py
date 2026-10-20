"""Compare complete histories at matched calendar times using saved allocations.

Run from the project root: python -m experiments.analyze_temporal_cq_states
No model inference or changes to the original day comparison are performed.
"""
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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figs/temporal_cq"


def main():
    snapshot = json.loads((OUTPUT / "manifest.json").read_text())
    metadata = json.loads((ROOT / "data/processed/cq/metadata.json").read_text())
    frame = pd.read_csv(OUTPUT / "selected_node_day_allocations.csv")
    node = snapshot["selected_node"]
    window = metadata["input_window"]
    assert window == 24 and frame.node_id.eq(node).all()
    indices = frame.sample_index.to_numpy(dtype=int)
    with np.load(ROOT / "data/processed/cq/all.npz") as source:
        x = source["X"][indices, :, node, :]
    histories = x[:, :, 0]
    np.testing.assert_allclose(histories[:, -1], frame.demand, rtol=0, atol=0)
    start = pd.Timestamp(metadata["time_range"]["start"])
    end = pd.Timestamp(metadata["time_range"]["end"])
    timeline = pd.date_range(start, end, freq=metadata["time_size"])
    timeline = timeline[(timeline.time >= start.time()) & (timeline.time <= end.time())]
    assert len(timeline) == metadata["num_timestamps"]
    for row, index in enumerate(indices):
        stamps = timeline[index:index + window]
        assert stamps[0] == pd.Timestamp(frame.history_start.iloc[row])
        assert stamps[-1] == pd.Timestamp(frame.history_end.iloc[row])
        assert stamps[-1] == pd.Timestamp(frame.timestamp.iloc[row])
        expected = np.array([[(t.hour + t.minute / 60) / 24, t.dayofweek / 7]
                             for t in stamps])
        np.testing.assert_allclose(x[row, :, 1:], expected, atol=1e-7)
    assert all(pd.to_datetime(frame.timestamp).dt.dayofweek == 2)
    alpha_names = [f"alpha_P{i}" for i in range(1, 9)]
    alpha = frame[alpha_names].to_numpy()
    assert np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all()
    np.testing.assert_allclose(alpha.sum(axis=1), 1, atol=1e-6)
    rows = []
    for clock, group in frame.groupby("time", sort=True):
        assert len(group) == 3 and group.date.nunique() == 3
        for i, j in combinations(group.index, 2):
            difference = histories[i] - histories[j]
            rows.append({
                "time": clock, "date_a": frame.date[i], "date_b": frame.date[j],
                "current_demand_gap": abs(frame.demand[i] - frame.demand[j]),
                "history_mae": np.abs(difference).mean(),
                "history_rmse": np.sqrt(np.square(difference).mean()),
                "allocation_tv": np.abs(alpha[i] - alpha[j]).sum() / 2,
                "dominant_changed": int(alpha[i].argmax() != alpha[j].argmax()),
            })
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 75
    # Select from input histories only, without consulting allocation weights.
    scores = pairs.groupby("time", sort=True).history_mae.mean()
    clock = scores.idxmax()  # earliest clock wins an exact tie
    portable = frame.copy()
    history_names = [f"history_lag{lag}" for lag in range(window - 1, -1, -1)]
    portable[history_names] = histories
    portable.to_csv(OUTPUT / "state_comparison_inputs.csv", index=False)
    pairs.to_csv(OUTPUT / "state_comparison_pairs.csv", index=False)
    report = {
        "selected_node": node, "history_window": window,
        "selected_time": clock, "dates": snapshot["dates"],
        "selection_rule": "Maximum mean pairwise 24-step history MAE across the three dates; earliest clock breaks ties. Allocation weights are not used for selection.",
        "selection_scores": scores.to_dict(),
        "num_pairs": len(pairs),
        "pearson_current_gap_tv": pairs.current_demand_gap.corr(pairs.allocation_tv),
        "pearson_history_mae_tv": pairs.history_mae.corr(pairs.allocation_tv),
        "pearson_history_rmse_tv": pairs.history_rmse.corr(pairs.allocation_tv),
        "mean_allocation_tv": pairs.allocation_tv.mean(),
        "dominant_changed_pairs": int(pairs.dominant_changed.sum()),
        "allocation_tv_definition": "0.5 * sum(abs(alpha_a - alpha_b)); range [0, 1]",
        "limitations": "Descriptive, selected training node/dates; overlapping windows and shared samples make pairs dependent. No significance or monotonic/causal claim.",
    }
    (OUTPUT / "state_comparison_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Python preview; MATLAB state_comparison.m produces the paper-style version.
    colors = ["#5c9fd3", "#dc7d7e", "#61b89a"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42})
    selected = frame[frame.time == clock].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), layout="constrained")
    lags = np.arange(1 - window, 1)
    for day, (index, row) in enumerate(selected.iterrows()):
        axes[0].plot(lags, histories[index], color=colors[day], marker="o",
                     markersize=3, linewidth=1.5, label=row.date)
        axes[0].fill_between(lags, histories[index], color=colors[day], alpha=.08)
        axes[1].bar(np.arange(8) + (day - 1) * .24, alpha[index], width=.24,
                    color=colors[day], label=row.date)
    axes[0].set(xlim=(-23, 0), ylim=(0, histories[selected.index].max() * 1.1),
                xticks=[-23, -18, -12, -6, 0], ylabel="Orders / 30 min",
                xlabel=f"History step (0 = Wednesday {clock}; overnight gap omitted)",
                title=f"Same node {node}, weekday and clock time: 24-step histories")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    weight_upper = min(1., np.ceil(alpha[selected.index].max() * 1.1 * 10) / 10)
    axes[1].set(xticks=np.arange(8), xticklabels=[f"T{i}" for i in range(8)],
                ylim=(0, weight_upper), ylabel="Allocation weight", xlabel="Temporal prototype",
                title=f"Allocation at {clock} for each history")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"cq_history_state_comparison_preview.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, column, label in zip(axes, ["current_demand_gap", "history_mae"],
                               ["Current demand absolute difference", "24-step history MAE"]):
        for (a, b), group in pairs.groupby(["date_a", "date_b"]):
            ax.scatter(group[column], group.allocation_tv, s=24, alpha=.7,
                       label=f"{a[5:]} / {b[5:]}")
        ax.set(xlabel=label, ylabel="Allocation total variation distance", ylim=(0, 1),
               title=f"All 75 same-clock pairs | r = {pairs[column].corr(pairs.allocation_tv):.3f}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[1].legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"cq_history_state_diagnostics.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "selection_scores"}, indent=2))


if __name__ == "__main__":
    main()
