"""Fix nodes 49/232, rank contrast nodes, and export data and plots for node X.

python -m experiments.select_radar_node
python -m experiments.select_radar_node --node 131
python -m experiments.select_radar_node --auto
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT


FIXED_NODES = (49, 232)


def read_inputs(folder):
    frame = pd.read_csv(folder / "all_node_demand_profiles.csv")
    snapshot = json.loads((folder / "spatial_attention_per_node.json").read_text())
    if snapshot.get("dataset") != "nyc_taxi" or snapshot.get("seed") != 88:
        raise ValueError("Expected the NYC-Taxi Full / seed 88 snapshot.")
    nodes = list(range(snapshot["num_nodes"]))
    columns = [f"node_{n}" for n in nodes]
    if len(frame) != 48 or not np.allclose(frame.hour, np.arange(48) / 2):
        raise ValueError("Expected 48 half-hour slots in the saved demand profiles.")
    profiles = frame[columns].to_numpy(dtype=float).T
    weights = np.array([snapshot["per_node_attention"][c] for c in columns])
    if weights.shape != (len(nodes), snapshot["num_spatial_prototypes"]):
        raise ValueError("Spatial allocation dimensions differ from the snapshot metadata.")
    for values in (profiles, weights):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Demand profiles and allocations must be finite and nonnegative.")
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
    return frame, profiles, weights, snapshot


def pair_metrics(profiles, weights, a, b):
    x, y = profiles[a], profiles[b]
    centered_x, centered_y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    correlation = float(centered_x @ centered_y / denominator) if denominator > 0 else np.nan
    mae = float(np.abs(x - y).mean())
    return dict(node_a=int(a), node_b=int(b), correlation=correlation,
                mae=mae, relative_mae=mae / ((x.mean() + y.mean()) / 2 + 1),
                allocation_tv=float(np.abs(weights[a] - weights[b]).sum() / 2))


def rank_candidates(profiles, weights, args):
    a, b = FIXED_NODES
    reference_mean = (profiles[a].mean() + profiles[b].mean()) / 2
    reference_range = (np.ptp(profiles[a]) + np.ptp(profiles[b])) / 2
    reference_tv = pair_metrics(profiles, weights, a, b)["allocation_tv"]
    rows = []
    for n in range(len(profiles)):
        if n in FIXED_NODES:
            continue
        pa, pb = [pair_metrics(profiles, weights, fixed, n) for fixed in FIXED_NODES]
        mean, spread = float(profiles[n].mean()), float(np.ptp(profiles[n]))
        row = dict(node_id=n, mean_demand=mean, min_demand=float(profiles[n].min()),
                   max_demand=float(profiles[n].max()), demand_range=spread,
                   mean_ratio=mean / reference_mean, range_ratio=spread / reference_range,
                   correlation_to_49=pa["correlation"], correlation_to_232=pb["correlation"],
                   relative_mae_to_49=pa["relative_mae"], relative_mae_to_232=pb["relative_mae"],
                   allocation_tv_to_49=pa["allocation_tv"], allocation_tv_to_232=pb["allocation_tv"],
                   min_allocation_tv=min(pa["allocation_tv"], pb["allocation_tv"]),
                   allocation_tv_ratio=min(pa["allocation_tv"], pb["allocation_tv"]) / max(reference_tv, 1e-12))
        checks = dict(
            active=mean >= args.min_mean and profiles[n].std() >= .1 * mean and spread > 0,
            comparable_volume=args.min_mean_ratio <= row["mean_ratio"] <= args.max_mean_ratio,
            visible_range=row["range_ratio"] >= args.min_range_ratio,
            different_demand=all(np.isfinite(p["correlation"]) and p["correlation"] <= args.max_demand_corr
                                 and p["relative_mae"] >= args.min_relative_mae for p in (pa, pb)),
            different_radar=row["allocation_tv_ratio"] >= args.min_tv_ratio)
        row.update({f"pass_{key}": bool(value) for key, value in checks.items()})
        row["eligible"] = all(checks.values())
        row["failed_criteria"] = ";".join(key for key, value in checks.items() if not value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["eligible", "min_allocation_tv", "node_id"],
        ascending=[False, False, True], kind="stable").reset_index(drop=True)


def plot_selection(folder, hours, profiles, weights, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4170a6", "#bb5753", "#358b70"]
    markers = ["o", "s", "^"]
    theta = np.linspace(0, 2 * np.pi, weights.shape[1] + 1)
    # All nodes share the same radial scale and the original probabilities.
    radial_max = np.ceil(weights[selected].max() / .02) * .02

    def demand(ax):
        for node, color, marker in zip(selected, colors, markers):
            ax.plot(hours, profiles[node], color=color, marker=marker,
                    markevery=4, markersize=3, linewidth=1.5, label=f"Node {node}")
        ax.set(xlabel="Time of day", ylabel="Mean orders / 30 min",
               title="NYC-Taxi test-average demand", xlim=(0, 23.5))
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(0, 24, 4), [f"{h:02}:00" for h in range(0, 24, 4)])
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)

    def radar(ax):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for node, color, marker in zip(selected, colors, markers):
            ax.plot(theta, np.r_[weights[node], weights[node, 0]], color=color,
                    marker=marker, markersize=3, linewidth=1.5, label=f"Node {node}")
        ax.set_xticks(theta[:-1], [f"S{i}" for i in range(weights.shape[1])])
        ax.set_ylim(0, radial_max)
        ax.set_rlabel_position(11.25)
        ax.tick_params(labelsize=8)
        ax.set_title("Full model spatial allocations", pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=False)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"{name}.{extension}", dpi=250, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(12, 4.8))
    demand(fig.add_subplot(121))
    radar(fig.add_subplot(122, projection="polar"))
    # Keep each tried x so comparisons are not lost when another x is selected.
    save(fig, f"nodes_49_232_{selected[-1]}_comparison")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    demand(ax)
    save(fig, "nyc_taxi_selected_demand_preview")
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "polar"})
    radar(ax)
    save(fig, "nyc_taxi_radar_preview")
    # Refresh the shared preview path used by the visualization documentation.
    import shutil
    for extension in ("png", "pdf"):
        shutil.copyfile(folder / f"nodes_49_232_{selected[-1]}_comparison.{extension}",
                        folder / f"nyc_taxi_demand_radar_preview.{extension}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radar-dir", type=Path, default=PROJECT_ROOT / "figs/radar")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--node", type=int, help="Export and plot [49, 232, NODE]; manual choices may fail automatic criteria.")
    action.add_argument("--auto", action="store_true", help="Export and plot the highest-ranked eligible node.")
    parser.add_argument("--top", type=int, default=20, help="Number of candidate rows printed.")
    parser.add_argument("--min-mean", type=float, default=5.)
    parser.add_argument("--min-mean-ratio", type=float, default=.5)
    parser.add_argument("--max-mean-ratio", type=float, default=2.)
    parser.add_argument("--min-range-ratio", type=float, default=.5)
    parser.add_argument("--max-demand-corr", type=float, default=.6)
    parser.add_argument("--min-relative-mae", type=float, default=.5)
    parser.add_argument("--min-tv-ratio", type=float, default=1.4)
    args = parser.parse_args(argv)
    if (args.top < 1 or not -1 <= args.max_demand_corr <= 1 or
            not 0 <= args.min_mean_ratio <= args.max_mean_ratio or
            any(not np.isfinite(v) or v < 0 for v in
                (args.min_mean, args.min_mean_ratio, args.max_mean_ratio,
                 args.min_range_ratio, args.min_relative_mae, args.min_tv_ratio))):
        parser.error("Invalid thresholds or --top.")
    frame, profiles, weights, snapshot = read_inputs(args.radar_dir)
    if args.node is not None and (args.node in FIXED_NODES or not 0 <= args.node < len(profiles)):
        parser.error(f"--node must be in [0, {len(profiles)-1}] and different from 49 and 232.")
    candidates = rank_candidates(profiles, weights, args)
    candidate_path = args.radar_dir / "fixed_pair_candidates.csv"
    candidates.to_csv(candidate_path, index=False)
    columns = ["node_id", "eligible", "mean_demand", "correlation_to_49", "correlation_to_232",
               "allocation_tv_to_49", "allocation_tv_to_232", "allocation_tv_ratio", "failed_criteria"]
    print(candidates[columns].head(args.top).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nEligible: {int(candidates.eligible.sum())}/{len(candidates)}; all metrics: {candidate_path}")
    if args.node is None and not args.auto:
        print("Ranking only. Choose x with --node X, or export the recommendation with --auto. No plots generated.")
        return
    eligible = candidates[candidates.eligible]
    if args.auto and eligible.empty:
        parser.error("No node meets the criteria. Review the CSV and adjust thresholds or specify --node X.")
    node = args.node if args.node is not None else int(eligible.iloc[0].node_id)
    selected = [*FIXED_NODES, node]
    row = candidates[candidates.node_id == node].iloc[0]
    thresholds = {key: value for key, value in vars(args).items()
                  if key not in ("radar_dir", "node", "auto", "top")}
    rule = ("Fixed nodes 49 and 232. Filter contrast nodes by demand volume/range, demand correlation "
            "and relative MAE, and minimum allocation TV relative to the fixed pair. "
            "Rank eligible nodes by minimum allocation TV descending, then node ID. "
            "Manual --node choices are exported even when they fail the automatic filters. "
            "All threshold values and eligibility are recorded in this manifest.")
    meta = dict(dataset=snapshot["dataset"], seed=snapshot["seed"], checkpoint=snapshot["checkpoint"],
                split=snapshot["split"], sample_start=snapshot["sample_start"],
                sample_count=snapshot["sample_count"], selected_node_ids=selected,
                similar_nodes=list(FIXED_NODES), contrast_node=node, selection_rule=rule,
                selection_mode="manual" if args.node is not None else "auto",
                selection_uses_allocations=True, thresholds=thresholds,
                automatic_criteria_passed=bool(row.eligible), failed_criteria=row.failed_criteria,
                pair_metrics=[pair_metrics(profiles, weights, a, b)
                              for a, b in [(49, 232), (49, node), (232, node)]],
                scope="Jointly selected explanatory illustration using test-average demand and allocations.")
    # A manually chosen constant-demand node has undefined Pearson correlation.
    for metric in meta["pair_metrics"]:
        if not np.isfinite(metric["correlation"]):
            metric["correlation"] = None
    snapshot.update(selected_node_ids=selected, selection_rule=rule, selection_uses_allocations=True)
    for name, data in [("demand_selection.json", meta), ("spatial_attention_per_node.json", snapshot)]:
        (args.radar_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    frame[["hour", *(f"node_{n}" for n in selected)]].to_csv(
        args.radar_dir / "selected_node_demand_profiles.csv", index=False)
    print(f"\nExported nodes: {selected}; automatic criteria passed: {bool(row.eligible)}")
    if not row.eligible:
        print(f"Manual choice failed criteria: {row.failed_criteria}")
    plot_selection(args.radar_dir, frame.hour.to_numpy(), profiles, weights, selected)
    print(f"Saved PNG/PDF plots in {args.radar_dir}: nodes_49_232_{node}_comparison, "
          "nyc_taxi_selected_demand_preview, nyc_taxi_radar_preview.")
    print("For MATLAB styling, run:")
    print(f"run('{(args.radar_dir / 'radar.m').as_posix()}');")


if __name__ == "__main__":
    main()
