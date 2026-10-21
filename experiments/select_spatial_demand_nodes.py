"""Select an illustrative radar comparison using demand and allocation contrast."""
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils.paths import PROJECT_ROOT as ROOT


def select_nodes(x, start, output, weights=None):
    # One observation per sample: overlapping histories are not counted repeatedly.
    demand = np.asarray(x[start:, -1, :, 0], dtype=np.float64)
    slots = np.rint(x[start:, -1, 0, 1] * 48).astype(int)
    profiles = np.stack([demand[slots == s].mean(0) for s in range(48)]).T
    means = profiles.mean(1)
    active = np.flatnonzero((means >= 5) & (profiles.std(1) >= .1 * means))
    centered = profiles - profiles.mean(1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    corr = normalized @ normalized.T
    mae = np.abs(profiles[:, None] - profiles[None, :]).mean(-1)
    relative = mae / ((means[:, None] + means[None, :]) / 2 + 1)
    pairs = []
    for pos, a in enumerate(active):
        for b in active[pos + 1:]:
            pairs.append(dict(node_a=int(a), node_b=int(b), correlation=float(corr[a, b]),
                              relative_mae=float(relative[a, b]), mae=float(mae[a, b])))
    ranking = pd.DataFrame(pairs).sort_values(
        ["relative_mae", "node_a", "node_b"], kind="stable")
    ranking.to_csv(output / "demand_pair_candidates.csv", index=False)
    similar = ranking[ranking.correlation.between(.93, .97) &
                      ranking.relative_mae.between(.20, .35)].copy()
    # Require visible morning/evening structure in BOTH demand curves.
    # Smoothing is for peak detection only; exports and plots stay unsmoothed.
    peak_counts = []
    for profile in profiles:
        smoothed = np.convolve(np.pad(profile, (2, 2), mode="wrap"),
                               np.ones(5) / 5, mode="valid")
        peaks, _ = find_peaks(smoothed, prominence=.12 * np.ptp(smoothed), distance=6)
        peak_counts.append(len(peaks))
    similar["peaks_a"] = [peak_counts[int(n)] for n in similar.node_a]
    similar["peaks_b"] = [peak_counts[int(n)] for n in similar.node_b]
    similar = similar[(similar.peaks_a >= 2) & (similar.peaks_b >= 2) &
                      (means[similar.node_a.to_numpy()] >= 20) &
                      (means[similar.node_b.to_numpy()] >= 20)]
    if similar.empty:
        raise ValueError("No similar pair satisfies the similarity and multi-peak demand criteria.")
    similar["selection_score"] = ((similar.correlation - .95).abs() +
                                  (similar.relative_mae - .275).abs())
    similar = similar.sort_values(["selection_score", "node_a", "node_b"], kind="stable")
    similar.to_csv(output / "similar_demand_pair_candidates.csv", index=False)
    a, b = int(similar.iloc[0].node_a), int(similar.iloc[0].node_b)
    if weights is None:
        snapshot = json.loads((output / "spatial_attention_per_node.json").read_text())
        assert snapshot["dataset"] == "nyc_taxi" and snapshot["sample_start"] == start
        weights = np.array([snapshot["per_node_attention"][f"node_{n}"]
                            for n in range(len(profiles))])
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
    tv = np.abs(weights[:, None] - weights[None, :]).sum(-1) / 2
    pair_tv = float(tv[a, b])
    pair_mean = (means[a] + means[b]) / 2
    pair_range = (np.ptp(profiles[a]) + np.ptp(profiles[b])) / 2
    contrasts = pd.DataFrame([
        dict(node_id=int(n), max_correlation=float(max(corr[a, n], corr[b, n])),
             min_relative_mae=float(min(relative[a, n], relative[b, n])),
             mean_demand=float(means[n]),
             mean_ratio=float(means[n] / pair_mean),
             range_ratio=float(np.ptp(profiles[n]) / pair_range),
             min_allocation_tv=float(min(tv[a, n], tv[b, n])),
             allocation_tv_to_a=float(tv[a, n]),
             allocation_tv_to_b=float(tv[b, n]))
        for n in active if n not in (a, b)])
    contrasts = contrasts.sort_values(
        ["min_allocation_tv", "max_correlation", "node_id"],
        ascending=[False, True, True], kind="stable")
    contrasts.to_csv(output / "demand_contrast_candidates.csv", index=False)
    different = contrasts[(contrasts.min_relative_mae >= .5) &
                          contrasts.mean_ratio.between(.5, 2.) &
                          (contrasts.range_ratio >= .5) &
                          (contrasts.max_correlation <= .6) &
                          (contrasts.min_allocation_tv >= 1.4 * pair_tv)]
    if different.empty:
        raise ValueError("No contrast node satisfies joint demand and allocation criteria.")
    c = int(different.iloc[0].node_id)
    selected = [a, b, c]
    metrics = [{**row, "role": "similar pair" if row["node_a"] == a and row["node_b"] == b else "contrast"}
               for row in [dict(node_a=i, node_b=j, correlation=float(corr[i, j]),
                                mae=float(mae[i, j]), relative_mae=float(relative[i, j]),
                                allocation_tv=float(tv[i, j]))
                           for i, j in [(a, b), (a, c), (b, c)]]]
    all_profiles = pd.DataFrame(profiles.T, columns=[f"node_{n}" for n in range(len(profiles))])
    all_profiles.insert(0, "hour", np.arange(48) / 2)
    all_profiles.to_csv(output / "all_node_demand_profiles.csv", index=False)
    pd.DataFrame({"hour": np.arange(48) / 2,
                  **{f"node_{n}": profiles[n] for n in selected}}).to_csv(
                      output / "selected_node_demand_profiles.csv", index=False)
    rule = ("Test mean daily demand profiles (48 half-hour slots, one last-slot observation per sample). "
            "Eligible nodes: mean >= 5 and profile std >= 0.1 * mean. Similar pair: correlation in [.93, .97], "
            "relative MAE in [.20, .35], both means >= 20, both profiles have >= 2 peaks after "
            "5-slot circular moving-average smoothing (prominence >= .12 * smoothed range, "
            "minimum peak separation 6 slots). Smoothing is only used for selection, not plotting. "
            "Minimize abs(correlation-.95) + abs(relative_MAE-.275) "
            "to select near the interval centers, excluding almost identical curves. Contrast: relative MAE >= .5 to both, "
            "mean demand within 0.5 to 2 times the pair average, daily range >= 0.5 times the pair average range; "
            "maximum demand correlation <= .6 and minimum allocation TV >= 1.4 times the similar-pair TV. "
            "Maximize minimum allocation TV to the pair; ties by demand correlation then node ID. "
            "Relative MAE = MAE / (pair mean + 1). TV = half the L1 distance of allocation vectors. "
            "Similar pair uses demand only; contrast selection uses both demand and prototype weights.")
    manifest = dict(dataset="nyc_taxi", seed=88, selected_node_ids=selected,
                    similar_nodes=[a, b], contrast_node=c, selection_rule=rule,
                    selection_uses_allocations=True, split="test", sample_start=start,
                    sample_count=len(demand), eligible_nodes=len(active), pair_metrics=metrics,
                    scope="Illustrative case jointly selected on demand and model outputs, not unbiased evidence of their association. Average daily profiles do not imply every day matches.")
    (output / "demand_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return selected, rule


def update_radar(x, start, output):
    selected, rule = select_nodes(x, start, output)
    path = output / "spatial_attention_per_node.json"
    data = json.loads(path.read_text())
    assert data["dataset"] == "nyc_taxi" and data["sample_start"] == start
    data.update(selected_node_ids=selected, selection_rule=rule, selection_uses_allocations=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    frame = pd.read_csv(output / "selected_node_demand_profiles.csv")
    colors = ["#4170a6", "#358b70", "#bb5753"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                            gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for n, color in zip(selected, colors):
        ax.plot(frame.hour, frame[f"node_{n}"], color=color, label=f"Node {n}")
    ax.set(xlabel="Time of day", ylabel="Mean orders / 30 min", title="Observed test-average demand")
    ax.set_xticks(range(0, 24, 4), [f"{h:02}:00" for h in range(0, 24, 4)])
    ax.legend()
    axes[1].remove()
    ax = fig.add_subplot(122, projection="polar")
    k = data["num_spatial_prototypes"]
    theta = np.linspace(0, 2 * np.pi, k + 1)
    for n, color in zip(selected, colors):
        w = data["per_node_attention"][f"node_{n}"]
        ax.plot(theta, w + w[:1], color=color, marker="o", markersize=3, label=f"Node {n}")
    ax.set_xticks(theta[:-1], [f"S{i}" for i in range(k)])
    ax.set_title("Full model spatial allocations", pad=20)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"nyc_taxi_demand_radar_preview.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Also replace the standalone preview so it cannot retain the old nodes.
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 5))
    for n, color in zip(selected, colors):
        w = data["per_node_attention"][f"node_{n}"]
        ax.plot(theta, w + w[:1], color=color, marker="o", label=f"Node {n}")
    ax.set_xticks(theta[:-1], [f"S{i}" for i in range(k)])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"nyc_taxi_radar_preview.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print((output / "demand_selection.json").read_text(), flush=True)


def main():
    output = ROOT / "figs/radar"
    with tempfile.TemporaryDirectory(prefix="radar_demand_") as tmp:
        with zipfile.ZipFile(ROOT / "data/processed/nyc_taxi/all.npz") as archive:
            archive.extract("X.npy", tmp)
        x = np.load(Path(tmp) / "X.npy", mmap_mode="r")
        start = round(len(x) * .6) + round(len(x) * .2)
        update_radar(x, start, output)


if __name__ == "__main__":
    main()
