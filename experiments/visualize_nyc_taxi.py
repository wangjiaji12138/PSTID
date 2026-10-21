"""Reproduce NYC-Taxi temporal cases and spatial snapshots from seed 88."""
import argparse
import json
import tempfile
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline.dataset import StandardScaler
from data_pipeline.normalization import normalize_input
from models.PROPHET.model import PROPHET
from utils.paths import PROJECT_ROOT as ROOT


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_model(suffix, nodes):
    paths = list((ROOT / "results_88/nyc_taxi").glob(f"*__{suffix}/checkpoint/best_model.pth"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} checkpoint: {paths}")
    ck = torch.load(paths[0], map_location="cpu", weights_only=True)
    cfg = ck["config"]
    assert cfg["data"] == "nyc_taxi" and cfg["seed"] == 88
    assert all(cfg.get(k, True) for k in ("use_proto", "use_spatio", "use_temporal", "use_his"))
    assert bool(cfg["use_adj"]) == (suffix == "Full")
    model = PROPHET.from_args(cfg, nodes, None, torch.device("cpu")).eval()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model, StandardScaler(ck["scaler_mean"], ck["scaler_std"]), paths[0]


def embeddings(model, scaler, x):
    return model._get_embeddings(
        {"X": normalize_input(torch.from_numpy(np.array(x, dtype=np.float32)), scaler)})


def temporal(model, scaler, x):
    chunks = []
    basis = model.proto_module.temporal_basis
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            _, h, tid, diw, _, _ = embeddings(model, scaler, x[start:start + 32])
            inputs = torch.cat([h[:, -1], tid[:, -1], diw[:, -1]], -1)
            w = torch.softmax(basis.allocator(inputs) / basis.temp, -1)
            actual, _ = basis(h, tid, diw)
            torch.testing.assert_close(w @ basis.prototypes, actual[:, 0])
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))
            chunks.append(w.numpy())
    return np.concatenate(chunks)


def spatial(model, scaler, x, start):
    """Average actual batch-dependent spatial allocations over the test split."""
    branch = model.proto_module.spatial_codebook
    total = np.zeros((x.shape[2], branch.num_protos), dtype=np.float64)
    with torch.inference_mode():
        for i in range(start, len(x), 64):
            batch = x[i:min(i + 64, len(x))]
            _, h, _, _, node, adj = embeddings(model, scaler, batch)
            identity = adj if adj is not None else node
            q = branch.proj(torch.cat([identity, h.mean(1).mean(0)], -1))
            w, enhanced = branch._compute_attention(q)
            actual = branch(h, adj, node)
            torch.testing.assert_close(enhanced, actual["proto_enhanced"][0, 0])
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[:, 0]))
            total += len(batch) * w.numpy()
    return total / (len(x) - start)


def spatial_json(w, checkpoint, start, count):
    return {
        "dataset": "nyc_taxi", "seed": 88, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "split": "test", "sample_start": start, "sample_count": count,
        "batch_size": 64,
        "aggregation": "Sequential test batches; sample-count-weighted mean of actual batch allocations. Spatial queries depend on batch mean history.",
        "num_nodes": len(w), "num_spatial_prototypes": w.shape[1],
        "per_node_attention": {f"node_{n}": row.tolist() for n, row in enumerate(w)},
        "dominant_prototype": w.argmax(-1).tolist(),
        "mean_attention_per_proto": w.mean(0).tolist(),
    }


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(x, temporal_only=False):
    out = ROOT / "figs/temporal_nyc_taxi"
    out.mkdir(parents=True, exist_ok=True)
    model, scaler, checkpoint = load_model("Full", x.shape[2])
    # The preprocessing pipeline starts at 2016-01-01 00:00; DOW is a
    # relative seven-day code, NOT Python's calendar weekday.
    window = x.shape[1]
    start = round(len(x) * .6) + round(len(x) * .2)
    origin = pd.Timestamp("2016-01-01")
    for i in (0, start, len(x) - 1):
        slots = np.arange(i, i + window)
        np.testing.assert_allclose(x[i, :, 0, 1], slots % 48 / 48, atol=1e-7)
        np.testing.assert_allclose(x[i, :, 0, 2], (slots // 48) % 7 / 7, atol=1e-7)
    end_times = pd.date_range(origin + pd.Timedelta(minutes=30 * (start + window - 1)),
                              origin + pd.Timedelta(minutes=30 * (len(x) + window - 2)), freq="30min")
    days = pd.DatetimeIndex(sorted(set(t.normalize() for t in end_times if t.dayofweek == 2)))
    hours = np.arange(0, 24, .5)
    days = pd.DatetimeIndex([d for d in days if d >= end_times[0] and d + pd.Timedelta(hours=23.5) <= end_times[-1]])
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=float(h)) for d in days for h in hours])
    indices = ((times - origin) / pd.Timedelta(minutes=30)).astype(int).to_numpy() - window + 1
    assert indices.min() >= start and indices.max() < len(x)
    demand = np.asarray(x[indices, -1, :, 0], dtype=np.float64).reshape(len(days), 48, x.shape[2])
    # Compare raw same-clock demand; model allocations do not affect selection.
    thresholds = dict(min_daily_mean=5., min_similar_correlation=.85,
                      max_similar_relative_mae=.30, min_contrast_relative_mae=.60,
                      min_contrast_ratio=2.)
    records = []
    for node in range(x.shape[2]):
        curves = demand[:, :, node]
        means = curves.mean(1)
        centered = curves - means[:, None]
        norms = np.linalg.norm(centered, axis=1)
        corr = np.divide(centered @ centered.T, norms[:, None] * norms[None, :],
                         out=np.zeros((len(days), len(days))),
                         where=norms[:, None] * norms[None, :] > 0)
        mae = np.abs(curves[:, None] - curves[None, :]).mean(-1)
        relative = mae / ((means[:, None] + means[None, :]) / 2 + 1)
        for triple in combinations(range(len(days)), 3):
            a, b = min(combinations(triple, 2), key=lambda pair: relative[pair])
            c = next(d for d in triple if d not in (a, b))
            contrast = min(relative[a, c], relative[b, c])
            ratio = contrast / max(relative[a, b], 1e-8)
            eligible = (means[list(triple)].min() >= thresholds['min_daily_mean']
                        and corr[a, b] >= thresholds['min_similar_correlation']
                        and relative[a, b] <= thresholds['max_similar_relative_mae']
                        and contrast >= thresholds['min_contrast_relative_mae']
                        and ratio >= thresholds['min_contrast_ratio'])
            records.append(dict(node_id=node, similar_day_a=a, similar_day_b=b, contrast_day=c,
                similar_date_a=str(days[a].date()), similar_date_b=str(days[b].date()),
                contrast_date=str(days[c].date()), min_daily_mean=float(means[list(triple)].min()),
                similar_correlation=float(corr[a, b]), similar_relative_mae=float(relative[a, b]),
                min_contrast_relative_mae=float(contrast), contrast_ratio=float(ratio),
                selection_score=float(contrast - relative[a, b]), eligible=bool(eligible)))
    candidates = pd.DataFrame(records).sort_values(
        ['eligible', 'selection_score', 'node_id', 'similar_day_a', 'similar_day_b', 'contrast_day'],
        ascending=[False, False, True, True, True, True], kind='stable')
    candidates.to_csv(out / 'curve_candidates.csv', index=False)
    eligible = candidates[candidates.eligible]
    if eligible.empty:
        raise ValueError('No similar-pair/contrast Wednesday case; see curve_candidates.csv.')
    chosen = eligible.iloc[0]
    node = int(chosen.node_id)
    selected_days = [int(chosen[k]) for k in ('similar_day_a', 'similar_day_b', 'contrast_day')]
    selected_indices = indices.reshape(len(days), 48)[selected_days].reshape(-1)
    alpha = temporal(model, scaler, np.array(x[selected_indices], dtype=np.float32)).reshape(3, 48, x.shape[2], -1)
    days, demand = days[selected_days], demand[selected_days]
    indices = selected_indices
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=float(h)) for d in days for h in hours])
    ds = list(range(3))
    pair_metrics = []
    for a, b in combinations(ds, 2):
        ca, cb = demand[a, :, node], demand[b, :, node]
        pair_metrics.append(dict(date_a=str(days[a].date()), date_b=str(days[b].date()),
            correlation=float(np.corrcoef(ca, cb)[0, 1]), mae=float(np.abs(ca-cb).mean()),
            relative_mae=float(np.abs(ca-cb).mean() / ((ca.mean()+cb.mean())/2+1)),
            allocation_tv=float(np.abs(alpha[a, :, node]-alpha[b, :, node]).sum(-1).mean()/2)))
    rows = []
    for day in ds:
        for slot, hour in enumerate(hours):
            idx = day * 48 + slot
            rows.append(dict(date=str(days[day].date()), time=times[idx].strftime("%H:%M"),
                timestamp=str(times[idx]), history_end=str(times[idx]),
                history_start=str(times[idx] - pd.Timedelta(minutes=30 * (window - 1))),
                node_id=node, sample_index=int(indices[idx]), demand=float(demand[day, slot, node]),
                **{f"alpha_P{k+1}": float(v) for k, v in enumerate(alpha[day, slot, node])}))
    pd.DataFrame(rows).to_csv(out / "selected_node_day_allocations.csv", index=False)
    curves = demand[ds, :, node]
    step = max(10., float(10 ** np.floor(np.log10(max(float(curves.max()), 1))) / 2))
    ymax = float(np.ceil(curves.max() * 1.1 / step) * step)
    manifest = dict(dataset="nyc_taxi", seed=88, checkpoint=str(checkpoint.relative_to(ROOT)),
        selected_node=node, dates=[str(days[d].date()) for d in ds], hours=hours.tolist(),
        slots_per_day=48, allocation_shape=list(alpha.shape), history_window=window,
        daily_displayed_slot_totals=curves.sum(1).tolist(), demand_y_limits=[0, ymax],
        demand_y_tick_interval=step, split="test", test_sample_start=start,
        timeline_origin=str(origin), calendar_note="Dates follow preprocessing start 2016-01-01; saved DOW is relative day index modulo 7, not Monday=0. Model inputs are unchanged.",
        selection_rule="All nodes and all triples of complete test Wednesdays, without month or consecutive-week restriction. Choose the closest pair by relative MAE; apply recorded similarity/contrast thresholds. Rank by minimum contrast relative MAE minus similar-pair relative MAE, then node and dates.",
        selection_uses_allocations=False, thresholds=thresholds, pair_metrics=pair_metrics,
        similar_dates=[str(days[d].date()) for d in (0, 1)], contrast_date=str(days[2].date()),
        day_roles=['Similar A', 'Similar B', 'Contrast'], weekday='Wednesday',
        candidate_date_range=[str(end_times[0]), str(end_times[-1])], candidate_count=len(candidates), eligible_count=len(eligible),
        scope="Selected explanatory case, not unbiased evidence of demand/allocation association.",
        checks=["strict checkpoint loading", "saved time-feature alignment", "test-only samples",
                "probability sums", "temporal weighted prototypes equal actual forward"])
    write_json(out / "manifest.json", manifest)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    for row, day in enumerate(ds):
        axes[row, 0].plot(hours, demand[day, :, node])
        axes[row, 0].fill_between(hours, demand[day, :, node], alpha=.2)
        axes[row, 0].set(ylim=(0, ymax), ylabel="Orders / 30 min", title=f"{manifest['day_roles'][row]} | {days[day].date()}")
        for k in range(alpha.shape[-1]):
            axes[row, 1].plot(hours, alpha[day, :, node, k], label=f"T{k}")
        axes[row, 1].set(ylim=(0, 1), ylabel="Allocation weight")
    axes[-1, 1].legend(ncol=4, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(np.arange(0, 24, 4), [f"{h:02}:00" for h in range(0, 24, 4)])
    fig.suptitle(f"NYC-Taxi | Node {node} | seed 88 | test Wednesdays")
    fig.tight_layout()
    save(fig, out, "nyc_taxi_wednesday_day_comparison_preview")
    fig, ax = plt.subplots(figsize=(9, 4))
    for day in ds:
        ax.plot(hours, demand[day, :, node], label=f"{manifest['day_roles'][day]} | {days[day].date()}")
    ax.set(xlabel="Time of day", ylabel="Orders / 30 min", ylim=(0, ymax))
    ax.legend()
    save(fig, out, "nyc_taxi_wednesday_day_overlay_preview")
    print("Temporal case:", node, manifest["dates"], flush=True)

    if temporal_only:
        return

    w = spatial(model, scaler, x, start)
    other, other_scaler, other_path = load_model("wo_Adj", x.shape[2])
    w_other = spatial(other, other_scaler, x, start)
    data = spatial_json(w, checkpoint, start, len(x) - start)
    from experiments.select_spatial_demand_nodes import select_nodes
    selected, selection_rule = select_nodes(x, start, ROOT / "figs/radar", weights=w)
    data["selected_node_ids"] = selected
    data["selection_rule"] = selection_rule
    data["selection_uses_allocations"] = True
    write_json(ROOT / "figs/radar/spatial_attention_per_node.json", data)
    write_json(ROOT / "figs/lap/use_lap.json", data)
    write_json(ROOT / "figs/lap/out_lap.json", spatial_json(w_other, other_path, start, len(x) - start))
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 5))
    theta = np.linspace(0, 2 * np.pi, w.shape[1] + 1)
    for n in selected:
        ax.plot(theta, np.r_[w[n], w[n, 0]], marker="o", label=f"Node {n}")
    ax.set_xticks(theta[:-1], [f"S{k}" for k in range(w.shape[1])])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, ROOT / "figs/radar", "nyc_taxi_radar_preview")
    fig, ax = plt.subplots(figsize=(8, 4))
    ids = np.arange(w.shape[1])
    for offset, value, label in [(-.2, w, "Full"), (.2, w_other, "w/o Adj")]:
        counts = np.bincount(value.argmax(-1), minlength=len(ids)) / len(value) * 100
        ax.bar(ids + offset, counts, width=.4, label=label)
    ax.set(xlabel="Prototype ID (independently trained codebooks)", ylabel="Dominant node share (%)")
    ax.set_xticks(ids)
    ax.legend()
    save(fig, ROOT / "figs/lap", "nyc_taxi_lap_preview")
    from experiments.select_spatial_demand_nodes import update_radar
    update_radar(x, start, ROOT / "figs/radar")
    print("Spatial selected nodes:", selected, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-only", action="store_true", help="Update temporal figures only")
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Unzip once into a temporary memory-mapped file instead of loading the
    # multi-gigabyte sliding-window tensor repeatedly into RAM.
    with tempfile.TemporaryDirectory(prefix="nyc_taxi_vis_") as tmp:
        with zipfile.ZipFile(ROOT / "data/processed/nyc_taxi/all.npz") as archive:
            archive.extract("X.npy", tmp)
        x = np.load(Path(tmp) / "X.npy", mmap_mode="r")
        run(x, temporal_only=args.temporal_only)


if __name__ == "__main__":
    main()
