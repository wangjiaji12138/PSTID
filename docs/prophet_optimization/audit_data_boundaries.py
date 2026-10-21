#!/usr/bin/env python3
"""Read existing NPZ files and audit the current PROPHET data contract.

This script does not regenerate data, train a model, or evaluate predictions.
Run from any directory: python docs/prophet_optimization/audit_data_boundaries.py
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("cq", "hz", "nyc_taxi", "chi_taxi", "nyc_bike")


def matching_target_overlap(left, right):
    """Largest suffix/prefix equality; these datasets use chronological stride 1."""
    for steps in range(min(len(left), len(right)) - 1, 0, -1):
        if np.array_equal(left[-steps:], right[:steps]):
            return steps
    return 0


def audit_dataset(path):
    with np.load(path) as archive:
        x, y = archive["X"], archive["y"]
    sample_count, input_window = x.shape[:2]
    output_window = y.shape[1]
    train_count = round(sample_count * 0.6)
    val_count = round(sample_count * 0.2)
    split_starts = {"train": 0, "val": train_count, "test": train_count + val_count}

    overlap_checks = {}
    for split, start in split_starts.items():
        overlap_checks[split] = {
            "sample_index": start,
            "next_sample_last_input_equals_first_target": bool(
                np.array_equal(x[start + 1, -1, :, 0], y[start, 0, :, 0])
            ),
            "sample_plus_24_input_equals_entire_target": bool(
                input_window == output_window == 24
                and np.array_equal(x[start + 24, :, :, 0], y[start, :, :, 0])
            ),
            "first_40_targets_equal_inputs_24_to_63_in_same_batch": bool(
                input_window == output_window == 24
                and np.array_equal(x[start + 24:start + 64, :, :, 0], y[start:start + 40, :, :, 0])
            ),
        }

    boundaries = {}
    for name, right in (("train_val", train_count), ("val_test", train_count + val_count)):
        boundaries[name] = {
            "left_sample_index": right - 1,
            "right_sample_index": right,
            "adjacent_target_overlap_steps": matching_target_overlap(y[right - 1], y[right]),
            "left_target_except_first_equals_right_target_except_last": bool(
                np.array_equal(y[right - 1, 1:], y[right, :-1])
            ),
        }

    calendar = {}
    # Dataset.__getitem__ converts to float32; model indexes via (feature * size).long().
    # Collect the observed final-history identities, which are the ones the model uses.
    for name, channel, size in (("time_of_day", 1, 48), ("day_of_week", 2, 7)):
        values = x[:, -1, 0, channel].astype(np.float32)
        scaled = values * np.float32(size)
        truncated = scaled.astype(np.int64)
        rounded = np.rint(scaled).astype(np.int64)
        mismatch = truncated != rounded
        calendar[name] = {
            "embedding_size": size,
            "observed_indices": np.unique(truncated).tolist(),
            "truncation_vs_nearest_integer_mismatches": int(mismatch.sum()),
            "max_distance_from_integer": float(np.max(np.abs(scaled - rounded))),
            "out_of_range_count": int(((truncated < 0) | (truncated >= size)).sum()),
        }

    finite = np.isfinite(y)
    train_mask = ~np.isnan(y)  # criterion's default null_val=np.nan
    eval_mask = y > -0.9  # evaluate(..., null_value=-1.0)
    targets = {
        "count": int(y.size),
        "nonfinite_count": int((~finite).sum()),
        "nan_count": int(np.isnan(y).sum()),
        "negative_count": int((y < 0).sum()),
        "zero_count": int((y == 0).sum()),
        "zero_fraction": float((y == 0).mean()),
        "minimum": float(np.min(y)),
        "maximum": float(np.max(y)),
        "train_mae_and_eval_mae_mask_difference_count": int((train_mask != eval_mask).sum()),
        "mape_eligible_count": int((eval_mask & (y >= 0.5)).sum()),
    }
    result = {
        "source": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "sample_counts": {
            "all": sample_count,
            "train": train_count,
            "val": val_count,
            "test": sample_count - train_count - val_count,
        },
        "within_split_window_checks": overlap_checks,
        "split_boundaries": boundaries,
        "calendar_float32_index_audit": calendar,
        "targets_all_windows": targets,
    }
    del x, y, finite, train_mask, eval_mask
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data_audit.json"))
    args = parser.parse_args()
    report = {
        "scope": "Existing data and source-code audit; no model performance claim.",
        "source_contract": {
            "chronological_split": "data_pipeline/dataset.py:39-65",
            "sequential_dataloaders": "data_pipeline/loaders.py:31-33; shuffle defaults to False",
            "sliding_windows": "data_pipeline/npz.py:180-188; data_pipeline/grid.py:182-187",
            "cross_sample_history_aggregation": "models/PROPHET/spatial.py:93-105",
            "calendar_index_conversion": "models/PROPHET/model.py:191,196; data_pipeline/dataset.py:103",
            "training_loss": "training/engine.py:56; utils/metrics.py:13-29",
            "evaluation_masks": "training/engine.py:123; utils/metrics.py:22,69-70",
        },
        "default_batch_future_information_path": {
            "batch_size": 64,
            "input_window": 24,
            "output_window": 24,
            "stride": 1,
            "zero_based_batch_indices_with_complete_target_present_in_later_input": [0, 39],
            "number_of_such_samples": 40,
            "equality": "X[i + 24, :, :, 0] == y[i, :, :, 0]",
            "derivation": "For i=0..39, i+24 remains inside a 64-sample batch. Later inputs enter the shared spatial query through the batch mean of history embeddings.",
            "qualification": "Existence of a future-information path is established; its effect on trained-model metrics is not measured by this script.",
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        report["datasets"][dataset] = audit_dataset((args.data_root / dataset / "all.npz").resolve())
        print(f"Audited {dataset}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
