"""Training-only weekly flow anchors for STSSDL's shared history encoder."""

import numpy as np


def fit_weekly_flow_anchor(train_x, slots_per_day):
    """Fit raw flow means by (weekday, time slot, node).

    Inputs are chronological sliding windows with channels [flow, TOD, DOW].
    Remove overlap between neighboring windows so boundary timestamps are not
    underweighted. Zero demand is a valid observation. Missing calendar cells
    fall back to the node's mean over the same training observations.
    """
    x = np.asarray(train_x)
    if x.ndim != 4 or x.shape[-1] != 3 or not x.shape[0] or not x.shape[1]:
        raise ValueError('Expected nonempty training X with shape [samples, time, nodes, 3]')
    if slots_per_day <= 0:
        raise ValueError('slots_per_day must be positive')

    pieces = [x[0]]
    for previous, current in zip(x[:-1], x[1:]):
        overlap = 0
        for length in range(x.shape[1] - 1, 0, -1):
            if np.array_equal(previous[-length:], current[:length], equal_nan=True):
                overlap = length
                break
        pieces.append(current[overlap:])
    observations = np.concatenate(pieces, axis=0)
    calendar = observations[..., 1:3]
    if not np.isfinite(calendar).all():
        raise ValueError('Training calendar contains nonfinite values')
    if not np.allclose(calendar, calendar[:, :1], atol=1e-6):
        raise ValueError('Expected the same timestamp across nodes')
    if np.any(calendar < 0) or np.any(calendar >= 1):
        raise ValueError('Expected normalized TOD and DOW in [0, 1)')
    tod = np.rint(calendar[:, 0, 0] * slots_per_day).astype(np.int64)
    dow = np.rint(calendar[:, 0, 1] * 7).astype(np.int64)
    if np.any(tod >= slots_per_day) or np.any(dow >= 7):
        raise ValueError('Training calendar is outside the configured daily/weekly slots')
    if not np.allclose(calendar[:, 0, 0] * slots_per_day, tod, atol=1e-4):
        raise ValueError('time_intervals does not match the training time slots')
    if not np.allclose(calendar[:, 0, 1] * 7, dow, atol=1e-4):
        raise ValueError('Expected weekday / 7 in channel 3')

    values = observations[..., 0]
    valid = np.isfinite(values)
    node_counts = valid.sum(axis=0)
    if np.any(node_counts == 0):
        raise ValueError('Cannot fit history for a node without finite training flow')
    safe_values = np.where(valid, values, 0)
    fallback = safe_values.sum(axis=0, dtype=np.float64) / node_counts
    sums = np.zeros((7 * slots_per_day, x.shape[2]), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    keys = dow * slots_per_day + tod
    np.add.at(sums, keys, safe_values)
    np.add.at(counts, keys, valid)
    anchor = np.broadcast_to(fallback, sums.shape).copy()
    np.divide(sums, counts, out=anchor, where=counts > 0)
    return (anchor.reshape(7, slots_per_day, x.shape[2]).astype(np.float32),
            counts.reshape(7, slots_per_day, x.shape[2]))
