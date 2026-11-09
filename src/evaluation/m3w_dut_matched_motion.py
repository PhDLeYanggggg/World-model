"""Past-only matching and paired scoring for a DUT development comparison."""

import numpy as np

from src.evaluation.m3w_dut_readout import score_arrays


def disjoint_queries(frames, history=8, future=12):
    frames = np.asarray(frames)
    if frames.ndim != 1 or np.any(np.diff(frames) <= 0):
        raise ValueError("Require increasing, unique query frames")
    use = []
    for f in frames:
        if not use or f - use[-1] >= history + future:
            use.append(int(f))
    return np.asarray(use, dtype=np.int64)


def ols4(history, horizon=12):
    """Fit the slope of the last four samples and anchor at the last observation."""
    h = np.asarray(history)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Require a finite eight-step history")
    slope = np.einsum("t,ntd->nd", np.array([-1.5, -.5, .5, 1.5]) / 5., h[:, -4:] - h[:, -1:])
    return h[:, -1:] + np.arange(1, horizon + 1)[None, :, None] * slope[:, None]


def peak(delta):
    return np.linalg.norm(delta, axis=-1).max(1)


def matched(cv, neural, active, controls, eps=1e-10):
    """Keep the same eligible rows and peak radius for every direction."""
    active = np.asarray(active, bool)
    radius = peak(neural - cv)
    on = active & (radius > eps)
    deltas = {k: p - cv for k, p in controls.items()}
    for delta in deltas.values():
        on &= peak(delta) > eps
    out = {"neural": np.where(on[:, None, None], neural, cv)}
    for k, d in deltas.items():
        f = np.divide(radius, peak(d), out=np.zeros(len(cv)), where=on)
        out[k] = cv + f[:, None, None] * d
    for p in out.values():
        np.testing.assert_array_equal((p != cv).any((1, 2)), on)
        np.testing.assert_allclose(peak(p - cv), np.where(on, radius, 0.), atol=1e-10)
    return out, on


def arms_summary(cv, pred, y, valid, scale, easy_cut, hard_cut, kinds):
    a = score_arrays(cv, pred, y, valid, scale, easy_cut=easy_cut, hard_cut=hard_cut)
    active = (pred != cv).any((1, 2))
    subsets = {"complete": a["complete"], "endpoint": valid[:, -1],
               "easy": a["easy"], "hard": a["hard"],
               "selected_easy": a["easy"] & active,
               "pedestrian": a["complete"] & (kinds == "pedestrian"),
               "vehicle": a["complete"] & (kinds == "vehicle")}
    scores = {}
    for k, use in subsets.items():
        n = int(use.sum())
        # ADE for an endpoint-only row has no full-path interpretation.
        score = dict(n=n, fde_sum=float(a["fde"][use].sum()))
        if k != "endpoint":
            err, ref = a["ade"][use], a["baseline_ade"][use]
            score.update(ade_sum=float(err.sum()), cv_sum=float(ref.sum()),
                         harm_sum=float(np.maximum(err - ref, 0.).sum()),
                         p95=float(np.quantile(err, .95)) if n else None)
        scores[k] = score
    return dict(slices=scores, edits=int(active.sum()),
                peak_sum=float(peak(pred - cv).sum()),
                lower_gain=float(a["gain_lower"].sum()), upper_gain=float(a["gain_upper"].sum()))


def paired_summary(ref, pred, y, valid, scale, easy_cut, hard_cut):
    a = score_arrays(ref, pred, y, valid, scale, easy_cut=easy_cut, hard_cut=hard_cut)
    on = a["complete"]
    return dict(lower=float(a["gain_lower"].sum()), upper=float(a["gain_upper"].sum()),
                complete_gain=float((a["baseline_ade"][on] - a["ade"][on]).sum()),
                complete_ref=float(a["baseline_ade"][on].sum()))
