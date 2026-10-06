"""TRAIN-only component diagnosis; never produces a new deployment threshold."""
import math
import numpy as np

from src.world_model.m3w_component_calibration import eligible

BUDGET = .02


def moving_from_features(x):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 380 or not np.isfinite(x).all():
        raise ValueError('Frozen finite 380-column causal schema required')
    # These are the normalized CV rollout, not the selected floor or labels.
    return np.any(x[:, 355:379] != 0, axis=1)


def quantiles(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError('Do not silently drop nonfinite observations')
    return dict(count=len(x), q=np.quantile(x, [0, .25, .5, .75, 1]).tolist() if len(x) else None)


def components(pred, truth):
    p, y = np.asarray(pred, float), np.asarray(truth, float)
    if (p.shape != y.shape or p.ndim != 2 or p.shape[1] != 5
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or np.any(p < 0) or np.any(y < 0)):
        raise ValueError('Paired finite nonnegative five-moment costs required')
    harm = y[:, 4]-p[:, 4]
    reference = BUDGET*(p[:, 3]-y[:, 3])
    slack = p[:, 4]-BUDGET*p[:, 3]
    actual = y[:, 4]-BUDGET*y[:, 3]
    np.testing.assert_allclose(actual, slack+harm+reference, rtol=1e-10, atol=1e-9)
    # Independent scalar check of the decomposition, including negative terms.
    scalar = [math.fsum(float(v) for v in a) for a in (slack, harm, reference, actual)]
    assert math.isclose(scalar[3], math.fsum(scalar[:3]), rel_tol=1e-10, abs_tol=1e-8)
    den = float(y[:, 3].sum())
    return dict(rows=len(y), predicted_easy_harm_mass=float(p[:, 4].sum()),
        actual_easy_harm_mass=float(y[:, 4].sum()), predicted_easy_reference_mass=float(p[:, 3].sum()),
        actual_easy_reference_mass=den, predicted_slack_mass=scalar[0],
        harm_underprediction_mass=scalar[1], reference_overprediction_budget_mass=scalar[2],
        realized_excess_mass=scalar[3], known_positive_easy_risk=float(y[:, 4].sum()/den) if den > 0 else None,
        actual_easy_rows=int((y[:, 3] > 0).sum()),
        row_false_safe=int(((slack <= 0) & (actual > 0)).sum()),
        harm_underpredicted_rows=int((harm > 0).sum()), reference_overpredicted_rows=int((reference > 0).sum()),
        scalar_checks=4*len(y)+1)


def grouped(pred, truth, selected_known, keys):
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(tuple(key), []).append(i)
    risks, excess, no_reference = [], [], 0
    with_selection = bad = 0
    for ids in groups.values():
        ids = np.asarray(ids); ids = ids[selected_known[ids]]
        if not len(ids):
            continue
        with_selection += 1
        p, y = pred[ids], truth[ids]
        e = float((y[:, 4]-BUDGET*y[:, 3]).sum())
        excess.append(e); bad += int(e > 0)
        assert float((p[:, 4]-BUDGET*p[:, 3]).sum()) <= 1e-8
        den = float(y[:, 3].sum())
        if den > 0:
            risks.append(float(y[:, 4].sum()/den))
        else:
            no_reference += 1
    return dict(total_groups=len(groups), selected_known_groups=with_selection,
        positive_excess_groups=bad, zero_reference_groups=no_reference,
        known_easy_risk=quantiles(risks), realized_excess=quantiles(excess))


def diagnose(pred, truth, moving, support, recordings, frames, scale):
    p, y = np.asarray(pred, float), np.asarray(truth, float)
    if p.shape != y.shape or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Aligned moments and positive TRAIN scale required')
    known = np.isfinite(y).all(1)
    if np.isinf(y).any() or not np.array_equal(np.isnan(y).all(1), ~known):
        raise ValueError('Whole-row unknown support required')
    take = eligible(p, moving, support)
    selected = take & known
    rec, frame = np.asarray(recordings), np.asarray(frames)
    if rec.shape != (len(y),) or frame.shape != rec.shape:
        raise ValueError('Aligned query and recording keys required')
    out = dict(rows=len(y), known=int(known.sum()), unknown=int((~known).sum()),
        moving=int(np.asarray(moving).sum()), supported=int(np.asarray(support).sum()),
        selected=int(take.sum()), selected_known=int(selected.sum()), selected_unknown=int((take & ~known).sum()),
        known_selected=components(p[selected]/scale, y[selected]/scale),
        known_all=components(p[known]/scale, y[known]/scale),
        query=grouped(p/scale, y/scale, selected, zip(rec.tolist(), frame.tolist())),
        recording=grouped(p/scale, y/scale, selected, ((r,) for r in rec.tolist())),
        resubstitution_only=True, inference_label_filter=False, budget=BUDGET,
        normalized_by_frozen_TRAIN_cost_scale=True)
    # Replacement of labels in policy scores is an oracle diagnostic, not a policy.
    out['label_replacement_diagnostic_only'] = dict(
        harm_only_positive_excess_rows=int(((y[selected, 4]-BUDGET*p[selected, 3]) > 0).sum()),
        reference_only_positive_excess_rows=int(((p[selected, 4]-BUDGET*y[selected, 3]) > 0).sum()),
        actual_positive_excess_rows=int(((y[selected, 4]-BUDGET*y[selected, 3]) > 0).sum()))
    return out
