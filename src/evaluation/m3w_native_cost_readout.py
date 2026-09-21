"""Descriptive conditional-cost checks, not fitted calibration or deployment."""
import numpy as np


def fixed_interventions(scores, same):
    p, same = np.asarray(scores), np.asarray(same)
    if p.ndim != 2 or p.shape[1] != 2 or same.shape != (len(p),) or same.dtype != bool or not np.isfinite(p).all():
        raise ValueError('Finite paired benefit/harm scores and causal equality mask required')
    positive = (p[:, 0] > p[:, 1]) & ~same
    return dict(positive_gain=positive, harm_fraction_0p1=positive & (p[:, 1] <= .1*p[:, 0]))


def conditional_report(scores, benefit, harm, same):
    p, b, h = np.asarray(scores), np.asarray(benefit), np.asarray(harm)
    masks = fixed_interventions(p, same)
    if (b.shape != (len(p),) or h.shape != b.shape or np.isinf(b).any() or np.isinf(h).any()
            or not np.array_equal(np.isnan(b), np.isnan(h))
            or np.any(b[np.isfinite(b)] < 0) or np.any(h[np.isfinite(h)] < 0)):
        raise ValueError('Aligned supported nonnegative supervision required')
    known = np.isfinite(b); truth = np.column_stack((b, h))
    def group(mask):
        use = known & mask
        if not use.any():
            return dict(indexed_rows=int(mask.sum()), supported_rows=0,
                unknown_rows=int((mask & ~known).sum()), realized_gain=None, realized_harm=None,
                predicted_harm=None, harm_underprediction_fraction=None)
        return dict(indexed_rows=int(mask.sum()), supported_rows=int(use.sum()),
            unknown_rows=int((mask & ~known).sum()), realized_gain=float((b[use]-h[use]).mean()),
            realized_harm=float(h[use].mean()), predicted_harm=float(p[use, 1].mean()),
            harm_underprediction_fraction=float((p[use, 1] < h[use]).mean()))
    rows = np.flatnonzero(known)
    ordered = rows[np.argsort(p[rows, 1], kind='stable')]
    deciles = []
    for ids in np.array_split(ordered, 10):
        if len(ids):
            mask = np.zeros(len(p), bool); mask[ids] = True
            deciles.append(group(mask))
    return dict(rows=len(p), supported_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        benefit_mse=float(np.mean((p[known, 0]-b[known])**2)) if known.any() else None,
        harm_mse=float(np.mean((p[known, 1]-h[known])**2)) if known.any() else None,
        gain_mae=float(np.mean(np.abs((p[known, 0]-p[known, 1])-(b[known]-h[known])))) if known.any() else None,
        negative_harm_predictions=int((p[:, 1] < 0).sum()),
        groups={k:group(v) for k,v in dict(all=np.ones(len(p), bool), **masks).items()},
        descriptive_harm_deciles=deciles, calibrated_probability=False)
