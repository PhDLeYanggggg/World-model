"""Recording-local forecast metrics with the unchanged 2% support contract."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors, fraction, policy


def comparison_masses(left, right, denominator, target, valid, weights):
    a, _ = errors(left, target, valid); b, _ = errors(right, target, valid)
    base, _ = errors(denominator, target, valid); known = valid.any(1)
    disagreement = np.linalg.norm(right[~known]-left[~known], axis=-1).max(1)
    return dict(reference=float(weights[known]@base[known]),
        known_delta=float(weights[known]@(a[known]-b[known])),
        unknown_radius=float(weights[~known]@disagreement))


def merge_comparisons(blocks):
    mass, delta, radius = (sum(b[k] for b in blocks) for k in ('reference','known_delta','unknown_radius'))
    return dict(reference=mass, known_delta=delta, unknown_radius=radius,
        known_pp=fraction(100*delta, mass), completion_lower_pp=fraction(100*(delta-radius), mass),
        completion_upper_pp=fraction(100*(delta+radius), mass))


def score_recording(paths, pairs, target, valid, queries, recording, *, easy_cut, hard_cut):
    """Call the original scorer only after candidate paths and choices are fixed."""
    if len(queries) != len(valid):
        raise ValueError('Aligned query and label masks required')
    if len(valid) == 0:
        return dict(recording=recording, rows=0, queries=0, status='no_requested_rows',
            metrics={k:None for k in paths}, comparisons={k:dict(reference=0., known_delta=0., unknown_radius=0.) for k in pairs})
    floor = paths['constant_velocity_causal_fd']
    _, index, count = np.unique(queries, return_inverse=True, return_counts=True)
    weight = 1./count[index]
    metrics = {}
    for key, path in paths.items():
        result = policy(path, floor, floor, target, valid, np.full(len(valid), recording),
                        easy_cut=easy_cut, hard_cut=hard_cut)
        metrics[key] = dict(risk=result.pop('recordings')[recording], **result)
    return dict(recording=recording, rows=len(valid), queries=len(count), status='scored', metrics=metrics,
        comparisons={key:comparison_masses(paths[left], paths[right], paths[denom], target, valid, weight)
            for key,(left,right,denom) in pairs.items()})
