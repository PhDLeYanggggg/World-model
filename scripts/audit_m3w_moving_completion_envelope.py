"""Separate displacement and missing-label limits on frozen Moving development."""

from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement import moving_completion
from scripts import package_m3w_moving_peak_certificate as prior

np, verify = prior.np, prior.verify
OUT = prior.OUT.parent/'moving_completion_envelope_v1'
favorable = partial(moving_completion.favorable, peak=verify)
bounds = partial(moving_completion.bounds, peak=verify)


def segments(error, valid):
    """Slopes of sum(min(radius, known_error)) plus missing_count * radius."""
    if error.shape != valid.shape or error.ndim != 2 or error.shape[1] != verify.HORIZON or not len(error):
        raise ValueError('Expected a nonempty query on the frozen prediction grid')
    if not np.isfinite(error).all() or np.any(error < 0) or np.any(error[~valid] != 0.):
        raise ValueError('Invalid observed error')
    values = np.sort(error, axis=1)
    increments = np.diff(values, axis=1, prepend=0.)
    missing = (~valid).sum(1)
    slope = np.arange(verify.HORIZON, 0, -1)[None, :] + missing[:, None]
    widths = np.bincount(slope[increments > 0.], weights=increments[increments > 0.], minlength=13)
    if len(widths) != 13:
        raise ValueError('An observed segment overlaps the missing-step ray')
    return widths[1:].tolist(), int(missing.max())


def aggregate(cv, ols, target, valid, query, radii):
    base = prior.aggregate(cv, ols, target, valid, query, radii)
    ids, group, sizes = np.unique(query, return_inverse=True, return_counts=True)
    target = np.where(valid[..., None], target, 0.)
    error = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    queries = []
    for i, n in enumerate(sizes):
        at = group == i
        lengths, ray = segments(error[at], valid[at])
        queries.append(dict(actors=int(n), budget=float(radii[i]), favorable_lengths=lengths,
                            favorable_ray=ray))
    if len(ids) != len(base['queries']):
        raise ValueError('Query grouping differs from the archived bound')
    base['completion_queries'] = queries
    return base


def run():
    payload, expected, hashes = prior.export(aggregate_view=aggregate)
    old = verify.replay(payload)
    prior.check(old, expected)
    return dict(**moving_completion.report(payload['views'], old, verify),
                archive_input_sha256=hashes, result_source='fresh_post_hoc_bound_on_frozen_exposed_development',
                original_2pct_unchanged=True, independent_confirmation=False, external_outcomes_read=False,
                learned_policy=False, risk_evaluated=False, submission_ready=False)


if __name__ == '__main__':
    prior.audit.torch.set_num_threads(4)
    prior.audit.torch.set_num_interop_threads(1)
    result = prior.old.stable(run())
    OUT.mkdir(parents=True, exist_ok=True)
    prior.old.once(OUT/'receipt.json', result)
    print(json.dumps(result, indent=2))
