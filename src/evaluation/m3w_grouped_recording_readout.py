"""Aggregate frozen recording scores by source, retaining unsupported views."""
from collections import defaultdict

import numpy as np

from src.evaluation.m3w_recording_forecast_comparison import merge_comparisons


def grouped_readout(rows, roster, seeds, *, resamples=3000, random_seed=73107):
    """Merge query-weighted masses within groups, then average seeds and groups."""
    if not roster or not seeds or len(set(seeds)) != len(seeds) or resamples < 1:
        raise ValueError('A fixed nonempty recording/seed roster is required')
    expected = {(rec, seed) for rec in roster for seed in seeds}
    keys = {(r['recording'], r['seed']) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError('Missing or duplicate recording/seed views')
    groups = sorted(set(roster.values()))
    by_cell = defaultdict(list)
    pairs, arms = tuple(rows[0]['comparisons']), tuple(rows[0]['metrics'])
    for r in rows:
        group = roster[r['recording']]
        if (r['source'] != group or set(r['comparisons']) != set(pairs)
                or set(r['metrics']) != set(arms)):
            raise ValueError('Source or frozen arm roster changed')
        by_cell[group, r['seed']].append(r)
    draws = np.random.default_rng(random_seed).integers(len(groups), size=(resamples, len(groups)))
    effects = {}
    for pair in pairs:
        combined = [merge_comparisons([r['comparisons'][pair] for r in by_cell[g, s]])
                    for g in groups for s in seeds]
        result = {}
        for metric in ('known_pp', 'completion_lower_pp', 'completion_upper_pp'):
            values = [v[metric] for v in combined]
            missing = sum(v is None for v in values)
            if missing:
                result[metric] = dict(point=None, interval95=None, missing_source_seed_cells=missing)
            else:
                x = np.asarray(values, float).reshape(len(groups), len(seeds)).mean(1)
                result[metric] = dict(point=float(x.mean()),
                    interval95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    missing_source_seed_cells=0)
        effects[pair] = result
    risk = {}
    for arm in arms:
        all_metrics = [r['metrics'][arm] for r in rows]
        a = [v['risk'] for v in all_metrics if v is not None]
        signed = [v['known_easy_net_degradation'] for v in a if v['known_easy_net_degradation'] is not None]
        tails = [v['p95_known_ADE_pixel'] for v in all_metrics
                 if v is not None and v['p95_known_ADE_pixel'] is not None]
        switch = sum(v['selected'] for v in a)
        n = sum(v['rows'] for v in a)
        risk[arm] = dict(required_views=len(rows), scored_views=len(a),
            supported_views=sum(v['supported'] for v in a),
            undefined_easy_views=len(rows)-sum(v['completion_easy_risk_upper'] is not None for v in a),
            all_risk_over_2_views=sum(v['completion_all_risk_upper'] is not None
                                     and v['completion_all_risk_upper'] > .02 for v in a),
            signed_easy_over_2_views=sum(v > .02 for v in signed),
            worst_signed_easy_percent=100*max(signed) if signed else None,
            worst_recording_p95_known_ade_pixel=max(tails) if tails else None,
            switches=switch, switch_rate=switch/n if n else None)
    return dict(recordings=len(roster), source_groups=len(groups), views=len(rows), seeds=list(seeds),
        bootstrap_draws=resamples, bootstrap_unit='source_group_after_seed_average',
        effects=effects, risk=risk)
