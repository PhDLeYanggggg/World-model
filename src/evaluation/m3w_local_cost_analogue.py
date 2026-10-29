"""Loss-aligned local cost regression on frozen, policy-selected donors."""
import hashlib
import json
import statistics

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.evaluation import m3w_selection_conditioned_analogue as previous

base = previous.base
MODES = ('selected_nearest32', 'weighted_harm32', 'local_signed32',
         'global_signed_mean', 'shuffled_signed32')
ARMS = ('origin',) + MODES


def solve(target_fraction, weights):
    target_fraction, weights = np.asarray(target_fraction), np.asarray(weights)
    if (target_fraction.shape[:-1] != weights.shape or not np.isfinite(target_fraction).all()
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError('Finite targets and positive quadratic weights required')
    return np.clip((target_fraction*weights[..., None]).sum(-2)/weights.sum(-1)[..., None], 0, 1)


def retrieve(bank, query, frozen, *, k=32, heartbeat=lambda **_: None):
    x, env, y, recs, eligible, scales, sources = (np.asarray(bank[key]) for key in
        ('x', 'envelope', 'labels', 'recordings', 'eligible', 'scale', 'sources'))
    query, frozen = np.asarray(query), np.asarray(frozen)
    n = len(x)
    if (x.ndim != 2 or not np.isfinite(x).all() or env.shape != (n,)
            or not np.isfinite(env).all() or (env < 0).any() or y.shape != (n, 5)
            or recs.shape != (n,) or eligible.shape != (n,) or sources.shape != (n,)
            or query.shape != (n,) or query.dtype != bool or eligible.dtype != bool
            or (query & ~eligible).any() or k < 1 or scales.shape != (n,)
            or not np.isfinite(scales).all() or (scales <= 0).any()
            or frozen.shape != (n, 5) or not np.isfinite(frozen).all()):
        raise ValueError('Aligned causal arrays, positive source scale and frozen predictions required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all(): raise ValueError('Partial future support forbidden')
    causal = eligible & (env > 0)
    labels = causal & known & query
    fraction = np.zeros((n, 2)); signed = fraction.copy()
    fraction[labels] = y[labels][:, [1, 4]]/env[labels, None]
    if (fraction[labels] < -1e-8).any() or (fraction[labels] > 1+1e-5).any():
        raise ValueError('Harm outside causal envelope')
    fraction = np.clip(fraction, 0, 1)
    # Frozen donor reference errors are offsets in the evaluated signed-risk loss.
    signed[labels] = (y[labels][:, [1, 4]] + .02*(frozen[labels][:, [2, 3]]
                        - y[labels][:, [2, 3]]))/env[labels, None]
    shuffled = signed.copy()
    for rec in sorted(set(recs[labels].tolist())):
        ix = np.flatnonzero(labels & (recs == rec))
        salt = int.from_bytes(hashlib.sha256(('20261029|'+str(rec)).encode()).digest()[:8], 'little')
        shuffled[ix] = signed[np.random.default_rng(salt).permutation(ix)]
    indices = np.flatnonzero(query); position = np.full(n, -1, int)
    position[indices] = np.arange(len(indices))
    predictions = {a: np.full((len(indices), 2), np.nan) for a in MODES}
    support = np.zeros(len(indices), bool); references = []
    diagnostic = {key: np.full(len(indices), np.nan) for key in ('effective_donors', 'max_weight')}
    donor_loss = {key: np.full((len(indices), 2), np.nan) for key in ('uniform', 'optimum')}
    with threadpool_limits(limits=4):
        for rec in sorted(set(recs[query].tolist())):
            ix = np.flatnonzero(query & (recs == rec))
            norm = np.flatnonzero(causal & (recs != rec))
            donors = np.flatnonzero(labels & (recs != rec))
            counts = {s: int((sources[donors] == s).sum()) for s in sorted(set(sources.tolist()))}
            source, local = json.loads(rec)
            r = dict(recording=rec, query_source=source, local_recording=local,
                query_rows=len(ix), causal_donors=len(norm), known_donors=len(donors),
                donor_recordings=len(set(recs[donors].tolist())),
                query_sha256=base.fingerprint(ix), donor_sha256=base.fingerprint(donors),
                same_recording_donors=0, unselected_donors=int((~query[donors]).sum()),
                known_donors_by_source=counts, supported=len(donors) >= k)
            if len(donors) < k: references.append(r); continue
            mean = x[norm].astype(float).mean(0)
            std = np.maximum(x[norm].astype(float).std(0), 1e-6)
            z = np.clip((x[donors]-mean)/std, -8, 8).astype(np.float32)
            search = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=1).fit(z)
            weights = np.zeros(n)
            weights[donors] = (env[donors]/scales[donors])**2 / np.array([counts[s] for s in sources[donors]])
            global_fit = solve(signed[donors], weights[donors])
            trace = hashlib.sha256()
            for start in range(0, len(ix), 256):
                q = ix[start:start+256]; pos = position[q]
                qz = np.clip((x[q]-mean)/std, -8, 8).astype(np.float32)
                _, neighbor = search.kneighbors(qz); chosen = donors[neighbor]
                if np.any(recs[chosen] == rec) or not query[chosen].all():
                    raise AssertionError('Reference exclusion or frozen selection changed')
                trace.update(chosen.tobytes())
                w = weights[chosen]; f = fraction[chosen]; t = signed[chosen]
                uniform = f.mean(1); opt = solve(t, w)
                fractions = dict(selected_nearest32=uniform, weighted_harm32=solve(f, w),
                    local_signed32=opt, global_signed_mean=global_fit,
                    shuffled_signed32=solve(shuffled[chosen], w))
                for arm, value in fractions.items(): predictions[arm][pos] = env[q, None]*value
                wn = w/w.sum(1, keepdims=True)
                diagnostic['effective_donors'][pos] = 1/(wn**2).sum(1)
                diagnostic['max_weight'][pos] = wn.max(1)
                donor_loss['uniform'][pos] = (wn[..., None]*(uniform[:, None]-t)**2).sum(1)
                donor_loss['optimum'][pos] = (wn[..., None]*(opt[:, None]-t)**2).sum(1)
                if np.any(donor_loss['optimum'][pos] > donor_loss['uniform'][pos]+1e-12):
                    raise AssertionError('Closed-form constrained optimum fails donor loss check')
                support[pos] = True
                heartbeat(state='analogue_queries', recording=rec,
                    completed=min(start+256, len(ix)), total=len(ix))
            r.update(mean_sha256=base.fingerprint(mean), std_sha256=base.fingerprint(std),
                neighbor_sha256=trace.hexdigest(), weight_sha256=base.fingerprint(weights[donors]),
                corrected_fraction_outside_bounds=((signed[donors]<0)|(signed[donors]>1)).sum(0).tolist())
            references.append(r)
    return dict(indices=indices, predictions=predictions, supported=support,
                references=references, diagnostics=diagnostic, donor_loss=donor_loss)


def summarize(rows, expected):
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(keys) != 72 or len(set(keys)) != 72 or set(keys) != set(expected):
        raise ValueError('Complete72 frozen views required')
    means = {a: {t: statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
        if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
        for t in ('all', 'easy')} for a in ARMS}
    wins = 0
    for context, seed in sorted({(r['context'], r['seed']) for r in rows}):
        group = [r for r in rows if (r['context'], r['seed']) == (context, seed)]
        values = {a: [r['scores'][a]['easy']['full_cohort_MSE'] for r in group]
                  for a in ('origin', 'selected_nearest32', 'local_signed32')}
        if all(None not in v for v in values.values()):
            wins += statistics.fmean(values['local_signed32']) < min(
                statistics.fmean(values['origin']), statistics.fmean(values['selected_nearest32']))-1e-12
    new, old, origin = (means[a] for a in ('local_signed32', 'selected_nearest32', 'origin'))
    conditions = dict(all_views_supported=all(r['scores']['local_signed32']['unsupported'] == 0 for r in rows),
        easy_MSE_beats_origin=new['easy'] is not None and new['easy'] < origin['easy']-1e-12,
        easy_MSE_beats_selected_nearest=new['easy'] is not None and new['easy'] < old['easy']-1e-12,
        all_MSE_not_worse_than_selected_nearest=new['all'] is not None and new['all'] <= old['all']+1e-12,
        at_least12_of18_pair_wins=wins >= 12)
    return dict(views=72, equal_view_MSE=means, pair_wins=int(wins), magnitude_screen=conditions,
        magnitude_screen_pass=all(conditions.values()),
        occurrences={k:sum(r['scores']['local_signed32'][k] for r in rows) for k in
                     ('rows', 'known', 'unknown', 'supported', 'unsupported')},
        fitting_diagnostic_only=True, whole_pipeline_OOF=False, held_inference=False,
        independent_roles_read=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
