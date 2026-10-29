"""Fixed-policy donor conditioning, with normalization and queries unchanged."""
import hashlib
import json
import statistics

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.evaluation import m3w_selected_analogue as base

MODES = ('mean', 'nearest32', 'shuffled32')
ARMS = ('origin', 'estimation', 'pooled_mean', 'pooled_nearest32',
        'pooled_shuffled32', 'selected_mean', 'selected_nearest32', 'selected_shuffled32')


def retrieve(bank, query, *, selected, k=32, heartbeat=lambda **_: None):
    x, env, y, recs, eligible = (np.asarray(bank[key]) for key in
        ('x', 'envelope', 'labels', 'recordings', 'eligible'))
    query = np.asarray(query)
    n = len(x)
    if (x.ndim != 2 or not np.isfinite(x).all() or env.shape != (n,) or
            not np.isfinite(env).all() or (env < 0).any() or y.shape != (n, 5) or
            recs.shape != (n,) or eligible.shape != (n,) or query.shape != (n,) or
            query.dtype != bool or eligible.dtype != bool or (query & ~eligible).any() or
            not isinstance(selected, bool) or k < 1):
        raise ValueError('Aligned causal inputs and frozen eligible policy mask required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all():
        raise ValueError('Partial future support forbidden')
    causal = eligible & (env > 0)
    labels = causal & known & (query if selected else True)
    fractions = np.zeros((n, 2))
    fractions[labels] = y[labels][:, [1, 4]]/env[labels, None]
    if (fractions[labels] < -1e-8).any() or (fractions[labels] > 1+1e-5).any():
        raise ValueError('Harm outside causal envelope')
    fractions = np.clip(fractions, 0, 1)
    shuffled = fractions.copy()
    for rec in sorted(set(recs[labels].tolist())):
        ix = np.flatnonzero(labels & (recs == rec))
        salt = int.from_bytes(hashlib.sha256(('20261029|'+str(rec)).encode()).digest()[:8], 'little')
        shuffled[ix] = fractions[np.random.default_rng(salt).permutation(ix)]
    indices = np.flatnonzero(query)
    position = np.full(n, -1, int); position[indices] = np.arange(len(indices))
    predictions = {m: np.full((len(indices), 2), np.nan) for m in MODES}
    probabilities = np.full((len(indices), 2), np.nan)
    severity = probabilities.copy()
    supported = np.zeros(len(indices), bool); receipts = []
    with threadpool_limits(limits=4):
        for rec in sorted(set(recs[query].tolist())):
            ix = np.flatnonzero(query & (recs == rec))
            # Selection changes supervised donors only, never distance normalization.
            norm = np.flatnonzero(causal & (recs != rec))
            donors = np.flatnonzero(labels & (recs != rec))
            source, local = json.loads(rec)
            r = dict(recording=rec, query_source=source, local_recording=local,
                query_rows=len(ix), causal_donors=len(norm), known_donors=len(donors),
                donor_recordings=len(set(recs[donors].tolist())),
                query_sha256=base.fingerprint(ix), donor_sha256=base.fingerprint(donors),
                same_recording_donors=0, supported=len(donors) >= k,
                unselected_donors=int((~query[donors]).sum()),
                known_donors_by_source={s: int((bank['sources'][donors] == s).sum())
                                       for s in sorted(set(bank['sources'].tolist()))},
                donor_events=(fractions[donors] > 0).sum(0).tolist(),
                donor_fraction_sums=fractions[donors].sum(0).tolist())
            if len(donors) < k:
                receipts.append(r); continue
            mean = x[norm].astype(float).mean(0)
            std = np.maximum(x[norm].astype(float).std(0), 1e-6)
            z = np.clip((x[donors]-mean)/std, -8, 8).astype(np.float32)
            search = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=1).fit(z)
            donor_mean = fractions[donors].mean(0)
            trace = hashlib.sha256(); distance_sum = 0.
            for start in range(0, len(ix), 256):
                q = ix[start:start+256]; pos = position[q]
                qz = np.clip((x[q]-mean)/std, -8, 8).astype(np.float32)
                distance, neighbor = search.kneighbors(qz); chosen = donors[neighbor]
                if np.any(recs[chosen] == rec) or (selected and not query[chosen].all()):
                    raise AssertionError('Reference exclusion/selection violated')
                trace.update(chosen.tobytes()); distance_sum += float(distance.sum())
                f = fractions[chosen]; event = f > 0
                p = event.mean(1); count = event.sum(1)
                c = np.divide(f.sum(1), count, out=np.zeros_like(p), where=count > 0)
                probabilities[pos], severity[pos] = p, c
                predictions['mean'][pos] = env[q, None]*donor_mean
                predictions['nearest32'][pos] = env[q, None]*f.mean(1)
                predictions['shuffled32'][pos] = env[q, None]*shuffled[chosen].mean(1)
                np.testing.assert_allclose(p*c, f.mean(1), atol=1e-14)
                supported[pos] = True
                heartbeat(state='analogue_queries', bank='selected' if selected else 'full',
                    recording=rec, completed=min(start+256, len(ix)), total=len(ix))
            r.update(mean_sha256=base.fingerprint(mean), std_sha256=base.fingerprint(std),
                neighbor_sha256=trace.hexdigest(), mean_neighbor_distance=distance_sum/(len(ix)*k))
            receipts.append(r)
    return dict(indices=indices, predictions=predictions, probability=probabilities,
                severity=severity, supported=supported, references=receipts)


def decomposition(y, prediction, probability, severity, env, scale, support):
    known = np.isfinite(y).all(1) & support
    out = {}
    for j, (target, col) in enumerate((('all', 1), ('easy', 4))):
        truth = y[known, col]; pred = prediction[known, j]
        event = truth > 0; p = probability[known, j]; c = severity[known, j]
        squared = ((pred-truth)/scale)**2
        positive_supported = event & (p > 0)
        f = np.divide(truth, env[known], out=np.zeros_like(truth), where=env[known] > 0)
        out[target] = dict(known=int(known.sum()), events=int(event.sum()),
            predicted_event_mass=float(p.sum()), brier_sum=float(((p-event)**2).sum()),
            event_harm_SSE=float(squared[event].sum()), non_event_harm_SSE=float(squared[~event].sum()),
            harm_SSE=float(squared.sum()), predicted_harm_on_events=float(pred[event].sum()/scale),
            predicted_harm_on_non_events=float(pred[~event].sum()/scale),
            observed_harm=float(truth.sum()/scale),
            positive_queries_with_positive_donors=int(positive_supported.sum()),
            positive_queries_without_positive_donors=int((event & (p == 0)).sum()),
            positive_severity_SSE=float(((c-f)[positive_supported]**2).sum()),
            note='Labels used only for scoring; no oracle event/severity enters prediction')
    return out


def summarize(rows, expected):
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(keys) != 72 or len(set(keys)) != 72 or set(keys) != set(expected):
        raise ValueError('Complete72 frozen query views required')
    means = {a: {t: statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
                 if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
                 for t in ('all', 'easy')} for a in ARMS}
    pairs = sorted({(r['context'], r['seed']) for r in rows}); wins = 0
    for context, seed in pairs:
        group = [r for r in rows if (r['context'], r['seed']) == (context, seed)]
        scores = {a: [r['scores'][a]['easy']['full_cohort_MSE'] for r in group]
                  for a in ('origin', 'pooled_nearest32', 'selected_nearest32')}
        if all(None not in v for v in scores.values()):
            wins += statistics.fmean(scores['selected_nearest32']) < min(
                statistics.fmean(scores['origin']), statistics.fmean(scores['pooled_nearest32']))-1e-12
    n, o, f = (means[a] for a in ('selected_nearest32', 'origin', 'pooled_nearest32'))
    conditions = dict(all_views_supported=all(r['scores']['selected_nearest32']['unsupported'] == 0 for r in rows),
        easy_MSE_beats_origin=n['easy'] is not None and n['easy'] < o['easy']-1e-12,
        easy_MSE_beats_full_bank=n['easy'] is not None and n['easy'] < f['easy']-1e-12,
        all_MSE_not_worse_than_full_bank=n['all'] is not None and n['all'] <= f['all']+1e-12,
        at_least12_of18_pair_wins=wins >= 12)
    return dict(views=72, equal_view_MSE=means, pair_wins=int(wins), magnitude_screen=conditions,
        magnitude_screen_pass=all(conditions.values()),
        occurrences={k: sum(r['scores']['selected_nearest32'][k] for r in rows) for k in
                     ('rows', 'known', 'unknown', 'supported', 'unsupported')},
        fitting_diagnostic_only=True, whole_pipeline_OOF=False, held_inference=False,
        independent_roles_read=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
