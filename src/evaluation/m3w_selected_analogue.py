"""Recording-excluded causal analogues: fitting information probe, not a policy."""
import hashlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

MODES = ('source_mean', 'nearest32', 'shuffled_nearest32')


def fingerprint(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def retrieve(x, envelope, labels, recordings, eligible, query_mask, *, k=32, seed=20261029, heartbeat=lambda **_: None):
    """Each query recording is absent from its reference statistics and labels."""
    x, env, y, recs, eligible, query = map(np.asarray,
        (x, envelope, labels, recordings, eligible, query_mask))
    n = len(x)
    if (x.ndim != 2 or not np.isfinite(x).all() or env.shape != (n,) or
            not np.isfinite(env).all() or (env < 0).any() or y.shape != (n, 5) or
            recs.shape != (n,) or query.shape != (n,) or eligible.shape != (n,) or
            query.dtype != bool or eligible.dtype != bool or (query & ~eligible).any() or k <= 0):
        raise ValueError('Finite causal inputs and a frozen eligible query mask required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all(): raise ValueError('Partly known moments forbidden')
    target_indices = np.flatnonzero(query)
    position = np.full(n, -1, dtype=int); position[target_indices] = np.arange(len(target_indices))
    outputs = {a: np.full((len(target_indices), 2), np.nan) for a in MODES}
    support = np.zeros(len(target_indices), bool); receipts = []
    causal_bank = eligible & (env > 0)
    label_bank = causal_bank & known
    fractions = np.zeros((n, 2)); fractions[label_bank] = y[label_bank][:, [1, 4]]/env[label_bank, None]
    if (fractions[label_bank] < -1e-8).any() or (fractions[label_bank] > 1+1e-5).any():
        raise ValueError('Harm labels exceed causal disagreement envelope')
    fractions = np.clip(fractions, 0, 1)
    shuffled = fractions.copy()
    # Within-recording permutation preserves its label distribution and support.
    for rec in sorted(set(recs[label_bank].tolist())):
        ix = np.flatnonzero(label_bank & (recs == rec))
        salt = int.from_bytes(hashlib.sha256((str(seed)+'|'+str(rec)).encode()).digest()[:8], 'little')
        shuffled[ix] = fractions[np.random.default_rng(salt).permutation(ix)]
    with threadpool_limits(limits=4):
        for rec in sorted(set(recs[query].tolist())):
            queries = np.flatnonzero(query & (recs == rec))
            causal = np.flatnonzero(causal_bank & (recs != rec))
            donors = np.flatnonzero(label_bank & (recs != rec))
            receipt = dict(recording=str(rec), query_rows=len(queries), causal_donors=len(causal),
                known_donors=len(donors), donor_recordings=len(set(recs[donors].tolist())),
                query_sha256=fingerprint(queries), donor_sha256=fingerprint(donors),
                same_recording_donors=0, supported=len(donors) >= k)
            if len(donors) < k:
                receipts.append(receipt); continue
            if np.any(recs[donors] == rec): raise AssertionError('Target recording entered reference bank')
            mean = x[causal].astype(float).mean(0)
            std = np.maximum(x[causal].astype(float).std(0), 1e-6)
            z = np.clip((x[donors]-mean)/std, -8, 8).astype(np.float32)
            search = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=1).fit(z)
            mean_fraction = fractions[donors].mean(0)
            trace = hashlib.sha256(); distance_sum = 0.
            for start in range(0, len(queries), 256):
                ix = queries[start:start+256]; pos = position[ix]
                qz = np.clip((x[ix]-mean)/std, -8, 8).astype(np.float32)
                distance, neighbor = search.kneighbors(qz)
                chosen = donors[neighbor]
                if np.any(recs[chosen] == rec): raise AssertionError('Same-recording nearest neighbor')
                trace.update(chosen.tobytes()); distance_sum += float(distance.sum())
                outputs['source_mean'][pos] = env[ix, None]*mean_fraction
                outputs['nearest32'][pos] = env[ix, None]*fractions[chosen].mean(1)
                outputs['shuffled_nearest32'][pos] = env[ix, None]*shuffled[chosen].mean(1)
                support[pos] = True
                heartbeat(state='analogue_queries', recording=str(rec), completed=min(start+256,len(queries)), total=len(queries))
            receipt.update(mean_sha256=fingerprint(mean), std_sha256=fingerprint(std),
                neighbor_sha256=trace.hexdigest(), mean_neighbor_distance=distance_sum/(len(queries)*k))
            receipts.append(receipt)
    return dict(indices=target_indices, predictions=outputs, supported=support, references=receipts)


def score(y, prediction, envelope, scale, supported):
    known = np.isfinite(y).all(1); observed = known & supported
    result = dict(rows=len(y), known=int(known.sum()), unknown=int((~known).sum()),
        supported=int(supported.sum()), unsupported=int((~supported).sum()),
        supported_known=int(observed.sum()), supported_unknown=int((supported & ~known).sum()))
    for name, col, reference in (('all', 1, 2), ('easy', 4, 3)):
        error = ((prediction[observed,col]-.02*prediction[observed,reference])-
                 (y[observed,col]-.02*y[observed,reference]))/scale
        event = y[observed,col] > 0
        rank = prediction[observed,col]/np.maximum(envelope[observed], 1e-150)
        result[name] = dict(full_cohort_MSE=float(np.mean(error**2)) if observed.any() and supported.all() else None,
            supported_only_MSE=float(np.mean(error**2)) if observed.any() else None,
            events=int(event.sum()), event_prevalence=float(event.mean()) if len(event) else None,
            AUROC=float(roc_auc_score(event, rank)) if len(set(event.tolist())) == 2 else None,
            AP=float(average_precision_score(event, rank)) if event.any() else None,
            observed_harm=float(y[observed,col].sum()/scale), predicted_harm=float(prediction[observed,col].sum()/scale),
            false_safe_rows=int(((prediction[observed,col]-.02*prediction[observed,reference] <= 0) &
                                (y[observed,col]-.02*y[observed,reference] > 1e-12)).sum()))
    result['benefit_harm_envelope_exceedances'] = int(((prediction[supported,0]+prediction[supported,1]) >
                                                      envelope[supported]+1e-7).sum())
    return result


def summarize(rows, expected):
    keys = [(r['context'],r['source'],r['seed']) for r in rows]
    if len(keys) != 72 or len(set(keys)) != 72 or set(keys) != set(expected):
        raise ValueError('All72 registered views required')
    arms = ('origin','estimation')+MODES
    means = {a: {target: {
        metric: float(np.mean([r['scores'][a][target][metric] for r in rows]))
            if all(r['scores'][a][target][metric] is not None for r in rows) else None
        for metric in ('full_cohort_MSE','AUROC','AP')} for target in ('all','easy')} for a in arms}
    pairs = sorted({(r['context'],r['seed']) for r in rows}); wins = 0
    for context, seed in pairs:
        group = [r for r in rows if (r['context'],r['seed']) == (context,seed)]
        values = {a: [r['scores'][a]['easy']['full_cohort_MSE'] for r in group] for a in MODES}
        if all(all(v is not None for v in vv) for vv in values.values()):
            wins += np.mean(values['nearest32']) < min(np.mean(values['source_mean']),
                                                    np.mean(values['shuffled_nearest32']))-1e-12
    conditions = {f'easy_MSE_vs_{other}': means['nearest32']['easy']['full_cohort_MSE'] is not None and
        means[other]['easy']['full_cohort_MSE'] is not None and
        means['nearest32']['easy']['full_cohort_MSE'] < means[other]['easy']['full_cohort_MSE']-1e-12
        for other in ('source_mean','shuffled_nearest32')}
    conditions['at_least12_of18_pair_wins'] = bool(wins >= 12)
    return dict(views=72, equal_view_metrics=means, pair_wins=int(wins), information_screen=conditions,
        conditional_signal_screen_pass=all(conditions.values()),
        occurrences={k:sum(r['scores']['nearest32'][k] for r in rows) for k in
            ('rows','known','unknown','supported','unsupported','supported_known','supported_unknown')},
        incomplete_MSE_views=sum(r['scores']['nearest32']['easy']['full_cohort_MSE'] is None for r in rows),
        same_recording_donors=sum(v['same_recording_donors'] for r in rows for v in r['references']),
        fitting_diagnostic_only=True, whole_pipeline_OOF=False, held_inference=False,
        independent_roles_read=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
