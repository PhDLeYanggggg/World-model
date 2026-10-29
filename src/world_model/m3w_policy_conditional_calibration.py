"""Frozen-score, source-excluded excess-risk calibration; no coverage guarantee."""
import numpy as np

from src.evaluation.m3w_selected_tail_audit import actions, BUDGET
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds

ARMS = ('raw', 'global', 'support_only', 'conditional')
GAIN_EDGES = (0., .02, .1)
MIN_RECORDINGS = 3
MIN_SOURCES = 2
MIN_KNOWN = 32


def inputs(prediction, envelope, eligible):
    p, e, ok = np.asarray(prediction, float), np.asarray(envelope, float), np.asarray(eligible, bool)
    if (p.shape != (len(e), 5) or ok.shape != e.shape or e.ndim != 1
            or not np.isfinite(p).all() or not np.isfinite(e).all()
            or (p < 0).any() or (e < 0).any()):
        raise ValueError('Aligned finite nonnegative causal inputs required')
    return p, e, ok


def cells(prediction, envelope):
    p, e, _ = inputs(prediction, envelope, np.ones(len(envelope), bool))
    gain = np.divide(p[:, 0]-p[:, 1], e, out=np.zeros(len(e)), where=e > 0)
    return (4*np.searchsorted(GAIN_EDGES, gain, side='right')
            + 2*(p[:, 1] <= BUDGET*p[:, 2]) + (p[:, 4] <= BUDGET*p[:, 3]))


def excess(prediction):
    p = np.asarray(prediction, float)
    return np.column_stack((p[:, 1]-BUDGET*p[:, 2], p[:, 4]-BUDGET*p[:, 3]))


def fit(calibration, target_source):
    """Only the other sources' head-excluded TRAIN recordings supply labels."""
    if target_source in calibration or len(calibration) != 3:
        raise ValueError('Exactly three calibration sources; target must be excluded')
    totals = {str(i): [] for i in range(16)}
    totals['global'] = []
    for source, row in sorted(calibration.items()):
        p, env, eligible = inputs(row['prediction'], row['envelope'], row['eligible'])
        y, rec = np.asarray(row['targets'], float), np.asarray(row['recordings'])
        if y.shape != p.shape or rec.shape != env.shape:
            raise ValueError('Aligned calibration labels/recordings required')
        known = np.isfinite(y).all(1)
        if np.any(np.isfinite(y).any(1) != known) or (y[known] < 0).any():
            raise ValueError('Labels must be all-known or all-unknown, never imputed')
        completion_bounds(y, np.zeros(len(y), bool), env)
        ids, residual = cells(p, env), excess(y)-excess(p)
        for recording in np.unique(rec):
            for key in totals:
                at = eligible & (env > 0) & (rec == recording)
                if key != 'global':
                    at &= ids == int(key)
                n, unknown = int((at & known).sum()), int((at & ~known).sum())
                mass = float(env[at & known].sum())
                if n+unknown:
                    totals[key].append(dict(source=source, recording=str(recording), known=n,
                        unknown=unknown, residual=(residual[at & known].sum(0)/mass).tolist() if mass > 0 else None))
    table = {}
    for key, rows in totals.items():
        by_source = {}
        for r in rows:
            if r['residual'] is not None:
                by_source.setdefault(r['source'], []).append(r['residual'])
        n = sum(r['known'] for r in rows)
        informative = sum(r['residual'] is not None for r in rows)
        supported = informative >= MIN_RECORDINGS and len(by_source) >= MIN_SOURCES and n >= MIN_KNOWN
        estimate = (np.mean([np.mean(v, axis=0) for v in by_source.values()], axis=0)
                    if by_source else np.zeros(2))
        table[key] = dict(offset=np.maximum(estimate, 0).tolist(), signed_residual=estimate.tolist(),
            supported=bool(supported), recordings=informative, sources=len(by_source), known=n,
            unknown=sum(r['unknown'] for r in rows))
    return dict(target_source=target_source, calibration_sources=sorted(calibration), table=table,
        budget=BUDGET, gain_edges=list(GAIN_EDGES), prediction_target='risk_excess_not_FDE',
        no_population_guarantee=True, labels_used_only_in_calibration=True)


def decide(prediction, envelope, eligible, calibrator):
    p, env, ok = inputs(prediction, envelope, eligible)
    if calibrator['budget'] != BUDGET or calibrator['gain_edges'] != list(GAIN_EDGES):
        raise ValueError('Frozen budget and causal cell boundaries required')
    raw = actions(p, ok)
    ids = cells(p, env)
    table = calibrator['table']
    conditional_supported = np.array([table[str(i)]['supported'] for i in ids])
    offsets = np.array([table[str(i)]['offset'] for i in ids])
    global_offset = np.asarray(table['global']['offset'])
    base_excess = excess(p)
    # Penalties are a decision overlay, not a replacement coherent moment decoder.
    global_ok = (base_excess+env[:, None]*global_offset <= 0).all(1)
    conditional_ok = (base_excess+env[:, None]*offsets <= 0).all(1)
    policies = dict(raw=raw, global_=raw & bool(table['global']['supported']) & global_ok,
        support_only=raw & conditional_supported,
        conditional=raw & conditional_supported & conditional_ok)
    policies['global'] = policies.pop('global_')
    if any((v & ~raw).any() for v in policies.values()):
        raise AssertionError('Calibration must not expand the original decision set')
    return policies


def lineage(contexts):
    """Fail closed on producer/controller exposure or changed recording partitions."""
    out = []
    if len(contexts) != 6:
        raise ValueError('Complete six-context design required')
    for context, items in sorted(contexts.items()):
        if len(items) != 4 or len({r['packet']['identity']['source'] for r in items}) != 4:
            raise ValueError('Four unique calibration/readout sources per context required')
        for item in items:
            ref, partition = item['packet'], item['partition']
            identity = ref['identity']; source = identity['source']; upstream = identity['upstream']
            if (upstream['context'] != context or source in upstream['producer_sites']
                    or source in upstream['controller_sites']
                    or set(upstream['producer_sites']) != set(upstream['upstream_context']['fit_sites'])):
                raise ValueError('Whole producer/controller exclusion required')
            fit_set, held_set = map(set, (partition['fit_recordings'], partition['held_recordings']))
            outer = identity['partition']
            if (not fit_set or not held_set or fit_set & held_set
                    or fit_set | held_set != set(outer['train_recordings'])
                    or (fit_set | held_set) & set(outer['validation_recordings'])):
                raise ValueError('Exhaustive disjoint original-TRAIN recording roles required')
            out.append(dict(context=context, source=source, producer_excluded=True,
                controller_excluded=True, head_fit_recordings=len(fit_set),
                calibration_readout_recordings=len(held_set), original_TRAIN_only=True))
    return out
