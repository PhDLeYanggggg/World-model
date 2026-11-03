"""Disjoint proposal/critic/evaluation roles for the admitted moving-source bank."""
import numpy as np

from src.data_unification.m3w_moving_forecast_data import array_hash
from src.world_model import m3w_shared_motion_cost as api
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_frame_action_allocation import query_groups


def aligned_rows(cached_ids, wanted):
    ids, wanted = np.asarray(cached_ids), np.asarray(wanted)
    if (ids.ndim != 1 or wanted.ndim != 1 or ids.dtype.kind not in 'iu'
            or wanted.dtype.kind not in 'iu' or not len(ids) or not len(wanted)
            or np.any(np.diff(ids) <= 0) or np.any(np.diff(wanted) <= 0)):
        raise ValueError('Sorted unique nonempty source row IDs required')
    at = np.searchsorted(ids, wanted)
    if np.any(at >= len(ids)) or not np.array_equal(ids[at], wanted):
        raise ValueError('Every requested row must be in the excluded-source prediction bank')
    return at


def job_designs(data, folds, producers, settings):
    present = set(data['sites'])
    if set(folds.values()) != {0, 1, 2} or not present <= set(folds):
        raise ValueError('Complete three-fold source mapping required')
    scopes = {}
    for fold in range(3):
        names = sorted(g for g in present if folds[g] == fold)
        ids = np.flatnonzero(np.isin(data['sites'], names))
        if not names or not len(ids):
            raise ValueError('Every fitting/evaluation fold needs actual source rows')
        queries = query_groups(data['recordings'][ids], data['frames'][ids], ids)
        qnames = np.asarray([data['sites'][ids[ix[0]]] for ix in queries])
        source_groups = [np.flatnonzero(qnames == s) for s in names]
        if settings['query_batch_size'] % len(names):
            raise ValueError('Equal source sampling must fit every actual fold cardinality')
        budget = api.base.base.steps_for(dict(source_groups=source_groups), settings)
        scopes[fold] = dict(sites=names, rows=len(ids), ids_sha256=array_hash(ids),
            queries=len(queries), source_query_counts=list(map(len, source_groups)), budget=budget)
    keys = {(p['fold'], p['seed']) for p in producers}
    if len(producers) != 9 or keys != {(f, s) for f in range(3) for s in (17, 29, 43)}:
        raise ValueError('Exactly nine frozen three-seed EqMotion producers required')
    jobs = []
    for producer in producers:
        fit_fold = producer['fold']
        declared = sorted(g for g in folds if folds[g] == fit_fold)
        if producer['family'] != 'eqmotion' or sorted(producer['fit_groups']) != declared:
            raise ValueError('Do not choose a forecaster family or change producer fitting exposure')
        for critic_fold in sorted({0, 1, 2}-{fit_fold}):
            held_fold = next(iter({0, 1, 2}-{fit_fold, critic_fold}))
            jobs.append(dict(key=f"eqmotion_fold{fit_fold}_seed{producer['seed']}_critic{critic_fold}",
                forecast_key=f"eqmotion_fold{fit_fold}_seed{producer['seed']}", seed=producer['seed'],
                producer_fold=fit_fold, critic_fold=critic_fold, evaluation_fold=held_fold,
                producer_sites=declared, critic_sites=scopes[critic_fold]['sites'],
                evaluation_sites=scopes[held_fold]['sites'], floor_index=producer['baseline_index'],
                easy_cut=producer['easy_cut'], hard_cut=producer['hard_cut'],
                fit_scope=scopes[critic_fold], evaluation_scope=scopes[held_fold]))
    return jobs


def bank(data, job, cached_ids, prediction, *, training=False):
    names = job['critic_sites'] if training else job['evaluation_sites']
    if (set(names) & set(job['producer_sites']) or set(job['critic_sites']) & set(job['evaluation_sites'])):
        raise ValueError('Predictor, critic and held source groups must be disjoint')
    ids = np.flatnonzero(np.isin(data['sites'], names))
    scope = job['fit_scope'] if training else job['evaluation_scope']
    if len(ids) != scope['rows'] or array_hash(ids) != scope['ids_sha256']:
        raise ValueError('Fixed role-specific row population required')
    at = aligned_rows(cached_ids, ids)
    pred = np.asarray(prediction)
    if pred.shape != (len(cached_ids), 12, 2) or not np.isfinite(pred).all():
        raise ValueError('Finite, complete saved proposal population required')
    origin = data['origin'][ids]
    physical = np.stack([baseline_numpy(data['history'][ids], k)-origin[:, None] for k in range(6)], 1)
    paths = np.concatenate((physical, pred[at, None].astype(np.float64)), 1)
    causal = api.features(geometry=data['geometry'][ids], shared_context=data['shared_context'][ids],
        rollouts=paths, floor_index=job['floor_index'], recordings=data['recordings'][ids],
        frames=data['frames'][ids], ids=ids, sites=data['sites'][ids])
    # Fix the proposal across independent, uniform and joint controls. Causal floors
    # are not extra switch candidates in this neural-intervention experiment.
    causal['allowed'][:, :6] = False
    out = dict(ids=ids, paths=paths, causal=causal, recordings=data['recordings'][ids],
        frames=data['frames'][ids], sites=data['sites'][ids])
    if training:
        out['learning'] = api.base.training_data(causal, paths,
            data['target_eval'][ids]-origin[:, None], data['valid'][ids], job['floor_index'], job['easy_cut'])
        if set(out['learning']['preprocess']['source_names']) != set(job['critic_sites']):
            raise ValueError('Preprocessing must be fitted on critic sources only')
    return out
