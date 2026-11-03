"""Frozen secondary relational evaluation of fully replayed source forecasts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before scientific runtime imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.intake_m3w_movingdronecrowd import sha, once
from src.evaluation.m3w_moving_relational import past_edges, score

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_relational_diagnostic_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_relational_diagnostic_v1'
PARENT = ROOT/'outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1'
CONFIG = ROOT/'configs/m3w_moving_relational_v1.json'
FILES = (Path(__file__), CONFIG, PUBLIC/'protocol.md',
    ROOT/'src/evaluation/m3w_moving_relational.py', ROOT/'tests/test_m3w_moving_relational.py')
REG = PUBLIC/'registration.json'


def read(path):
    return json.loads(path.read_text())


def register():
    cfg = read(CONFIG)
    if (cfg['past_nearest_neighbors'] != 8 or cfg['models'] != ['cv', 'floor', 'pct', 'eqmotion']
            or any(cfg[k] for k in ('primary_or_safety_gate_changes', 'new_training',
                                  'new_source_admission', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Keep the fixed secondary diagnostic, never a new primary or gate')
    training = read(PARENT/'training.json')
    if training['complete_primary_models'] != 18 or not training['matched_sampling_all9pairs_exact']:
        raise ValueError('Use all original18trained predictors')
    parent = {p:sha(PARENT/p) for p in ('registration.json', 'population.json', 'training.json', 'job_roster.json')}
    value = dict(configuration=cfg, bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent=parent, registered_before_parent_readout=True, new_training=False,
        original_primary_or_risk_changed=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    if not REG.exists() and (PARENT/'analysis.json').exists():
        raise ValueError('Freeze this diagnostic before reading the complete parent result')
    once(REG, value)
    return cfg


def interval(rows, getter, sites, cfg):
    values = {s:[] for s in sites}
    for row in rows:
        values[row['site']].append(getter(row))
    if any(len(v) != 6 for v in values.values()):
        raise ValueError('Each supported group needs two excluded producers and three seeds')
    undefined = sorted(s for s,v in values.items() if any(x is None for x in v))
    if undefined:
        return dict(point=None, CI95=None, status='required_group_without_metric_support', undefined_groups=undefined)
    means = {s:float(np.mean(v)) for s,v in values.items()}
    x = np.asarray(list(means.values()))
    draw = np.random.default_rng(cfg['bootstrap_seed']).choice(x, size=(cfg['bootstrap_resamples'], len(x))).mean(1)
    return dict(point=float(x.mean()), CI95=np.quantile(draw, [.025, .975]).tolist(),
        by_source_group=means, bootstrap_resamples=cfg['bootstrap_resamples'],
        unit='conservative_source_group_after_producer_seed_mean_not_verified_IID_sites',
        status='secondary_development_diagnostic_not_independent_confirmation')


def cached(ref, data, identity):
    path = ROOT/ref['path']
    if sha(path) != ref['sha256'] or ref['identity'] != identity:
        raise ValueError('Changed cached forecast identity')
    with np.load(path, allow_pickle=False) as source:
        ids, prediction = source['ids'].copy(), source['relative_prediction'].copy()
    expected = np.flatnonzero(~np.isin(data['sites'], identity['fit_groups']))
    if (not np.array_equal(ids, expected) or prediction.shape != (len(ids), 12, 2)
            or prediction.dtype != np.float32 or not np.isfinite(prediction).all()):
        raise ValueError('Complete aligned finite producer-excluded forecasts required')
    return ids, prediction


def compute(cfg):
    from src.data_unification.m3w_moving_forecast_data import prepare
    from src.world_model.m3w_european_source_forecast import baseline_numpy
    parent = read(PARENT/'analysis.json')
    if not read(PARENT/'verify_execution.json')['real_Torch_training']:
        raise ValueError('Finish both complete refits and all saved-prediction replays first')
    source_cfg = read(ROOT/'configs/m3w_moving_source_forecast_supported_v1.json')
    data, population, _ = prepare(ROOT, source_cfg['admission'])
    if population != read(PARENT/'population.json'):
        raise ValueError('Do not alter the original all-row population')
    jobs = {r['key']:r['identity'] for r in read(PARENT/'job_roster.json')['jobs']}
    refs = {f"{r['identity']['family']}_fold{r['identity']['fold']}_seed{r['identity']['seed']}":r
            for r in parent['prediction_receipts']}
    graphs, rows = {}, []
    for site in sorted(set(data['sites'])):
        ids = np.flatnonzero(data['sites'] == site)
        _, starts = np.unique(data['queries'][ids], return_index=True)
        queries = np.split(np.arange(len(ids)), starts[1:])
        edges = [q[past_edges(data['origin'][ids[q]], data['agents'][ids[q]], cfg['past_nearest_neighbors'])]
                 for q in queries if len(q) > 1]
        graphs[str(site)] = dict(ids=ids, edges=np.concatenate(edges) if edges else np.empty((0, 2), np.int64),
            queries=len(queries), queries_with_edges=sum(len(q)>1 for q in queries))
    for fold in range(3):
        for seed in (17, 29, 43):
            key = f'fold{fold}_seed{seed}'
            identity = jobs['pct_'+key]
            ids, pct = cached(refs['pct_'+key], data, identity)
            eids, eq = cached(refs['eqmotion_'+key], data, jobs['eqmotion_'+key])
            np.testing.assert_array_equal(ids, eids)
            for site in sorted(set(data['sites'][ids])):
                graph = graphs[str(site)]
                local = graph['ids']
                at = np.searchsorted(ids, local)
                np.testing.assert_array_equal(ids[at], local)
                origin = data['origin'][local]
                predictions = dict(cv=baseline_numpy(data['history'][local], 1)-origin[:, None],
                    floor=baseline_numpy(data['history'][local], identity['baseline_index'])-origin[:, None],
                    pct=pct[at], eqmotion=eq[at])
                result = score(graph['edges'], predictions,
                    data['target_eval'][local]-origin[:, None], data['valid'][local])
                rows.append(dict(site=str(site), producer_fold=fold, seed=seed,
                    queries=graph['queries'], queries_with_edges=graph['queries_with_edges'], **result))
            print(json.dumps(dict(state='secondary_group_scores', fold=fold, seed=seed, groups=len(rows))), flush=True)
    sites = sorted(graphs)
    contrasts = {metric:interval(rows, lambda r,m=metric:r[m], sites, cfg)
        for metric in ('paired_known_pp', 'paired_completion_lower_pp', 'paired_completion_upper_pp')}
    models = {key:{metric:interval(rows, lambda r,k=key,m=metric:r['metrics'][k][m], sites, cfg)
        for metric in ('available_pair_ADE_pixel', 'available_endpoint_pair_FDE_pixel')}
        for key in ('cv', 'floor', 'pct', 'eqmotion')}
    return dict(result_source='fresh_secondary_metric_on_hash_schema_and_parent_replay_verified_cached_predictions',
        registration_sha256=sha(REG), parent_analysis_sha256=sha(PARENT/'analysis.json'),
        parent_replay_sha256=sha(PARENT/'verify_execution.json'), groups=rows,
        unique_past_defined_edges=sum(len(v['edges']) for v in graphs.values()),
        unique_queries=sum(v['queries'] for v in graphs.values()), supported_groups=len(graphs),
        contrasts=contrasts, models=models, original24group_primary='not_estimable_not_passed',
        absolute_error_or_risk_gates_replaced=False, common_translation_blind=True,
        physical_safety_or_metric_claim=False, new_training=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'evaluate', 'verify'))
    args = parser.parse_args()
    cfg = register()
    if args.phase == 'register':
        print(json.dumps(dict(registration_sha256=sha(REG), before_parent_readout=True)))
        return
    for path in (*FILES, REG):
        blob = subprocess.run(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
            cwd=ROOT, check=True, capture_output=True).stdout
        if hashlib.sha256(blob).hexdigest() != sha(path):
            raise ValueError('Commit the exact diagnostic definition before evaluation')
    if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
        raise OSError('Preserve original10GiB reserve')
    start = time.monotonic()
    value = compute(cfg)
    once(PUBLIC/'analysis.json', value)
    once(PUBLIC/(args.phase+'_execution.json'), dict(pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_training=False, new_source_or_primary_or_risk_changed=False,
        all_saved_metric_scalars_exact_on_verify=args.phase == 'verify'))
    print(json.dumps(dict(phase=args.phase, complete=True, new_training=False)), flush=True)


if __name__ == '__main__':
    main()
