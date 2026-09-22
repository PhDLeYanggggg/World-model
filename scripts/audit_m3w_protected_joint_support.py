"""Checkpointed causal-only joint opportunity audit; never read future outcomes."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.build_m3w_native_scene_context import load_past_queries
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_scene_alignment import restore
from src.evaluation.m3w_protected_joint_support import audit_problem
from src.world_model.m3w_native_joint_controls import make_problem
from src.world_model.m3w_temporal_intervention import candidates

CONFIG = 'configs/m3w_protected_joint_support_v1.json'
CODE = ('scripts/audit_m3w_protected_joint_support.py',
        'src/evaluation/m3w_protected_joint_support.py',
        'tests/test_m3w_protected_joint_support.py',
        'src/world_model/m3w_native_joint_controls.py',
        'src/world_model/m3w_interaction_controls.py')
CONTEXT_KEYS = ('context_target_rows', 'context_frame_ids', 'context_agent_ids',
                'context_cv_rollout', 'context_cv_valid', 'context_xy')
SAFE_KEYS = set(CONTEXT_KEYS) | {'ids', 'prediction', 'origin', 'rotation',
    'stored_metric_scale', 'square_score', 'forest_ratio', 'log_strict',
    'square_strict', 'ramp_forest_score', 'ramp_neural_score'}


def causal_arrays(path, keys):
    if not set(keys) <= SAFE_KEYS:
        raise ValueError('Only enumerated causal arrays may enter this audit')
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    base = ROOT/'outputs/publication_readiness_2026_09'
    parent_path = base/'fraction_square_v1/analysis.json'
    parent = json.loads(parent_path.read_text())
    if (file_digest(parent_path) != cfg['parent_analysis_sha256']
            or cfg['pools'] != ['forest_ratio', 'log_strict', 'square_strict']
            or cfg['budget_fraction'] != .5 or cfg['enumeration_cap_per_query'] != 200000
            or cfg['geometry'] != dict(past_radius='median_target_past_scale', threshold_fraction=.1, pair_weight=1.)
            or any(cfg[k] for k in ('new_training', 'new_policy_selection', 'future_outcome_readout',
                'threshold_search', 'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed diagnostic scope changed')
    bindings = dict(parent['identity']['source_bindings'])

    def bind(path, expected=None):
        path = Path(path)
        key = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        sha = file_digest(ROOT/key)
        if (expected is not None and sha != expected) or (key in bindings and bindings[key] != sha):
            raise ValueError('Changed input: '+key)
        bindings[key] = sha
        return sha

    for name in ('replay.json', 'separate_verification.json'):
        p = parent_path.with_name(name)
        r = json.loads(p.read_text())
        if not r['all_checks_passed'] or r['analysis_sha256'] != file_digest(parent_path):
            raise ValueError('Verified parent result required')
        bind(p)
    for p in (CONFIG, *CODE, str((base/'protected_joint_support_v1/registration.md').relative_to(ROOT)),
              str(parent_path.relative_to(ROOT))):
        bind(p)
    paths = {k: base/v for k, v in dict(context='native_scene_context_v2/analysis.json',
        forest='forest_cost_v1/analysis.json', nested='native_nested_v1/materialized_views.json',
        alignment='native_scene_alignment_v1/analysis.json').items()}
    reports = {}
    for k, p in paths.items():
        bind(p)
        reports[k] = json.loads(p.read_text())
    context = reports['context']
    for p, sha in context['source_bindings'].items():
        bind(p, sha)
    vp = paths['context'].with_name('verification.json')
    verification = json.loads(vp.read_text())
    assert verification['all_checks_passed'] and verification['analysis_sha256'] == file_digest(paths['context'])
    bind(vp)
    assert reports['nested']['all_checks_passed'] and reports['nested']['outer_rows_in_training'] == 0
    data = load_past_queries(bindings)
    transform = reports['alignment']['cache']
    bind(transform['path'], transform['sha256'])
    data.update(causal_arrays(ROOT/transform['path'], ('origin', 'rotation', 'stored_metric_scale')))
    data['scale'] = data.pop('stored_metric_scale')
    data['sites'] = np.array([r.split('/')[0] for r in data['recordings']])
    views = {}
    for r in reports['nested']['views']:
        bind(r['receipt_path'], r['receipt_sha256'])
        meta = json.loads((ROOT/r['receipt_path']).read_text())
        key = f"{r['outer_site']}_seed{r['seed']}"
        pred = meta['outer_prediction']['prediction']
        bind(pred['path'], pred['sha256'])
        sq = next(a for a in parent['archives'] if a['view'] == key)
        fr = next(a for a in reports['forest']['archives'] if a['view'] == key)
        tr = next(a for a in parent['training'] if a['view'] == key)
        for a in (sq, fr): bind(a['path'], a['sha256'])
        bind(tr['checkpoint'], tr['checkpoint_sha256'])
        views[key] = dict(meta=meta, prediction=pred, square=sq, forest=fr, training=tr)
    for r in context['records']: bind(r['cache']['path'], r['cache']['sha256'])
    assert len(data['sites']) == 175756 and len(context['records']) == 33 and len(views) == 12
    assert set(data['sites']) == set(cfg['sites']) and set(m['meta']['seed'] for m in views.values()) == set(cfg['seeds'])
    identity = dict(config=cfg, source_bindings=bindings, numpy=np.__version__,
        architecture=platform.machine(), future_target_arrays_loaded=False,
        all_four_sites_development_exposed=True)
    assert_current(identity)
    return cfg, data, context, views, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.verify and args.audit_only: raise ValueError('Choose one mode')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text())
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    root.mkdir(parents=True, exist_ok=True)

    def beat(**row):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **row)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    with (root/'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, data, context, views, identity = load()
        immutable_json(root/'identity.json', identity)
        identity_sha = file_digest(root/'identity.json')
        if args.audit_only:
            beat(state='preflight_pass', source_bindings=len(identity['source_bindings']))
            return
        records = []
        for key, view in views.items():
            a = causal_arrays(ROOT/view['prediction']['path'], ('ids', 'prediction'))
            ids = a['ids']
            np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == view['meta']['outer_site']))
            sq = causal_arrays(ROOT/view['square']['path'], ('ids', 'square_score', *cfg['pools']))
            fr = causal_arrays(ROOT/view['forest']['path'], ('ids', 'ramp_forest_score', 'ramp_neural_score'))
            np.testing.assert_array_equal(ids, sq['ids'])
            np.testing.assert_array_equal(ids, fr['ids'])
            cp = torch.load(ROOT/view['training']['checkpoint'], map_location='cpu', weights_only=False)
            pr = cp['preprocess']
            assert set(pr['training_sites']) == set(cfg['sites'])-{view['meta']['outer_site']}
            score = dict(forest_ratio=fr['ramp_forest_score'], log_strict=fr['ramp_neural_score'], square_strict=sq['square_score'])
            local_baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
            local_candidate = candidates(local_baseline, a['prediction'])[0]['ramp']
            b = restore(local_baseline, data['origin'][ids], data['rotation'][ids], data['scale'][ids])
            q = restore(local_candidate, data['origin'][ids], data['rotation'][ids], data['scale'][ids])
            lookup = np.full(len(data['sites']), -1, int)
            lookup[ids] = np.arange(len(ids))
            for record in context['records']:
                rec = record['recording']
                if rec.split('/')[0] != view['meta']['outer_site']: continue
                path = root/'queries'/key/(rec.replace('/', '_')+'.json')
                if path.exists() and not args.verify:
                    r = json.loads(path.read_text())
                    if r['identity_sha256'] != identity_sha: raise ValueError('Changed cached audit identity')
                    records.append(r)
                    beat(state='cached_verified', view=key, recording=rec)
                    continue
                if args.verify and not path.exists(): raise ValueError('Replay cannot create audit results')
                c = causal_arrays(ROOT/record['cache']['path'], CONTEXT_KEYS)
                order = np.argsort(c['context_frame_ids'], kind='stable')
                _, starts = np.unique(c['context_frame_ids'][order], return_index=True)
                groups = np.split(order, starts[1:])
                out = []
                for rows in groups:
                    target = c['context_target_rows'][rows] >= 0
                    tid = c['context_target_rows'][rows][target]
                    loc = lookup[tid]
                    if not len(tid) or (loc < 0).any(): raise ValueError('Context target/view alignment failed')
                    n = len(rows)
                    base = c['context_cv_rollout'][rows].astype(float)
                    base[target] = b[loc]
                    cand = base.copy()
                    cand[target] = q[loc]
                    valid = c['context_cv_valid'][rows].all(1)
                    valid[target] = True
                    p, geo = make_problem(baseline=base, candidate=cand, current=c['context_xy'][rows],
                        forecast_valid=valid, target_mask=target, eligible=np.zeros(n, bool),
                        benefit=np.zeros(n), harm=np.zeros(n), scale=pr['cost_scale'], past_target_scales=data['scale'][tid])
                    entry = dict(frame=int(c['context_frame_ids'][rows[0]]), geometry=geo, pools={})
                    for name in cfg['pools']:
                        supported = np.zeros(n, bool)
                        supported[target] = sq[name][loc]
                        gain, harm = np.zeros(n), np.zeros(n)
                        costs = score[name][loc].astype(float)
                        gain[target] = (costs[:, 0]-costs[:, 1])/pr['cost_scale']
                        harm[target] = costs[:, 1]/pr['cost_scale']
                        problem = replace(p, supported=supported, expected_gain=gain, expected_harm=harm,
                            max_interventions=int(supported.sum()))
                        entry['pools'][name] = audit_problem(problem, c['context_agent_ids'][rows],
                            enumeration_cap=cfg['enumeration_cap_per_query'])
                    out.append(entry)
                r = dict(identity_sha256=identity_sha, view=key, recording=rec, queries=out)
                immutable_json(path, r)
                records.append(r)
                beat(state='replayed_recording' if args.verify else 'recording_complete', view=key,
                    recording=rec, queries=len(out), opportunities=sum(v['status']=='exhaustively_checked'
                        for e in out for v in e['pools'].values()))
        rows = [(r, q) for r in records for q in r['queries']]
        assert len(rows) == 20932*3
        summaries = {}
        for name in cfg['pools']:
            subsets = {'all': rows}
            subsets.update({s: [(r, q) for r, q in rows if r['recording'].split('/')[0] == s] for s in cfg['sites']})
            summaries[name] = {}
            for site, subset in subsets.items():
                values = [q['pools'][name] for _, q in subset]
                exact = [v for v in values if v['exact_enumeration']]
                opportunities = [(r, q) for r, q in subset if q['pools'][name]['status']!='structural_null']
                summaries[name][site] = dict(queries=len(values), eligible_agent_instances=sum(v['pool'] for v in values),
                    positive_half_count_queries=sum(v['count'] > 0 for v in values),
                    count_at_least_two_queries=sum(v['count'] >= 2 for v in values),
                    nonadditive_opportunities=len(opportunities),
                    unique_recording_frame_opportunities=len({(r['recording'], q['frame']) for r, q in opportunities}),
                    recordings_with_opportunity=sorted({r['recording'] for r, q in opportunities}),
                    enumeration_blockers=sum(v['status']=='enumeration_cap_blocker' for v in values),
                    enumerated_queries=len(exact), combinations_enumerated=sum(v['combinations'] for v in exact),
                    feasible_combinations=sum(v['feasible'] for v in exact),
                    unique_feasible_queries=sum(v['feasible']==1 for v in exact),
                    variable_product_queries=sum(v['product_range']>1e-12 for v in exact),
                    product_range_below_unary_gap=sum(v['product_range_below_unary_gap'] is True for v in exact),
                    changed_identity_queries=sum(v['changed_identities']>0 for v in exact),
                    meaningful_objective_advantage_queries=sum(v['meaningful_advantage'] for v in exact),
                    objective_advantage_sum=sum(v['objective_advantage'] for v in exact))
        result = dict(identity=identity, result_source='fresh_causal_support_audit_cached_verified_scores_context',
            summaries=summaries, query_receipts=[dict(path=str((root/'queries'/r['view']/(r['recording'].replace('/', '_')+'.json')).relative_to(ROOT)),
                sha256=file_digest(root/'queries'/r['view']/(r['recording'].replace('/', '_')+'.json'))) for r in records],
            queries=len(rows), future_target_arrays_loaded=False, future_mask_arrays_loaded=False,
            outcomes_evaluated=False, new_training=False, new_policy_selection=False, deployment=False,
            independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
        assert_current(identity)
        immutable_json(public/'analysis.json', result)
        if args.verify:
            immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
                queries_replayed=len(rows), pool_queries_replayed=len(rows)*len(cfg['pools']), all_checks_passed=True))
        beat(state='verified' if args.verify else 'complete', summaries={k:v['all'] for k,v in summaries.items()})


if __name__ == '__main__':
    main()
