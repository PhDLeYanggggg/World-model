"""Exhaustive control checks and original-source past-input probes; no training."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scripts.audit_m3w_sdd_state_support import load_source
from src.data_unification.m3w_causal_recordings import causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_interaction_control_evaluation import attach_interaction_controls, make_control_problem
from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_interaction_controls import compare_interaction_controls, decompose_pair_objective
from src.world_model.m3w_sdd_auxiliary import load_registration, source_entries
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter
import src.world_model.m3w_joint_intervention as legacy

CONFIG = 'configs/m3w_interaction_controls_v1.json'
DEPENDENCIES = [CONFIG, 'scripts/check_m3w_interaction_controls.py',
    'src/world_model/m3w_interaction_controls.py', 'src/evaluation/m3w_interaction_control_evaluation.py',
    'src/world_model/m3w_joint_intervention.py', 'src/evaluation/m3w_development_evaluation.py',
    'src/data_unification/m3w_causal_recordings.py', 'src/world_model/m3w_sdd_step_adapter.py',
    'tests/test_m3w_interaction_controls.py']


def synthetic_checks(cfg):
    rng = np.random.default_rng(cfg['synthetic_seed'])
    results = []
    enumeration = 0
    maximum_error = 0.
    for index in range(cfg['synthetic_problems']):
        n = 6+index%3
        edges = np.array([(i,j) for i in range(n) for j in range(i+1,n)])
        table = rng.random((len(edges),2,2))*.5; table[:,0,0] = 0
        kind = ['additive','product_only','synergy','general'][index%4]
        if kind == 'additive':
            table[:,1,1] = table[:,1,0]+table[:,0,1]
        elif kind == 'product_only':
            table[:,1,0] = table[:,0,1] = 0
        elif kind == 'synergy':
            table[:,1,1] = 0
        p = InterventionProblem(rng.normal(.15,.25,n), rng.random(n)*.2, rng.random(n) > .15,
                               edges, table, .7, .08, n//2)
        r = compare_interaction_controls(p, time_limit_seconds=cfg['solver_seconds'])
        if not r['matched']:
            raise RuntimeError('Synthetic exact solve did not complete')
        d = decompose_pair_objective(p)
        best_u, best_j = np.inf, np.inf
        for bits in product((0,1), repeat=n):
            x = np.array(bits)
            pair = table[np.arange(len(edges)),x[edges[:,0]],x[edges[:,1]]].mean()
            objective = -(x*p.expected_gain).mean()+p.pair_weight*pair
            expanded = d['unary_objective']@x+d['product_objective']@(x[edges[:,0]]*x[edges[:,1]])
            maximum_error = max(maximum_error, abs(objective-expanded))
            enumeration += 1
            if (x.sum() == r['reference_count'] and not np.any(x > p.supported)
                    and np.mean(x*np.maximum(p.expected_harm,-p.expected_gain)) <= p.max_mean_predicted_harm+1e-12):
                best_u = min(best_u,float(d['unary_objective']@x))
                best_j = min(best_j,float(objective))
        np.testing.assert_allclose(r['controls']['unary_geometry']['unary_objective'],best_u,atol=1e-8,rtol=0)
        np.testing.assert_allclose(r['controls']['joint']['full_objective'],best_j,atol=1e-8,rtol=0)
        results.append(dict(index=index,kind=kind,agents=n,reference_count=r['reference_count'],
            risk_only_full_objective=r['controls']['independent']['full_objective'],
            unary_full_objective=r['controls']['unary_geometry']['full_objective'],
            joint_full_objective=r['controls']['joint']['full_objective'],
            joint_advantage=r['predicted_full_objective_advantage'],
            product_sum_bound=r['absolute_product_sum_bound']))
    return dict(problems=len(results),enumerated_assignments=enumeration,max_decomposition_error=maximum_error,
        exact_optima_verified=len(results)*2,all_cases=results,
        constructed_scores_not_fitted_models=True,real_accuracy_evaluated=False)


def inputs_at(adapter, frame, cfg):
    current = adapter.points[adapter.index['current_row']]
    rows = np.flatnonzero(current[:,0] == frame)
    agents = []
    b, c, gains = [], [], []
    h = hashlib.sha256()
    for item in rows:
        inp = adapter.get_inputs(int(item))
        if (np.any(inp['history_frame_offsets'] > 0)
                or np.any(inp['neighbor_frame_offsets'][inp['neighbor_mask']] > 0)):
            raise ValueError('Future observed token')
        row = adapter.index[item]
        history = adapter.points[int(row['history_start']):int(row['current_row'])+1]
        transform = causal_coordinate_transform(history[:,2:],history[:,0],144)
        agent = int(current[item,1])
        agents.append(dict(agent_id=agent,inputs=inp,coordinate_transform=transform))
        for key,value in sorted(inp.items()):
            h.update(key.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
        b.append(inp['baseline_rollouts'][cfg['baseline_index']])
        c.append(inp['baseline_rollouts'][cfg['candidate_index']])
        gains.append(.05+.02*min(float(np.linalg.norm(inp['history_xy'][-1]-inp['history_xy'][-2])),1.))
    if not agents:
        raise ValueError('Past-indexed timestamp unexpectedly empty')
    scene = dict(recording_id=adapter.metadata['id'],physical_scene=adapter.metadata['physical_scene'],
                 frame_id=frame,horizon_raw=144,agents=agents)
    decision = dict(agent_ids=[a['agent_id'] for a in agents],baseline=np.array(b),candidate=np.array(c),
        predicted_gain=np.array(gains),predicted_harm=np.full(len(agents),.01),
        supported=np.ones(len(agents),bool),arms={})
    radius = float(np.median([a['coordinate_transform']['scale'] for a in agents]))
    geometry = dict(graph_radius=radius,proximity_threshold=radius*.1)
    return scene,decision,geometry,h.hexdigest()


def numerical_regression(problem, comparison):
    """Reproduce the original numerical failure without editing its frozen solver."""
    p = problem
    k, n = comparison['reference_count'], len(p.expected_gain)
    captured = {}
    original = legacy.milp
    def capture(**kw):
        result = original(**kw)
        captured.update(kwargs=kw, result=result)
        return result
    legacy.milp = capture
    try:
        old = legacy.select_interventions(p,mode='joint',exact_interventions=k)
    finally:
        legacy.milp = original
    parts = decompose_pair_objective(p)
    best, count = np.inf, 0
    for selected in combinations(range(n),k):
        x = np.zeros(n,dtype=bool); x[list(selected)] = True
        if (np.any(x & ~p.supported) or np.mean(x*parts['original_coherent_harm']) > p.max_mean_predicted_harm+1e-12):
            continue
        value = float(parts['unary_objective']@x+parts['product_objective']@(x[p.edges[:,0]] & x[p.edges[:,1]]))
        best = min(best,value); count += 1
    old_r = captured['result']
    variants = []
    for name,scale,abs_gap in [('absolute_gap_zero',1.,True),('global_scale_1e6',1e6,False)]:
        kw = captured['kwargs']
        options = {**kw['options']}
        if abs_gap: options['mip_abs_gap'] = 0.
        with warnings.catch_warnings(record=True) as caught:
            r = original(**{**kw,'options':options,'c':kw['c']*scale})
        x = r.x[:n] > .5
        value = float(parts['unary_objective']@x+parts['product_objective']@(x[p.edges[:,0]] & x[p.edges[:,1]]))
        variants.append(dict(name=name,success=bool(r.success),recomputed_objective=value,
            error_to_enumerated_optimum=value-best,reported_primal=float(r.fun/scale),
            reported_dual=float(r.mip_dual_bound/scale),warnings=[str(w.message) for w in caught]))
    new = comparison['controls']['joint']
    np.testing.assert_allclose(new['full_objective'],best,atol=1e-10,rtol=0)
    return dict(agents=n,edges=len(p.edges),exact_count=k,enumerated_feasible_assignments=count,
        enumerated_optimum=best,legacy_success=bool(old_r.success),
        legacy_reported_primal=float(old_r.fun),legacy_reported_dual=float(old_r.mip_dual_bound),
        legacy_reported_relative_gap=float(old_r.mip_gap),legacy_direct_objective=old['objective'],
        legacy_error_to_enumerated_optimum=old['objective']-best,
        corrected_objective=new['full_objective'],corrected_numerical=new['numerical'],
        controlled_variants=variants,diagnosis='scale_sensitive_numerical_termination_not_pair_decomposition_error',
        default_absolute_gap_alone_proven_cause=False,frozen_legacy_source_changed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT/CONFIG).read_text())
    out = ROOT/cfg['reports']; out.mkdir(parents=True,exist_ok=True)
    report_path = out/'checks.json'
    if report_path.exists() and not args.verify:
        raise FileExistsError('Completed checks are immutable; use --verify')
    identity = {p:file_digest(ROOT/p) for p in DEPENDENCIES}
    if file_digest(ROOT/cfg['source_population_audit']) != cfg['source_population_audit_sha256']:
        raise ValueError('Changed admitted source population')
    identity[cfg['source_population_audit']] = cfg['source_population_audit_sha256']
    reg = load_registration(ROOT,ROOT/cfg['source_registration'])
    entries = [e for e in source_entries(ROOT,reg) if e['annotation_key'].split('/')[0] in cfg['allowed_sites']]
    if len(entries) != cfg['expected_recordings']:
        raise ValueError('Wrong source roster')
    began = time.monotonic()
    synthetic = synthetic_checks(cfg)
    rows, label_calls, poison_checks, agents_checked = [],0,0,0
    numerical_case = None
    def no_labels(*a,**kw):
        nonlocal label_calls
        label_calls += 1
        raise AssertionError('No label API may be called by engineering probes')
    for entry in entries:
        key = entry['annotation_key']; path = ROOT/entry['annotations_path']
        if file_digest(path) != entry['annotations_sha256']:
            raise ValueError('Changed source annotation')
        identity[entry['annotations_path']] = entry['annotations_sha256']
        raw, labels = load_source(path)
        adapter = SDDStepAdapter(raw,labels,key,12)
        adapter.get_labels = adapter.get_scene_labels = no_labels
        frames = np.unique(adapter.points[adapter.index['current_row'],0]).astype(int)
        selected = frames[np.unique(np.linspace(0,len(frames)-1,cfg['queries_per_recording'],dtype=int))]
        for frame in selected:
            scene,decision,geometry,digest = inputs_at(adapter,int(frame),cfg)
            result = attach_interaction_controls(scene,decision,policy=cfg['policy'],geometry=geometry,
                                                 time_limit_seconds=cfg['solver_seconds'])
            future_points = adapter.points[:,0] > frame
            future_source = adapter.source[:,5] > frame
            saved_points = adapter.points[future_points,2:].copy()
            saved_source = adapter.source[future_source,1:5].copy()
            adapter.points[future_points,2:] = np.nan
            adapter.source[future_source,1:5] = np.nan
            try:
                ps,pd,pg,ph = inputs_at(adapter,int(frame),cfg)
                assert ph == digest and pg == geometry
                after = attach_interaction_controls(ps,pd,policy=cfg['policy'],geometry=pg,
                                                    time_limit_seconds=cfg['solver_seconds'])
                assert result['agent_ids'] == after['agent_ids']
                for name in result['arms']:
                    np.testing.assert_array_equal(result['arms'][name]['switch'],after['arms'][name]['switch'])
                    np.testing.assert_array_equal(result['arms'][name]['prediction'],after['arms'][name]['prediction'])
                poison_checks += 1
            finally:
                adapter.points[future_points,2:] = saved_points
                adapter.source[future_source,1:5] = saved_source
            r = result['interaction_controls']
            if key == cfg['numerical_regression']['recording'] and frame == cfg['numerical_regression']['frame']:
                problem = make_control_problem(scene,decision,policy=cfg['policy'],geometry=geometry)
                numerical_case = numerical_regression(problem,r)
            row = {k:v for k,v in r.items() if k != 'controls'}
            row.update(recording=key,frame=int(frame),agents=len(scene['agents']),input_sha256=digest,
                controls={k:{f:v[f] for f in ('solver_optimal','reason','full_objective','unary_objective',
                    'product_objective','mean_predicted_harm','mean_pair_proxy','predicted_constraints_satisfied',
                    'solver_version','numerical')}
                    for k,v in r['controls'].items()})
            rows.append(row); agents_checked += len(scene['agents'])
        print(json.dumps(dict(recording=key,queries_done=len(rows),agents_checked=agents_checked,
            elapsed_seconds=time.monotonic()-began,pid=os.getpid())),flush=True)
    if numerical_case is None:
        raise RuntimeError('Original numerical failure case was not replayed')
    report = dict(result_source='fresh_run_constructed_problems_and_cached_verified_raw_input_checks',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                         machine=platform.machine()),
        identity=identity,synthetic=synthetic,numerical_regression=numerical_case,
        real_inputs=dict(recordings=len(entries),queries=len(rows),
            agent_queries=agents_checked,poison_checks=poison_checks,label_api_calls=label_calls,
            matched_queries=sum(r['matched'] for r in rows),
            nonzero_matches=sum(r['nonzero_matched'] for r in rows),
            unmatched_queries=sum(not r['matched'] for r in rows),
            potential_product_queries=sum(r['potential_nonadditive_edges_at_count'] > 0 for r in rows),
            controls_differing_in_switch_identity=sum(r['joint_minus_unary_switch_identities'] > 0 for r in rows),
            constructed_score_proxy_advantage_queries=sum(r['predicted_full_objective_advantage'] is not None
                and r['predicted_full_objective_advantage'] > 1e-10 for r in rows),
            cases=rows,score_source='constructed_past_only_engineering_values_not_trained_gain_or_harm',
            forecasts='fixed_causal_CV_and_damped010_not_neural_models'),
        no_new_training=True,primary_metric_changed=False,new_deployment=False,
        real_predictive_accuracy='not_run',main_outer_bookstore_external_readout=False,
        stage5c_executed=False,smc_enabled=False)
    if args.verify:
        if json.loads(report_path.read_text()) != report:
            raise ValueError('Completed mechanism check replay differs')
        print('exact semantic replay passed',flush=True)
    else:
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report['real_inputs'].items() if k != 'cases'},indent=2),flush=True)


if __name__ == '__main__':
    main()
