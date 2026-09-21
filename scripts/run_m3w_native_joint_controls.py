"""Registered native joint controls with separate decision and label-readout phases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_scene_context import load_past_queries
from scripts.run_m3w_native_forecast import array_hash, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_scene_alignment import restore
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.world_model.m3w_native_joint_controls import make_problem, compare
import numpy as np
import scipy
import torch

CONFIG = 'configs/m3w_native_joint_controls_v1.json'
CODE = ('scripts/run_m3w_native_joint_controls.py','src/world_model/m3w_native_joint_controls.py',
        'src/world_model/m3w_joint_intervention.py','src/world_model/m3w_interaction_controls.py',
        'tests/test_m3w_native_joint_controls.py','scripts/build_m3w_native_scene_context.py')
ARMS = ('floor','uncontrolled','full_independent','full_unary','full_joint','full_scene_uniform',
        'half_independent','half_unary','half_joint','half_scene_uniform')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    cp = ROOT/'outputs/publication_readiness_2026_09/native_scene_context_v2/analysis.json'
    context = json.loads(cp.read_text()); cv = json.loads(cp.with_name('verification.json').read_text())
    gp = ROOT/'outputs/publication_readiness_2026_09/native_geometric_risk_v1/analysis.json'
    prior = json.loads(gp.read_text())
    if (file_digest(cp) != cfg['context_analysis_sha256'] or not cv['all_checks_passed']
            or cv['analysis_sha256'] != file_digest(cp) or file_digest(gp) != cfg['parent_analysis_sha256']
            or cfg['sites'] != ['coupa','deathCircle','gates','hyang'] or cfg['seeds'] != [17,29,43]
            or cfg['budgets'] != dict(full=1., half=.5)
            or cfg['geometry'] != dict(past_radius='median_target_past_scale', threshold_fraction=.1, pair_weight=1.)
            or any(cfg[k] for k in ('new_training','model_selection','threshold_search','risk_calibration',
                                  'independent_confirmation','deployment','closed_role_readout','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed source-only experiment and verified context required')
    bindings = dict(context['source_bindings'])
    assert_current(dict(source_bindings=bindings))
    def bind(path, expected=None):
        path = Path(path); key = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
        sha = file_digest(ROOT/key)
        if expected is not None and sha != expected: raise ValueError('Changed dependency: '+key)
        bindings[key] = sha
        return sha
    for p in (cp, cp.with_name('verification.json'), gp, CONFIG, cfg['registration'], *CODE): bind(p)
    gain_path = ROOT/'outputs/publication_readiness_2026_09/native_gain_harm_v1/analysis.json'
    gain = json.loads(gain_path.read_text()); bind(gain_path)
    materialized_path = ROOT/'outputs/publication_readiness_2026_09/native_nested_v1/materialized_views.json'
    materialized = json.loads(materialized_path.read_text()); bind(materialized_path)
    assert materialized['all_checks_passed'] and materialized['outer_rows_in_training'] == 0
    views = {}
    for r in materialized['views']:
        bind(r['receipt_path'], r['receipt_sha256'])
        meta = json.loads((ROOT/r['receipt_path']).read_text())
        key = f"{r['outer_site']}_seed{r['seed']}"; views[key] = meta
        op = meta['outer_prediction']['prediction']; bind(op['path'], op['sha256'])
    for r in gain['score_archives']: bind(r['path'], r['sha256'])
    for r in gain['training']:
        if r['arm'] == 'ridge': bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in context['records']: bind(r['cache']['path'], r['cache']['sha256'])
    align = json.loads((ROOT/'outputs/publication_readiness_2026_09/native_scene_alignment_v1/analysis.json').read_text())
    bind(align['cache']['path'], align['cache']['sha256'])
    with np.load(ROOT/align['cache']['path'], allow_pickle=False) as z:
        transform = {k:z[k] for k in ('origin','rotation','stored_metric_scale')}
    data = load_past_queries(bindings)
    data['sites'] = np.array([r.split('/')[0] for r in data['recordings']])
    data['scale'] = transform['stored_metric_scale']
    data.update(transform)
    assert len(data['sites']) == 175756 and len(context['records']) == 33 and len(views) == 12
    assert_current(dict(source_bindings=bindings))
    identity = dict(source_bindings=bindings, config=cfg, arms=list(ARMS), numpy=np.__version__,
        scipy=scipy.__version__, torch=torch.__version__, architecture=platform.machine(),
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0), target_arrays_loaded=False)
    return cfg, data, context, views, gain, prior, identity


def view_scores(key, meta, gain, data):
    with np.load(ROOT/meta['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
        ids, prediction = z['ids'].copy(), z['prediction'].copy()
    score = next(r for r in gain['score_archives'] if r['view'] == key)
    with np.load(ROOT/score['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); cost = z['mse'].copy()
    rec = next(r for r in gain['training'] if r['view'] == key and r['arm'] == 'ridge')
    pr = torch.load(ROOT/rec['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
    assert set(pr['training_sites']) == set(data['sites'])-{meta['outer_site']}
    baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
    xy = data['geometry'][ids, :16].reshape(-1, 8, 2)
    eligible = ((cost[:, 0] > cost[:, 1]) & (cost[:, 1] <= .1*cost[:, 0])
        & ~np.all(prediction == baseline, axis=(1, 2)) & ~np.all(xy[:, -1] == xy[:, -2], axis=1))
    return ids, prediction, baseline, cost, eligible, pr


def group_decisions(c, rows, data, ids_to_view, prediction, baseline, cost, eligible, pr, cfg):
    target_ids = c['context_target_rows'][rows]
    target = target_ids >= 0; tid = target_ids[target]; loc = ids_to_view[tid]
    if (loc < 0).any(): raise ValueError('Target outside predictor view')
    n = len(rows); valid = c['context_cv_valid'][rows].all(1)
    b = c['context_cv_rollout'][rows].copy(); cand = b.copy()
    b[target] = restore(baseline[loc], data['origin'][tid], data['rotation'][tid], data['scale'][tid])
    cand[target] = restore(prediction[loc], data['origin'][tid], data['rotation'][tid], data['scale'][tid])
    # Forecast support is a past-only input property, not future label support.
    valid[target] = True
    support = np.zeros(n, bool); support[target] = eligible[loc]
    benefit, harm = np.zeros(n), np.zeros(n)
    benefit[target], harm[target] = cost[loc, 0], cost[loc, 1]
    p, geometry = make_problem(baseline=b, candidate=cand, current=c['context_xy'][rows],
        forecast_valid=valid, target_mask=target, eligible=support, benefit=benefit, harm=harm,
        scale=pr['cost_scale'], past_target_scales=data['scale'][tid])
    decisions = dict(floor=np.zeros(n, bool), uncontrolled=target.copy()); summaries = {}
    for name, fraction in cfg['budgets'].items():
        chosen, summary = compare(p, c['context_agent_ids'][rows], target, fraction, cfg['solver_seconds'])
        decisions.update({name+'_'+k:v for k,v in chosen.items()}); summaries[name] = summary
    choices = np.column_stack([decisions[k][target] for k in ARMS])
    pairs = {}
    for arm, bits in decisions.items():
        picked = p.pair_cost[np.arange(len(p.edges)), bits[p.edges[:, 0]].astype(int), bits[p.edges[:, 1]].astype(int)]
        pairs[arm] = dict(sum=float(picked.sum()), edges=len(picked))
    return tid, choices, dict(geometry=geometry, budgets=summaries, pair_proxy=pairs,
        context_rows=n, target_rows=int(target.sum()))


def decide(cfg, data, context, views, gain, prior, identity, args, beat):
    output = ROOT/cfg['output']; immutable_json(output/'identity.json', identity)
    identity_sha = file_digest(output/'identity.json')
    receipts, new, reused, all_groups = [], 0, 0, 0
    for key, meta in views.items():
        ids, pred, baseline, cost, eligible, pr = view_scores(key, meta, gain, data)
        cap = next(r for r in prior['capacities'] if r['view'] == key)
        assert array_hash(ids[eligible]) == cap['selected_ids_sha256']['stop_mse_strict']
        index = np.full(len(data['sites']), -1, np.int64); index[ids] = np.arange(len(ids))
        for record in context['records']:
            rec = record['recording']
            if rec.split('/')[0] != meta['outer_site']: continue
            with np.load(ROOT/record['cache']['path'], allow_pickle=False) as z:
                c = {k:z[k].copy() for k in z.files}
            frames = np.unique(c['context_frame_ids'])
            for offset in range(0, len(frames), cfg['checkpoint_queries']):
                part = frames[offset:offset+cfg['checkpoint_queries']]
                folder = output/'decisions'/key/rec
                path = folder/f'{offset:06d}.npz'; mp = path.with_suffix('.json')
                if mp.exists():
                    old = json.loads(mp.read_text())
                    if old['identity_sha256'] != identity_sha or old['sha256'] != file_digest(path) or old['frames'] != part.tolist():
                        raise ValueError('Changed decision checkpoint')
                    if not args.verify:
                        receipts.append(dict(path=str(mp.relative_to(ROOT)), sha256=file_digest(mp)))
                        reused += len(part); all_groups += len(part); continue
                elif args.verify:
                    raise ValueError('Replay cannot create decisions')
                all_ids, all_choices, queries = [], [], []
                for frame in part:
                    rows = np.flatnonzero(c['context_frame_ids'] == frame)
                    tid, choices, report = group_decisions(c, rows, data, index, pred, baseline, cost, eligible, pr, cfg)
                    all_ids.append(tid); all_choices.append(choices)
                    queries.append(dict(recording=rec, frame=int(frame), view=key, **report))
                values = dict(ids=np.concatenate(all_ids), choices=np.concatenate(all_choices))
                write_arrays(path, values)
                record_out = dict(identity_sha256=identity_sha, sha256=file_digest(path), path=str(path.relative_to(ROOT)),
                    frames=part.tolist(), view=key, recording=rec, ids_sha256=array_hash(values['ids']),
                    choices_sha256=array_hash(values['choices']), queries=queries)
                immutable_json(mp, record_out)
                receipts.append(dict(path=str(mp.relative_to(ROOT)), sha256=file_digest(mp)))
                new += len(part); all_groups += len(part)
                beat(state='replaying_decisions' if args.verify else 'past_only_decisions', view=key, recording=rec,
                     groups_done=all_groups, new_groups=new, reused_groups=reused)
                if args.pilot_queries and new >= args.pilot_queries:
                    beat(state='pilot_complete_full_run_not_complete', groups_done=all_groups); return
    completion = dict(identity_sha256=identity_sha, receipts=receipts, scene_queries=all_groups,
        future_target_arrays_loaded=False, decisions_complete=True, new_training=False)
    assert all_groups == 20932*3
    assert_current(identity); immutable_json(output/'decisions_complete.json', completion)
    if args.verify:
        immutable_json(ROOT/cfg['reports']/'decision_replay.json', dict(decision_manifest_sha256=file_digest(output/'decisions_complete.json'),
            scene_queries_replayed=new, choices_replayed=175756*3*len(ARMS), all_checks_passed=True))
    beat(state='decisions_verified' if args.verify else 'decisions_complete', groups=all_groups)


def evaluate(cfg, data, views, gain, prior, identity, beat):
    output = ROOT/cfg['output']; completed = json.loads((output/'decisions_complete.json').read_text())
    assert completed['decisions_complete'] and not completed['future_target_arrays_loaded']
    assert json.loads((output/'identity.json').read_text()) == identity
    assert completed['identity_sha256'] == file_digest(output/'identity.json')
    n = len(data['sites']); seeds = cfg['seeds']
    selections = {s:np.zeros((n, len(ARMS)), bool) for s in seeds}; seen = {s:np.zeros(n, bool) for s in seeds}
    queries = []
    for r in completed['receipts']:
        path = ROOT/r['path']; assert file_digest(path) == r['sha256']; receipt = json.loads(path.read_text())
        assert file_digest(ROOT/receipt['path']) == receipt['sha256']
        seed = int(receipt['view'].rsplit('seed', 1)[1])
        with np.load(ROOT/receipt['path'], allow_pickle=False) as z:
            ids, choices = z['ids'], z['choices']
            assert not seen[seed][ids].any(); seen[seed][ids] = True; selections[seed][ids] = choices
        queries.extend(receipt['queries'])
    assert all(v.all() for v in seen.values())
    # All decision bytes are now fixed. Future arrays enter only this readout.
    source = ROOT/'data/stage_cvpr2027_experiments/sdd_auxiliary_v1/inputs'
    target = np.empty((n, 12, 2), np.float32); valid = np.zeros((n, 12), bool)
    for rec in np.unique(data['recordings']):
        ids = np.flatnonzero(data['recordings'] == rec)
        target[ids] = np.load(source/rec/'target.npy', allow_pickle=False)
        valid[ids] = np.load(source/rec/'valid.npy', allow_pickle=False)
    cv, cf = native_errors(data['geometry'][:, 332:356].reshape(n, 12, 2), target, valid, data['scale'])
    complete = valid.all(1); masks = dict(complete=complete, zero_CV=complete & (cv == 0),
        hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    ne = {s:np.full(n, np.nan) for s in seeds}; nf = {s:np.full(n, np.nan) for s in seeds}
    for key, meta in views.items():
        ids, prediction, _, _, eligible, pr = view_scores(key, meta, gain, data)
        seed = meta['seed']
        for name in ('full_independent','full_unary','full_joint'):
            np.testing.assert_array_equal(selections[seed][ids, ARMS.index(name)], eligible)
        ne[seed][ids], nf[seed][ids] = native_errors(prediction, target[ids], valid[ids], data['scale'][ids])
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
    def metric(m, b, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(m[mask], b[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'], seed=38113)
    summaries = {}
    for j, arm in enumerate(ARMS):
        details, errors = {}, []
        for seed in seeds:
            use = selections[seed][:, j]; ade, fde = np.where(use, ne[seed], cv), np.where(use, nf[seed], cf)
            errors.append(ade)
            details[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf),
                subsets={k:metric(ade, cv, mask) for k,mask in masks.items()},
                selected_rows=int(use.sum()), intervention_rate=float(use.mean()),
                unknown_ADE_selected_rows=int((use & ~valid.any(1)).sum()), incomplete_risk_selected_rows=int((use & ~complete).sum()),
                complete_zero_CV_harmed_rows=int((ade[masks['zero_CV']] > 0).sum()),
                complete_zero_CV_max_harm=float(ade[masks['zero_CV']].max()))
        average = np.mean(errors, axis=0)
        summaries[arm] = dict(seeds=details, ADE=metric(average, cv),
            subsets={k:metric(average, cv, mask) for k,mask in masks.items()})
    contrasts = {}
    for left, right in [('half_joint','half_unary'),('half_unary','half_independent'),('half_joint','half_independent')]:
        aa, bb = summaries[left]['ADE']['by_scene'], summaries[right]['ADE']['by_scene']
        contrasts[left+'_minus_'+right] = paired_scene_contrast([aa[s]['gain_percent'] for s in cfg['sites']], [bb[s]['gain_percent'] for s in cfg['sites']])
    mechanism = {}
    for budget in cfg['budgets']:
        rows = [q['budgets'][budget] for q in queries]
        mechanism[budget] = dict(scene_queries=len(rows), matched_queries=sum(r['matched'] for r in rows),
            nonzero_count_queries=sum(r['count'] > 0 for r in rows), count_sum=sum(r['count'] for r in rows),
            queries_with_nonadditive_opportunity=sum(r['nonadditive_supported_edges'] > 0 for r in rows),
            queries_with_changed_joint_unary=sum(r['joint_unary_changed_agents'] > 0 for r in rows),
            joint_unary_changed_agent_instances=sum(r['joint_unary_changed_agents'] for r in rows),
            queries_with_changed_unary_reference=sum(r['unary_reference_changed_agents'] > 0 for r in rows),
            objective_advantage_sum=float(sum(r['objective_advantage'] for r in rows if r['objective_advantage'] is not None)),
            statuses={name:{reason:sum(r['controls'][name]['reason'] == reason for r in rows)
                for reason in sorted({r['controls'][name]['reason'] for r in rows})} for name in ('independent','unary','joint')})
    proxy = {arm:dict(known_edge_excess_sum=float(sum(q['pair_proxy'][arm]['sum'] for q in queries)),
        known_edges=sum(q['pair_proxy'][arm]['edges'] for q in queries)) for arm in ARMS}
    result = dict(result_source='fresh_run_fixed_coupling_readout_cached_verified_native_forecasters_and_costs',
        identity=identity, decision_manifest_sha256=file_digest(output/'decisions_complete.json'),
        summaries=summaries, contrasts=contrasts, mechanism=mechanism, known_context_proxy=proxy,
        unknown_forecast_edges=sum(q['geometry']['unknown_forecast_edges'] for q in queries),
        queries_with_unsupported_context=sum(q['geometry']['unsupported_context_rows'] > 0 for q in queries),
        rows=n, repeated_query_seed_instances=n*len(seeds), scene_queries=len(queries),
        new_training=False, independent_confirmation=False, risk_calibrated=False,
        deployment=False, threshold_search=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(ROOT/cfg['reports']/'analysis.json', result)
    beat(state='evaluated', analysis_sha256=file_digest(ROOT/cfg['reports']/'analysis.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only','evaluate','verify','resume'): p.add_argument('--'+name, action='store_true')
    p.add_argument('--pilot-queries', type=int)
    args = p.parse_args()
    if sum((args.audit_only,args.evaluate,args.verify)) > 1 or (args.pilot_queries is not None and args.pilot_queries <= 0):
        raise ValueError('One explicit phase and positive pilot count required')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, context, views, gain, prior, identity = load()
    directory = ROOT/cfg['output']; directory.mkdir(parents=True, exist_ok=True)
    def beat(**v):
        value = dict(pid=os.getpid(), updated_unix=time.time(), **v)
        json_write(directory/'heartbeat.json', value); print(json.dumps(value), flush=True)
    with (directory/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), rows=len(data['sites']), target_arrays_loaded=False)
        elif args.evaluate:
            evaluate(cfg, data, views, gain, prior, identity, beat)
        else:
            decide(cfg, data, context, views, gain, prior, identity, args, beat)


if __name__ == '__main__': main()
