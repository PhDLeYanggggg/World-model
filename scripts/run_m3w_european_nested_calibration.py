"""Frozen-head inference, calibration-only decisions, then held-source readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_excess as parent
from src.world_model.m3w_nested_calibration import (
    role_pairs, support_distance, support_limit, static_guard, decide, calibrate, metrics)
import numpy as np
torch = parent.torch
inter = parent.parent.parent
PUBLIC = parent.PUBLIC.parent/'european_nested_calibration_v1'
PRIVATE = parent.PRIVATE.parent/'european_nested_calibration_v1'
CONFIG = 'configs/m3w_european_nested_calibration_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_nested_calibration.py',
    'src/world_model/m3w_nested_calibration.py', 'tests/test_m3w_nested_calibration.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json = parent.artifact, parent.digest, parent.immutable_json


def beat(state, **kw):
    r = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    inter.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); seal_path = parent.PUBLIC/'verification.json'
    assert digest(seal_path) == cfg['parent_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p) == h, p
    _, data, jobs, old_identity, identity = parent.load()
    assert cfg['risk_budget'] == .02 and cfg['support_quantile'] == .99
    assert not any(cfg[k] for k in ('new_neural_training','independent_roles_read','held_label_selection',
                                    'deployment_changed','stage5c_executed','smc_enabled'))
    bound = dict(parent_seal=artifact(seal_path), rosters=identity['rosters'],
        bindings={p: digest(ROOT/p) for p in FILES}, source_rows=len(data['sites']),
        independent_roles_read=False, forecasters='cached_verified', heads='cached_verified')
    if create: immutable_json(PUBLIC/'registration.json', bound)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == bound
        inter.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, identity, old_identity, bound


def contexts(data, jobs, rosters):
    for j in jobs:
        bank_ids, cv, bank = inter.candidates(j, data)
        for arm in ('dimensionless', 'damped'):
            x, env, _ = inter.inference_features(data['geometry'][bank_ids], cv, bank[arm])
            for controller in range(3):
                if controller == j['old_identity']['fold']: continue
                roles = inter.role_indices(data['sites'], rosters, j['old_identity']['fold'], controller)
                fit, outer = roles['controller'], roles['readout']
                fp, op = np.searchsorted(bank_ids, fit), np.searchsorted(bank_ids, outer)
                np.testing.assert_array_equal(bank_ids[fp], fit); np.testing.assert_array_equal(bank_ids[op], outer)
                producer_sites = sorted(set(data['sites'][roles['producer']]))
                controller_sites = sorted(set(data['sites'][fit])); outer_sites = sorted(set(data['sites'][outer]))
                yield dict(job=j, arm=arm, controller=controller, ids=outer, fit_ids=fit,
                    x=x[op], fit_x=x[fp], env=env[op], prediction=bank[arm][op],
                    producer_sites=producer_sites, controller_sites=controller_sites, outer_sites=outer_sites,
                    pairs=role_pairs(producer_sites, controller_sites, outer_sites),
                    name=inter.group_name(j, controller, arm))


def checked_record(path):
    r = json.loads(path.read_text())
    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    assert r['fit']['step'] == 2000 and r['fit']['complete']
    return r


def infer_head(record, x, envelope, expected_sites, task):
    state = torch.load(ROOT/record['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    pr = state['preprocess']
    assert pr['training_sites'] == expected_sites and state['step'] == 2000 and state['task'] == task
    model = inter.head.initialize_head(state['settings']['width'], pr, state['seed'],
        'all' if task == 'signed_budget_excess' else task, state['mean_envelope'])
    model.load_state_dict(state['model'])
    return inter.head.predict(model, x, envelope, pr), pr


def score_group(c, data, identity, old_identity):
    refs, scores, prs = [], {}, {}
    original_identity = json.loads((inter.PUBLIC/'registration.json').read_text())
    for task in ('utility','easy','all'):
        path = inter.PRIVATE/'heads'/(c['name']+'_'+task)/'complete.json'
        r = checked_record(path); assert r['identity']['experiment'] == original_identity
        p, pr = infer_head(r, c['x'], c['env'], c['controller_sites'], task)
        with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], c['ids']); np.testing.assert_array_equal(z['scores'], p)
        refs.append(artifact(path)); scores[task] = p; prs[task] = pr
    limit = support_limit(support_distance(c['fit_x'], prs['utility']['mean'], prs['utility']['std']),
                          prs['utility']['weights'])
    distance = support_distance(c['x'], prs['utility']['mean'], prs['utility']['std'])
    ensemble = dict(moments=[], excess=[])
    for held in c['controller_sites']:
        name = c['name']+'_held'+held
        expected = sorted(set(c['controller_sites'])-{held})
        for objective, home, ident, task in (
            ('moments', parent.parent.PRIVATE, old_identity, 'all'),
            ('excess', parent.PRIVATE, identity, 'signed_budget_excess')):
            path = home/'heads'/name/'complete.json'; r = checked_record(path)
            assert r['identity']['experiment'] == ident
            p, _ = infer_head(r, c['x'], c['env'], expected, task)
            ensemble[objective].append(parent.model_api.excess(p)); refs.append(artifact(path))
    moving = np.linalg.norm(data['history'][c['ids'], -1]-data['history'][c['ids'], -2], axis=1) > 0
    guard = static_guard(scores['utility'], scores['easy'], moving)
    legacy = inter.allowed(scores['utility'], scores['all'], scores['easy'], moving, .02)
    legacy_rec = json.loads((inter.PRIVATE/'decisions'/(c['name']+'.json')).read_text())
    for ref in legacy_rec['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    with np.load(ROOT/legacy_rec['artifacts']['arrays']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], c['ids']); np.testing.assert_array_equal(z['point'], legacy)
    arrays = dict(ids=c['ids'], guard=guard, supported=distance <= limit,
        support_distance=distance, moving=moving, legacy=legacy,
        **{k: np.mean(v, axis=0)/prs['utility']['cost_scale'] for k, v in ensemble.items()})
    info = dict(heads=refs, support_limit=limit, cost_scale=prs['utility']['cost_scale'],
        producer_sites=c['producer_sites'], controller_sites=c['controller_sites'], outer_sites=c['outer_sites'],
        original_head_and_decision_replay=True, labels_read=False, fits=0)
    return arrays, info


def write_arrays(path, arrays, replay):
    if replay:
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k, v in arrays.items(): np.testing.assert_array_equal(z[k], v)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp.npz'); np.savez(temp, **arrays); os.replace(temp, path)


def infer(cfg, data, jobs, identity, old_identity, bound, *, resume=False, replay=False):
    refs = []
    for c in contexts(data, jobs, bound['rosters']):
        path = PRIVATE/'scores'/(c['name']+'.json'); npz = path.with_suffix('.npz')
        if path.exists() and not replay:
            if not resume: raise ValueError('Existing scores require --resume')
            rec = json.loads(path.read_text()); assert rec['identity'] == bound
            assert artifact(npz) == rec['arrays']
        else:
            if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Keep10GiB; resume completed score groups')
            arrays, info = score_group(c, data, identity, old_identity)
            write_arrays(npz, arrays, replay)
            immutable_json(path, dict(identity=bound, group=c['name'], arrays=artifact(npz), **info))
        refs.append(artifact(path)); beat('inference_replayed' if replay else 'scores_frozen', group=c['name'])
    assert len(refs) == 36
    immutable_json(PUBLIC/'score_freeze.json', dict(identity=bound, groups=refs, groups_count=36,
        new_score_head_inferences=288, full_guard_head_replays=108, legacy_policy_replays=36,
        calibration_labels_read=False, evaluation_labels_read=False))
    if replay: immutable_json(PUBLIC/'inference_replay.json', dict(all_exact=True, groups=36,
        scores=artifact(PUBLIC/'score_freeze.json')))


def read_scores(c, bound):
    rec = json.loads((PRIVATE/'scores'/(c['name']+'.json')).read_text()); assert rec['identity'] == bound
    assert artifact(ROOT/rec['arrays']['path']) == rec['arrays']
    with np.load(ROOT/rec['arrays']['path'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(arrays['ids'], c['ids'])
    return arrays


def candidate_errors(c, data, pos):
    ids = c['ids'][pos]
    return inter.native_errors(c['prediction'][pos].astype(float)+data['origin'][ids, None],
        data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))


def calibrate_all(cfg, data, jobs, bound, *, resume=False, replay=False):
    inter.committed(PUBLIC/'score_freeze.json'); refs = []
    for c in contexts(data, jobs, bound['rosters']):
        z = read_scores(c, bound); sites = data['sites'][c['ids']]
        for pair_id, (cal_sites, held_sites) in enumerate(c['pairs']):
            name = c['name']+f'_pair{pair_id}'; path = PRIVATE/'decisions'/(name+'.json')
            npz = path.with_suffix('.npz')
            if path.exists() and not replay:
                if not resume: raise ValueError('Existing calibrations require --resume')
                rec = json.loads(path.read_text()); assert rec['identity'] == bound
                assert artifact(npz) == rec['arrays']
            else:
                cal = np.flatnonzero(np.isin(sites, cal_sites)); held = np.flatnonzero(np.isin(sites, held_sites))
                # Slice before any outcome access: held labels cannot reach threshold selection.
                error = candidate_errors(c, data, cal)[0]; cv = data['baseline_ade'][c['ids'][cal], 1]
                arrays = dict(ids=c['ids'][held], legacy=z['legacy'][held]); selections = {}
                for objective in cfg['objectives']:
                    for support_name, support in [('plain', np.ones(len(sites), bool)), ('support', z['supported'])]:
                        sel = calibrate(z[objective][cal], z['guard'][cal], support[cal], cv, error, sites[cal],
                            calibration_sites=cal_sites, thresholds=cfg['threshold_grid'],
                            easy_cut=c['job']['design']['easy_cut'], hard_cut=c['job']['design']['hard_cut'],
                            min_selected=cfg['min_selected_per_calibration_site'])
                        selections[objective+'_'+support_name] = sel
                        names = ('guarded','calibrated') if support_name == 'plain' else ('supported','calibrated_supported')
                        for policy, threshold in zip(names, (0., sel['threshold'])):
                            arrays[objective+'_'+policy] = decide(z[objective][held], z['guard'][held], support[held], threshold)
                write_arrays(npz, arrays, replay)
                immutable_json(path, dict(identity=bound, group=name, calibration_sites=cal_sites, held_sites=held_sites,
                    selections=selections, arrays=artifact(npz), held_outcomes_read=False))
            refs.append(artifact(path))
        beat('calibration_replayed' if replay else 'decisions_frozen', group=c['name'])
    assert len(refs) == 216
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=bound, groups=refs, count=216,
        held_outcomes_read=False, independent_roles_read=False, deployment_changed=False))
    if replay: immutable_json(PUBLIC/'calibration_replay.json', dict(all_exact=True, groups=216,
        decisions=artifact(PUBLIC/'decision_freeze.json')))


def evaluate(cfg, data, jobs, bound, replay=False):
    inter.committed(PUBLIC/'decision_freeze.json')
    frozen = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == bound
    for ref in frozen['groups']: assert artifact(ROOT/ref['path']) == ref
    rows, paired = [], []
    for c in contexts(data, jobs, bound['rosters']):
        ae, fe = candidate_errors(c, data, np.arange(len(c['ids'])))
        cv = data['baseline_ade'][c['ids'], 1]; cf = data['baseline_fde'][c['ids'], 1]
        for pair_id, (_, held_sites) in enumerate(c['pairs']):
            name = c['name']+f'_pair{pair_id}'; path = PRIVATE/'decisions'/(name+'.json')
            rec = json.loads(path.read_text()); assert rec['identity'] == bound
            assert artifact(ROOT/rec['arrays']['path']) == rec['arrays']
            with np.load(ROOT/rec['arrays']['path'], allow_pickle=False) as z:
                pos = np.searchsorted(c['ids'], z['ids']); np.testing.assert_array_equal(c['ids'][pos], z['ids'])
                for site in held_sites:
                    use = data['sites'][z['ids']] == site; ix = pos[use]
                    for key in z.files:
                        if key == 'ids': continue
                        take = z[key][use]
                        m = metrics(cv[ix], ae[ix], take, c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
                        known = np.isfinite(cf[ix]); chosen = np.where(take, fe[ix], cf[ix])
                        m['FDE_gain_percent'] = float(100*(1-chosen[known].sum()/cf[ix][known].sum())) if cf[ix][known].sum() > 0 else None
                        ak = np.isfinite(cv[ix]); ca = np.where(take, ae[ix], cv[ix])[ak]
                        m['p95_ADE_over_CV_mean'] = float(np.quantile(ca, .95)/cv[ix][ak].mean()) if cv[ix][ak].mean() > 0 else None
                        m['risk_budget_violation'] = int(m['zero_reference_harmed'] > 0 or (m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio'] > .02))
                        rows.append(dict(site=site, group=name, pair=pair_id, producer=c['job']['key'],
                            controller=c['controller'], candidate=c['arm'], seed=c['job']['old_identity']['seed'], policy=key, metric=m))
        beat('held_scored', group=c['name'])
    index = {(r['producer'], r['controller'], r['pair'], r['site'], r['policy'], r['candidate']): r for r in rows}
    for r in rows:
        if r['candidate'] != 'dimensionless': continue
        k = (r['producer'], r['controller'], r['pair'], r['site'], r['policy'], 'damped')
        d = index[k]['metric']; n = r['metric']
        paired.append(dict(site=r['site'], seed=r['seed'], policy=r['policy'], metric=dict(
            neural_vs_damping_ADE_gain_percent=100*(1-n['selected_error_sum']/d['selected_error_sum']) if d['selected_error_sum'] > 0 else None)))
    sites = sorted(set(data['sites'])); policies = sorted(set(r['policy'] for r in rows))
    def aggregate(rr):
        return {k: inter.paired_localities(rr, sites, k, cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in rr[0]['metric']}
    summary = {arm: {p: aggregate([r for r in rows if r['candidate']==arm and r['policy']==p]) for p in policies} for arm in cfg['candidates']}
    contrasts = {p: aggregate([r for r in paired if r['policy']==p]) for p in policies}
    primary = contrasts['excess_calibrated_supported']['neural_vs_damping_ADE_gain_percent']
    chosen = [r['metric'] for r in rows if r['candidate']=='dimensionless' and r['policy']=='excess_calibrated_supported']
    n = summary['dimensionless']['excess_calibrated_supported']
    gates = dict(primary_neural_vs_damping_lower_CI_positive=primary['ci95'] is not None and primary['ci95'][0] > 0,
        neural_gain_vs_CV_positive=n['all_gain_percent']['point'] is not None and n['all_gain_percent']['point'] > 0,
        every_held_view_easy_preserved=all(r['easy_gain_percent'] is not None and r['easy_gain_percent'] >= -2 for r in chosen),
        every_locality_nonzero_coverage=all(v is not None and v > 0 for v in n['switch_rate']['by_site'].values()),
        no_zero_reference_harm=all(r['zero_reference_harmed']==0 for r in chosen),
        no_observed_risk_violation=all(r['risk_budget_violation']==0 for r in chosen))
    gates['exploratory_complete_screen_pass'] = all(gates.values())
    gates.update(calibration_certificate=False, independent_confirmation=False, deployment_changed=False,
                 stage5c_executed=False, smc_enabled=False)
    result = dict(identity=bound, result_source='fresh_inference_calibration_readout_cached_verified_weights',
        summary=summary, paired_neural_vs_damping=contrasts,
        primary_by_seed={str(s): aggregate([r for r in paired if r['policy']=='excess_calibrated_supported' and r['seed']==s]) for s in cfg['seeds']},
        rows=rows, gates=gates, new_training_updates=0, independent_roles_read=False)
    immutable_json(PRIVATE/'summary_detail.json', result)
    light = {k:v for k,v in result.items() if k != 'rows'}
    light['worst_views'] = {arm: {p: dict(
        worst_easy_gain_percent=min(r['metric']['easy_gain_percent'] for r in rows if r['candidate']==arm and r['policy']==p and r['metric']['easy_gain_percent'] is not None),
        risk_violating_views=sum(r['metric']['risk_budget_violation'] for r in rows if r['candidate']==arm and r['policy']==p),
        zero_reference_harmed_views=sum(r['metric']['zero_reference_harmed'] for r in rows if r['candidate']==arm and r['policy']==p)) for p in policies} for arm in cfg['candidates']}
    immutable_json(PUBLIC/'summary.json', light); immutable_json(PUBLIC/'gates.json', gates)
    if replay: immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, summary=artifact(PUBLIC/'summary.json'), detail=artifact(PRIVATE/'summary_detail.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register','infer','replay_infer','calibrate','replay_calibrate','evaluate','replay_evaluate'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); beat('phase_started', phase=args.phase)
        cfg, data, jobs, identity, old_identity, bound = load(args.phase=='register')
        if args.phase in ('infer','replay_infer'): infer(cfg,data,jobs,identity,old_identity,bound,resume=args.resume,replay=args.phase=='replay_infer')
        elif args.phase in ('calibrate','replay_calibrate'): calibrate_all(cfg,data,jobs,bound,resume=args.resume,replay=args.phase=='replay_calibrate')
        elif args.phase in ('evaluate','replay_evaluate'): evaluate(cfg,data,jobs,bound,replay=args.phase=='replay_evaluate')
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
