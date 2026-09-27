"""A fixed one-objective change against sealed cross-fit moment controllers."""
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
from scripts import run_m3w_european_risk_moment_crossfit as parent
from scripts.replay_m3w_dimensionless_training import exact
from src.world_model import m3w_risk_excess as model_api
from src.evaluation.m3w_risk_excess import metric
import numpy as np
torch = parent.torch
PUBLIC = parent.PUBLIC.parent/'european_risk_excess_v1'
PRIVATE = parent.PRIVATE.parent/'european_risk_excess_v1'
CONFIG = 'configs/m3w_european_risk_excess_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_risk_excess.py', 'src/world_model/m3w_risk_excess.py',
    'src/evaluation/m3w_risk_excess.py', 'tests/test_m3w_risk_excess.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json = parent.artifact, parent.digest, parent.immutable_json


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.parent.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); path = parent.PUBLIC/'verification.json'
    assert digest(path) == cfg['parent_seal_sha256']
    seal = json.loads(path.read_text())
    for p, h in seal['source_bindings'].items(): assert digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert digest(parent.PUBLIC/p) == h, p
    old_cfg, data, jobs, old_identity = parent.load()
    for k in ('head_count', 'head_training', 'seeds', 'candidates'): assert cfg[k] == old_cfg[k]
    assert cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('threshold_search','outer_readout_scored','independent_roles_read',
                                   'deployment_changed','stage5c_executed','smc_enabled'))
    identity = dict(parent_seal=artifact(path), control_predictions=artifact(parent.PUBLIC/'prediction_freeze.json'),
        bindings={p: digest(ROOT/p) for p in FILES}, rosters=old_identity['rosters'], source_rows=len(data['sites']),
        changed_factor='training_objective_only', independent_roles_read=False)
    if create: immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        parent.parent.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, old_identity, identity


def prepare(c, data, old_identity, identity, held):
    name, fit_ids, held_ids, pos, y, pr, old_hid = parent.prepare(c, data, old_identity, held)
    old_path = parent.PRIVATE/'heads'/name/'complete.json'; old = parent.done_record(old_path, old_hid)
    hid = dict(experiment=identity, group=name, seed=old_hid['seed'], matched_control=artifact(old_path),
        fit_sites=pr['training_sites'], held_site=held, features_sha256=old_hid['features_sha256'],
        train_labels_sha256=old_hid['train_labels_sha256'], train_ids_sha256=old_hid['train_ids_sha256'])
    return name, fit_ids, held_ids, pos, y, pr, old, hid


def read_done(path, hid):
    r = json.loads(path.read_text()); assert r['identity'] == hid and r['fit']['step'] == 2000 and r['fit']['complete']
    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    assert r['matched_sampling_exact'] and r['control_inference_exact']
    return r


def paired_state(new, old):
    for k in ('settings','seed','preprocess','mean_envelope','draws','sampler_rng','torch_rng','step'):
        exact(new[k], old[k])
    assert new['task'] == 'signed_budget_excess' and old['task'] == 'all'


def control_prediction(c, pr, old, cfg):
    state = torch.load(ROOT/old['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    exact(state['preprocess'], pr)
    model = parent.parent.head.initialize_head(cfg['head_training']['width'], pr, state['seed'], 'all', state['mean_envelope'])
    model.load_state_dict(state['model'])
    p = parent.parent.head.predict(model, c['x'], c['envelope'], pr)
    with np.load(ROOT/old['artifacts']['scores']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], c['ids']); np.testing.assert_array_equal(z['scores'], p)
    return state, p


def train(cfg, data, jobs, old_identity, identity, resume=False, pilot=False, replay=False):
    refs, checks = [], []
    for c in parent.contexts(data, jobs, old_identity):
        for held in sorted(set(data['sites'][c['ids']])):
            name, fit_ids, held_ids, pos, y, pr, old, hid = prepare(c, data, old_identity, identity, held)
            home = PRIVATE/'heads'/name; done = home/'complete.json'
            if done.exists(): r = read_done(done, hid)
            else:
                if replay: raise FileNotFoundError(done)
                if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Keep10GiB; resume existing checkpoints')
                model, fit = model_api.fit(c['x'][pos], y, data['sites'][fit_ids], c['envelope'][pos], pr,
                    seed=hid['seed'], settings=cfg['head_training'], identity=hid, directory=home,
                    heartbeat=lambda **kw: beat(head=name, **kw), resume=resume,
                    stop_at=cfg['pilot_updates'] if pilot else None)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'))); return
                parts = model_api.predict(model, c['x'], c['envelope'], pr)
                old_state, old_pred = control_prediction(c, pr, old, cfg)
                new_state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
                paired_state(new_state, old_state)
                temp = home/'scores.tmp.npz'
                np.savez(temp, ids=c['ids'], basis=parts, score=model_api.excess(parts), control_score=model_api.excess(old_pred))
                os.replace(temp, home/'scores.npz')
                assert fit['unknown_rows_sampled'] == 0 and fit['parameters'] == 22914
                r = dict(identity=hid, fit=fit, constant_score=float(model_api.excess(pr['constant'][None])[0]),
                    cost_scale=pr['cost_scale'], fitting_rows=len(fit_ids), held_rows=len(held_ids),
                    matched_sampling_exact=True, control_inference_exact=True,
                    artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz')))
                immutable_json(done, r); beat('head_complete', head=name, seconds=fit['seconds'])
            refs.append(artifact(done))
            if replay:
                state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
                assert state['identity'] == hid; exact(state['preprocess'], pr)
                old_state, old_pred = control_prediction(c, pr, old, cfg); paired_state(state, old_state)
                model = parent.parent.head.initialize_head(cfg['head_training']['width'], pr, hid['seed'], 'all', state['mean_envelope'])
                model.load_state_dict(state['model'])
                parts = model_api.predict(model, c['x'], c['envelope'], pr)
                with np.load(home/'scores.npz', allow_pickle=False) as z:
                    for k, v in dict(ids=c['ids'], basis=parts, score=model_api.excess(parts), control_score=model_api.excess(old_pred)).items():
                        np.testing.assert_array_equal(z[k], v)
                checks.append(dict(head=name, new_exact=True, control_exact=True, matched_sampling_exact=True))
    assert len(refs) == 144
    immutable_json(PUBLIC/'prediction_freeze.json', dict(identity=identity, heads=refs, count=len(refs),
        updates=len(refs)*2000, held_labels_used=False, control_inference_fresh_verified=True))
    if replay: immutable_json(PUBLIC/'prediction_replay.json', dict(all_exact=True, checks=checks,
        predictions=artifact(PUBLIC/'prediction_freeze.json')))


def evaluate(cfg, data, jobs, old_identity, identity, replay=False):
    parent.parent.committed(PUBLIC/'prediction_freeze.json')
    frozen = json.loads((PUBLIC/'prediction_freeze.json').read_text()); assert frozen['identity'] == identity
    for ref in frozen['heads']: assert artifact(ROOT/ref['path']) == ref
    rows, local_rows = [], []
    for c in parent.contexts(data, jobs, old_identity):
        ids = c['ids']; sites = data['sites'][ids]
        error = parent.parent.native_errors(c['prediction'].astype(float)+data['origin'][ids, None],
            data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
        y = parent.labels(data['baseline_ade'][ids, 1], error)
        for held in sorted(set(sites)):
            name = parent.parent.group_name(c['job'], c['controller'], c['candidate'])+'_held'+held
            home = PRIVATE/'heads'/name; r = json.loads((home/'complete.json').read_text())
            assert r['identity']['experiment'] == identity
            for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                local = {str(s): metric(z['score'][sites == s], z['control_score'][sites == s], y[sites == s],
                    error[sites == s], r['constant_score'], r['cost_scale'], c['job']['design']['easy_cut'],
                    c['job']['design']['hard_cut']) for s in sorted(set(sites))}
            other = [v for s, v in local.items() if s != held]
            out = {'held_'+k: v for k, v in local[held].items()}
            out.update({'fit_'+k: float(np.mean([v[k] for v in other])) if all(v[k] is not None for v in other) else None
                        for k in local[held]})
            row = dict(site=held, group=name, candidate=c['candidate'], seed=c['job']['old_identity']['seed'], metric=out)
            rows.append(row); local_rows.append(dict(**row, localities=local))
        beat('scored_controller', trial=c['job']['key'], candidate=c['candidate'], controller=c['controller'])
    assert len(rows) == 144
    sites = sorted(set(data['sites']))
    def aggregate(arm, seed=None):
        selected = [r for r in rows if r['candidate'] == arm and (seed is None or r['seed'] == seed)]
        return {key: parent.parent.paired_localities(selected, sites, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for key in rows[0]['metric']}
    summaries = {a: aggregate(a) for a in cfg['candidates']}
    n = summaries['dimensionless']; ci = n['held_MSE_gain_vs_control_percent']['ci95']
    def every(key, fn): return all(v is not None and fn(v) for v in n[key]['by_site'].values())
    gates = dict(primary_signed_score_MSE_improved=ci is not None and ci[0] > 0,
        every_locality_nonzero_coverage=every('held_new_screen_rate', lambda x: x > 0),
        every_locality_screen_positive_harm_within_2percent=every('held_new_screen_positive_harm_ratio', lambda x: x <= .02),
        every_locality_easy_within_2percent=every('held_new_easy_ADE_gain_vs_CV_percent', lambda x: x >= -2),
        no_zero_reference_harm=every('held_new_zero_reference_harmed', lambda x: x == 0))
    gates['exploratory_loss_and_screen_checks_pass'] = all(gates.values())
    gates.update(calibration_certificate=False, full_policy_tested=False, independent_confirmation=False,
                 deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    result = dict(identity=identity, result_source='fresh_loss_refit_and_readout_cached_verified_controls_with_fresh_inference',
        by_candidate=summaries, by_seed={a: {str(s): aggregate(a, s) for s in cfg['seeds']} for a in cfg['candidates']},
        rows=rows, gates=gates, component_moments_identified=False, independent_roles_read=False)
    immutable_json(PRIVATE/'detailed_diagnostic.json', local_rows)
    immutable_json(PUBLIC/'summary.json', result); immutable_json(PUBLIC/'gates.json', gates)
    if replay: immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, summary=artifact(PUBLIC/'summary.json'),
        detail=artifact(PRIVATE/'detailed_diagnostic.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register','pilot','train','replay','evaluate','verify_eval'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('phase_started', phase=args.phase)
        cfg, data, jobs, old_identity, identity = load(args.phase == 'register')
        if args.phase in ('pilot','train','replay'):
            train(cfg, data, jobs, old_identity, identity, resume=args.resume, pilot=args.phase == 'pilot', replay=args.phase == 'replay')
        elif args.phase in ('evaluate','verify_eval'):
            evaluate(cfg, data, jobs, old_identity, identity, replay=args.phase == 'verify_eval')
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
