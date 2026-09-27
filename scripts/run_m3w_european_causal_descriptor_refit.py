"""Matched continuous-descriptor training without reserved-source access."""
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
from scripts import run_m3w_european_fixed_floor_slices as diagnosis
from src.world_model import m3w_causal_descriptor_head as api
import numpy as np
base = diagnosis.parent
PUBLIC = base.PUBLIC.parent/'european_causal_descriptor_refit_v1'
PRIVATE = base.PRIVATE.parent/'european_causal_descriptor_refit_v1'
CONFIG = 'configs/m3w_european_causal_descriptor_refit_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_causal_descriptor_head.py',
    'scripts/run_m3w_european_causal_descriptor_refit.py', 'tests/test_m3w_causal_descriptor_head.py',
    'tests/test_m3w_causal_descriptor_protocol.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load(register=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); sealpath = diagnosis.PUBLIC/'verification.json'
    assert base.digest(sealpath) == cfg['parent_seal_sha256']
    seal = json.loads(sealpath.read_text())
    for p, h in seal['source_bindings'].items(): assert base.digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert base.digest(diagnosis.PUBLIC/p) == h, p
    pc, data, jobs, oid, pid, pbound, bound = base.load()
    assert cfg['head_training'] == pc['head_training'] and cfg['groups'] == cfg['new_heads'] == 108
    assert cfg['descriptor_features'] == list(api.FEATURES) and cfg['risk_budget'] == .02
    assert not any(cfg[k] for k in ('threshold_search', 'independent_roles_read', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent_seal=base.artifact(sealpath), bindings={p: base.digest(ROOT/p) for p in FILES},
        source_rows=len(data['sites']), rosters=oid['rosters'], new_risk_heads=108,
        changed_factor='six_continuous_causal_descriptors_zero_initialized_branch', independent_roles_read=False)
    if register: base.immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        base.inter.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, oid, pid, pbound, bound, identity


def prepare(c, data, pair, pid, pbound, bound, identity):
    name, fit, held, y, pr, old, _, mse, previous = base.control(c, data, pair, pid, pbound, bound)
    path = base.PRIVATE/'heads'/name/'complete.json'; rec = base.done(path, previous)
    state = base.torch.load(base.PRIVATE/'heads'/name/'checkpoint.pt', map_location='cpu', weights_only=False)
    with np.load(base.PRIVATE/'heads'/name/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], c['ids'][held]); control = z['scores'].copy()
    u = api.descriptors(data['geometry'][c['ids']], c['floor'], c['prediction'], c['x'], pr)
    ident = dict(experiment=identity, control=base.artifact(path), roles_and_targets=previous['roles_and_targets'],
        descriptor_features=list(api.FEATURES), fitting_descriptors_sha256=base.inter.array_hash(u[fit]))
    return name, fit, held, y, pr, old, mse, state, control, u, ident, rec


def actions(c, data, held, old, mse, control, new):
    eligible = c['moving'][held] & old['support'] & (old['scores'][:, 5] > old['scores'][:, 6])
    def take(p): return eligible & (p[:, 1] <= .02*p[:, 0]) & (p[:, 3] <= .02*p[:, 2])
    a = take(new); ids = c['ids'][held]
    q = base.api.signed(control.astype(float)).max(1)
    matched = base.parent.api.match_counts(a, eligible, q, base.parent.query_keys(data, ids), data['frames'][ids], ids)
    return dict(ids=ids, eligible=eligible, descriptor=a, control=take(control), mse=take(mse),
        ridge=old['floor_safe'], control_matched_count=matched)


def write_arrays(path, arrays, replay=False):
    if replay:
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k, v in arrays.items(): np.testing.assert_array_equal(v, z[k])
    else:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays); os.replace(tmp, path)


def model_from(state):
    model = api.initialize(state['preprocess'], state['settings']['width'], state['seed'], state['mean_envelope'])
    model.load_state_dict(state['model']); return model


def done(path, identity):
    doc = json.loads(path.read_text()); assert doc['identity'] == identity and doc['fit']['complete'] and doc['fit']['step'] == 2000
    for ref in doc['artifacts'].values(): assert base.artifact(ROOT/ref['path']) == ref
    return doc


def train(cfg, data, jobs, oid, pid, pbound, bound, identity, *, resume=False, pilot=False, replay=False):
    refs = []
    for c in base.floor_api.contexts(data, jobs, oid):
        for pair in range(6):
            name, fit, held, y, pr, old, mse, control_state, control, u, ident, rec = prepare(c, data, pair, pid, pbound, bound, identity)
            home = PRIVATE/'heads'/name; path = home/'complete.json'; checkpoint = home/'checkpoint.pt.gz'
            if path.exists():
                if not resume and not replay: raise ValueError('Use --resume to preserve finished fits')
                doc = done(path, ident)
            else:
                if replay: raise FileNotFoundError(path)
                if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB and completed checkpoints')
                model, info = api.fit(c['x'][fit], u[fit], y, data['sites'][c['ids'][fit]], c['env'][fit], pr,
                    seed=control_state['seed'], settings=cfg['head_training'], identity=ident, directory=home,
                    heartbeat=lambda **kw: beat(head=name, **kw), resume=resume, stop_at=cfg['pilot_updates'] if pilot else None)
                if pilot:
                    base.immutable_json(PRIVATE/'pilot.json', dict(fit=info, checkpoint=base.artifact(checkpoint),
                        checkpoint_bytes=checkpoint.stat().st_size, disk_free_bytes=shutil.disk_usage(PRIVATE).free)); return
                state = api.read_checkpoint(checkpoint); api.assert_matched(state, control_state)
                p = api.predict(model, c['x'][held], u[held], c['env'][held], pr, state['descriptor_preprocess'])
                f = api.predict(model, c['x'][fit], u[fit], c['env'][fit], pr, state['descriptor_preprocess']).astype(float)
                error = ((base.api.signed(f[pr['known']])-base.api.signed(y[pr['known']]))/pr['cost_scale'])**2
                quality = (error*pr['weights'][pr['known'], None]).sum(0).tolist()
                a = actions(c, data, held, old, mse, control, p)
                write_arrays(home/'decisions.npz', a)
                doc = dict(identity=ident, fit=info, fitting_signed_MSE=quality,
                    control_fitting_signed_MSE=rec['training_quality']['excess'],
                    prediction_sha256=base.inter.array_hash(p), held_labels_read=False,
                    artifacts=dict(checkpoint=base.artifact(checkpoint), decisions=base.artifact(home/'decisions.npz')))
                base.immutable_json(path, doc)
            state = api.read_checkpoint(checkpoint); api.assert_matched(state, control_state)
            if replay:
                model = model_from(state)
                p = api.predict(model, c['x'][held], u[held], c['env'][held], pr, state['descriptor_preprocess'])
                assert base.inter.array_hash(p) == doc['prediction_sha256']
                write_arrays(home/'decisions.npz', actions(c, data, held, old, mse, control, p), True)
            refs.append(base.artifact(path)); beat('head_replayed' if replay else 'head_complete', head=name, count=len(refs))
    assert len(refs) == 108
    base.immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, heads=refs, new_heads=108,
        updates=216000, matched_sampler_pairs=108, held_labels_used_for_selection=False))
    if replay: base.immutable_json(PUBLIC/'prediction_replay.json', dict(exact=True, groups=108, future_score_bank_duplicated=False))


def evaluate(cfg, data, jobs, oid, pid, pbound, bound, identity, *, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    freeze = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert freeze['identity'] == identity
    for ref in freeze['heads']: assert base.artifact(ROOT/ref['path']) == ref
    rows = []; quality = []
    for c in base.floor_api.contexts(data, jobs, oid):
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name, fit, held, y, pr, old, mse, cs, control, u, ident, rec = prepare(c, data, pair, pid, pbound, bound, identity)
            home = PRIVATE/'heads'/name; doc = done(home/'complete.json', ident); state = api.read_checkpoint(home/'checkpoint.pt.gz')
            p = api.predict(model_from(state), c['x'][held], u[held], c['env'][held], pr, state['descriptor_preprocess'])
            assert base.inter.array_hash(p) == doc['prediction_sha256']
            a = actions(c, data, held, old, mse, control, p); write_arrays(home/'decisions.npz', a, True)
            for site in c['pairs'][pair][1]:
                local = data['sites'][a['ids']] == site; ix = held[local]; known = np.isfinite(cv[ix])
                common = dict(site=site, seed=c['job']['old_identity']['seed'], group=name)
                for policy in ('floor', 'ridge', 'mse', 'control', 'descriptor', 'control_matched_count'):
                    take = np.zeros(len(ix), bool) if policy == 'floor' else a[policy][local]
                    m = base.floor_api.metric(cv[ix], floor[ix], neural[ix], cf[ix], ff[ix], nf[ix],
                        np.where(take, neural[ix], floor[ix]), np.where(take, nf[ix], ff[ix]), take,
                        data['valid'][c['ids'][ix]], c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
                    rows.append(dict(**common, policy=policy, metric=m))
                target = base.api.signed(base.floor_api.api.targets(cv[ix], floor[ix], neural[ix], c['job']['design']['easy_cut'])[:, [2,1,3,4]])/pr['cost_scale']
                for arm, pred in [('control', control[local]), ('descriptor', p[local])]:
                    err = ((base.api.signed(pred.astype(float))/pr['cost_scale'])[known]-target[known])**2
                    trainq = doc['fitting_signed_MSE'] if arm == 'descriptor' else [doc['control_fitting_signed_MSE'][k] for k in ('all_normalized_excess_MSE','easy_normalized_excess_MSE')]
                    quality.append(dict(**common, policy=arm, metric=dict(all_signed_MSE=float(err[:,0].mean()),
                        easy_signed_MSE=float(err[:,1].mean()), all_fit_MSE=trainq[0], easy_fit_MSE=trainq[1])))
        beat('held_scored', group=c['name'])
    sites = sorted(set(data['sites']))
    def grouped(rr):
        return {p: {k: base.inter.paired_localities([r for r in rr if r['policy']==p], sites, k,
            cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in rr[0]['metric']} for p in sorted(set(r['policy'] for r in rr))}
    idx = {(r['group'],r['site'],r['policy']):r['metric'] for r in rows}; contrasts = []
    chosen = [r for r in rows if r['policy']=='descriptor']
    for r in chosen:
        n = r['metric']
        for control_name in ('control', 'control_matched_count', 'mse', 'ridge'):
            old = idx[(r['group'],r['site'],control_name)]
            harm = None if n['selected_positive_harm_ratio'] is None or old['selected_positive_harm_ratio'] is None else 100*(old['selected_positive_harm_ratio']-n['selected_positive_harm_ratio'])
            contrasts.append(dict(site=r['site'],seed=r['seed'],policy=control_name,metric=dict(positive_harm_reduction_pp=harm,
                ADE_gain_percent=100*(1-n['error_sum']/old['error_sum']) if old['error_sum']>0 else None,
                intervention_difference_pp=100*(n['intervention_rate']-old['intervention_rate']))))
    summary = grouped(rows); paired = grouped(contrasts); primary = paired['control_matched_count']['positive_harm_reduction_pp']
    m = [r['metric'] for r in chosen]; new = summary['descriptor']
    gates = dict(primary_equal_count_harm_reduction=primary['ci95'] is not None and primary['ci95'][0]>0,
        equal_count_ADE_advantage=paired['control_matched_count']['ADE_gain_percent']['ci95'][0]>0,
        floor_ADE_advantage=new['all_gain_floor']['ci95'][0]>0,
        nonzero_each_locality=all(v is not None and v>0 for v in new['intervention_rate']['by_site'].values()),
        every_view_easy_preserved=all(v['easy_gain_CV'] is not None and v['easy_gain_CV']>=-2 for v in m),
        no_zero_CV_harm=all(v['zero_CV_harmed']==0 for v in m),
        every_view_defined_risk_within_budget=all(v['selected_positive_harm_ratio'] is not None and v['selected_positive_harm_ratio']<=.02 for v in m))
    gates['exploratory_joint_screen_pass'] = all(gates.values())
    gates.update(independent_confirmation=False, calibration_certificate=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    doc = dict(identity=identity, result_source='fresh_causal_descriptor_training_readout_cached_verified_controls',
        summary=summary, paired=paired, quality=grouped(quality), gates=gates, new_heads=108, updates=216000,
        by_seed={str(s):grouped([r for r in rows if r['seed']==s]) for s in cfg['seeds']},
        worst_views={p:dict(worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None),
            risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
            undefined_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p)) for p in summary},
        independent_roles_read=False)
    base.immutable_json(PRIVATE/'details.json', dict(rows=rows, quality=quality, contrasts=contrasts))
    base.immutable_json(PUBLIC/'summary.json', doc); base.immutable_json(PUBLIC/'gates.json', gates)
    if replay: base.immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True, summary=base.artifact(PUBLIC/'summary.json'), details=base.artifact(PRIVATE/'details.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register', 'pilot', 'train', 'replay', 'evaluate', 'replay_evaluate'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    base.torch.set_num_threads(4); base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat('phase_started', phase=args.phase)
        values = load(args.phase=='register')
        if args.phase in ('pilot', 'train', 'replay'): train(*values, resume=args.resume, pilot=args.phase=='pilot', replay=args.phase=='replay')
        elif args.phase in ('evaluate', 'replay_evaluate'): evaluate(*values, replay=args.phase=='replay_evaluate')
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
