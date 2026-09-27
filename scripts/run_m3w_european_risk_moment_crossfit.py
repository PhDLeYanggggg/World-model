"""Controller-only three-fit/one-held cost-moment diagnostics."""
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
from scripts import run_m3w_european_dimensionless_intervention as parent
from src.evaluation.m3w_risk_moment_crossfit import (split_controller, labels, score_edges,
    summarize, flatten_pair, MEASURES)
import numpy as np
torch = parent.torch
PUBLIC = parent.PUBLIC.parent/'european_risk_moment_crossfit_v1'
PRIVATE = parent.PRIVATE.parent/'european_risk_moment_crossfit_v1'
CONFIG = 'configs/m3w_european_risk_moment_crossfit_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_risk_moment_crossfit.py',
    'src/evaluation/m3w_risk_moment_crossfit.py', 'tests/test_m3w_risk_moment_crossfit.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json = parent.artifact, parent.digest, parent.immutable_json


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    seal_path = parent.PUBLIC/'verification.json'
    assert digest(seal_path) == cfg['parent_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for path, value in seal['source_bindings'].items(): assert digest(ROOT/path) == value, path
    for path, value in seal['artifacts'].items(): assert digest(parent.PUBLIC/path) == value, path
    _, data, jobs, previous, _ = parent.load()
    assert cfg['head_count'] == 144 and cfg['head_training']['steps'] == 2000
    assert cfg['candidates'] == ['dimensionless', 'damped'] and cfg['seeds'] == [17, 29, 43]
    assert not any(cfg[k] for k in ('policy_selection','outer_readout_scored','independent_roles_read',
                                   'deployment_changed','stage5c_executed','smc_enabled'))
    identity = dict(parent_seal=artifact(seal_path), rosters=previous['rosters'],
        bindings={p: digest(ROOT/p) for p in FILES}, source_rows=len(data['sites']),
        future_inputs=False, outer_readout_scored=False, independent_roles_read=False)
    if create: immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        parent.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, identity


def contexts(data, jobs, identity):
    for j in jobs:
        bank_ids, cv_rollout, bank = parent.candidates(j, data)
        for candidate in ('dimensionless', 'damped'):
            prediction = bank[candidate]
            x, envelope, _ = parent.inference_features(data['geometry'][bank_ids], cv_rollout, prediction)
            for controller in range(3):
                if controller == j['old_identity']['fold']: continue
                roles = parent.role_indices(data['sites'], identity['rosters'], j['old_identity']['fold'], controller)
                ids = roles['controller']; pos = np.searchsorted(bank_ids, ids)
                np.testing.assert_array_equal(bank_ids[pos], ids)
                yield dict(job=j, candidate=candidate, controller=controller, ids=ids, x=x[pos],
                    envelope=envelope[pos], prediction=prediction[pos],
                    producer_sites=sorted(set(data['sites'][roles['producer']])),
                    readout_sites=sorted(set(data['sites'][roles['readout']])))


def targets(data, ids, prediction):
    error = parent.native_errors(prediction.astype(float)+data['origin'][ids, None],
        data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
    return labels(data['baseline_ade'][ids, 1], error)


def prepare(c, data, identity, held):
    ids = c['ids']; sites = data['sites'][ids]
    fit_ids, held_ids = split_controller(ids, sites, held, c['producer_sites'], c['readout_sites'])
    fit_pos = np.searchsorted(ids, fit_ids)
    # Only the three fitting localities supply supervision and sufficient statistics.
    y = targets(data, fit_ids, c['prediction'][fit_pos])
    pr = parent.preprocess(c['x'][fit_pos], y, data['baseline_ade'][fit_ids, 1], data['sites'][fit_ids], held)
    name = parent.group_name(c['job'], c['controller'], c['candidate'])+'_held'+held
    hid = dict(experiment=identity, group=name, held_site=held, fitting_sites=pr['training_sites'],
        train_ids_sha256=parent.array_hash(fit_ids), train_labels_sha256=parent.array_hash(y),
        features_sha256=parent.array_hash(ids, c['x'], c['envelope']),
        seed=c['job']['old_identity']['seed'], task='all')
    return name, fit_ids, held_ids, fit_pos, y, pr, hid


def done_record(path, hid):
    rec = json.loads(path.read_text())
    assert rec['identity'] == hid and rec['fit']['step'] == 2000 and rec['fit']['complete']
    for ref in rec['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    return rec


def train(cfg, data, jobs, identity, resume=False, pilot=False, replay=False):
    refs, checks = [], []
    for c in contexts(data, jobs, identity):
        for held in sorted(set(data['sites'][c['ids']])):
            name, fit_ids, held_ids, pos, y, pr, hid = prepare(c, data, identity, held)
            home = PRIVATE/'heads'/name; done = home/'complete.json'
            if done.exists():
                rec = done_record(done, hid)
            else:
                if replay: raise FileNotFoundError(done)
                if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Keep10GiB; use saved resume checkpoints')
                model, fit = parent.head.fit(c['x'][pos], y, data['sites'][fit_ids], c['envelope'][pos], pr,
                    seed=hid['seed'], task='all', settings=cfg['head_training'], identity=hid, directory=home,
                    heartbeat=lambda **kw: beat(head=name, **kw), resume=resume,
                    stop_at=cfg['pilot_updates'] if pilot else None)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt')))
                    return
                pred = parent.head.predict(model, c['x'], c['envelope'], pr)
                edges = score_edges(pred[pos], pr['weights'], cfg['score_quantiles'])
                temp = home/'scores.tmp.npz'; np.savez(temp, ids=c['ids'], scores=pred)
                os.replace(temp, home/'scores.npz')
                rec = dict(identity=hid, fit=fit, constant=pr['constant'].tolist(), cost_scale=pr['cost_scale'],
                    edges=edges, held_rows=len(held_ids), fitting_rows=len(fit_ids),
                    artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz')))
                assert fit['unknown_rows_sampled'] == 0
                immutable_json(done, rec); beat('head_complete', head=name, seconds=fit['seconds'])
            refs.append(artifact(done))
            if replay:
                state = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
                assert state['identity'] == hid and state['step'] == 2000
                for k in ('mean','std','constant','weights','known'):
                    np.testing.assert_array_equal(state['preprocess'][k], pr[k])
                assert state['preprocess']['cost_scale'] == pr['cost_scale']
                model = parent.head.initialize_head(cfg['head_training']['width'], pr, hid['seed'], 'all', state['mean_envelope'])
                model.load_state_dict(state['model'])
                with np.load(home/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], c['ids'])
                    np.testing.assert_array_equal(z['scores'], parent.head.predict(model, c['x'], c['envelope'], pr))
                    assert rec['edges'] == score_edges(z['scores'][pos], pr['weights'], cfg['score_quantiles'])
                checks.append(dict(head=name, exact=True))
    assert len(refs) == cfg['head_count']
    immutable_json(PUBLIC/'prediction_freeze.json', dict(identity=identity, heads=refs, count=len(refs),
        updates=2000*len(refs), held_labels_used=False, outer_readout_scored=False))
    if replay:
        immutable_json(PUBLIC/'prediction_replay.json', dict(exact=True, checks=checks,
            predictions=artifact(PUBLIC/'prediction_freeze.json')))


def evaluate(cfg, data, jobs, identity, replay=False):
    parent.committed(PUBLIC/'prediction_freeze.json')
    frozen = json.loads((PUBLIC/'prediction_freeze.json').read_text())
    assert frozen['identity'] == identity
    for ref in frozen['heads']: assert artifact(ROOT/ref['path']) == ref
    rows, records = [], []
    for c in contexts(data, jobs, identity):
        ids = c['ids']; sites = data['sites'][ids]
        y = targets(data, ids, c['prediction'])
        for held in sorted(set(sites)):
            name = parent.group_name(c['job'], c['controller'], c['candidate'])+'_held'+held
            home = PRIVATE/'heads'/name; rec = json.loads((home/'complete.json').read_text())
            assert rec['identity']['experiment'] == identity
            assert artifact(home/'scores.npz') == rec['artifacts']['scores']
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids); pred = z['scores'].copy()
            local = {str(s): summarize(pred[sites == s], y[sites == s], rec['constant'], rec['cost_scale'], rec['edges'])
                     for s in sorted(set(sites))}
            fit = [r for s, r in local.items() if s != held]
            row = dict(site=held, group=name, candidate=c['candidate'], seed=c['job']['old_identity']['seed'],
                metric=flatten_pair(local[held], fit))
            rows.append(row); records.append(dict(**row, localities=local, edges=rec['edges']))
        beat('scored_controller', trial=c['job']['key'], candidate=c['candidate'], controller=c['controller'])
    assert len(rows) == 144
    sites = sorted(set(data['sites']))
    def aggregate(candidate, seed=None):
        selected = [r for r in rows if r['candidate'] == candidate and (seed is None or r['seed'] == seed)]
        return {key: parent.paired_localities(selected, sites, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
                for key in rows[0]['metric']}
    summary = dict(identity=identity, result_source='fresh_crossfit_heads_cached_verified_forecaster_bank',
        by_candidate={a: aggregate(a) for a in cfg['candidates']},
        by_seed={a: {str(s): aggregate(a, s) for s in cfg['seeds']} for a in cfg['candidates']},
        rows=rows, heads=144, new_updates=288000, outer_readout_scored=False,
        independent_roles_read=False, policy_evaluation=False, deployment_changed=False)
    immutable_json(PRIVATE/'diagnostic_detail.json', records)
    immutable_json(PUBLIC/'summary.json', summary)
    if replay:
        immutable_json(PUBLIC/'evaluation_replay.json', dict(exact=True,
            summary=artifact(PUBLIC/'summary.json'), detail=artifact(PRIVATE/'diagnostic_detail.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['register','pilot','train','replay','evaluate','verify_eval'], required=True)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('phase_started', phase=args.phase)
        cfg, data, jobs, identity = load(create=args.phase == 'register')
        if args.phase in ('pilot','train','replay'):
            train(cfg, data, jobs, identity, resume=args.resume, pilot=args.phase == 'pilot', replay=args.phase == 'replay')
        elif args.phase in ('evaluate','verify_eval'):
            evaluate(cfg, data, jobs, identity, replay=args.phase == 'verify_eval')
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
