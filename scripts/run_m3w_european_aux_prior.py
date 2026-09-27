"""Single-factor prior repair; freeze all predictions before source-held readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_strong_cap_auxiliary as parent
from src.world_model import m3w_aux_prior as method
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_aux_prior_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_aux_prior_v1'
CONFIG = 'configs/m3w_european_aux_prior_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_aux_prior.py', 'tests/test_m3w_aux_prior.py',
    'scripts/run_m3w_european_aux_prior.py', 'scripts/report_m3w_european_aux_prior.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def checked_seal(path):
    doc = json.loads(path.read_text()); assert doc['all_passed']
    for name, h in doc['artifacts'].items(): assert digest(path.parent/name) == h
    for name, h in doc['source_bindings'].items(): assert digest(ROOT/name) == h
    return artifact(path)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); old_cfg, old_identity = parent.registration()
    seals = [checked_seal(parent.PUBLIC/'verification.json'), checked_seal(
        ROOT/'outputs/publication_readiness_2026_09/european_aux_trajectory_v1/verification.json')]
    assert cfg['head_training'] == old_cfg['head_training']
    assert (cfg['views'], cfg['new_heads'], cfg['updates']) == (144, 288, 576000)
    assert cfg['arms'] == ['cap_aux', 'shuffled_aux'] and cfg['initialization'] == 'fitting_cap'
    for key, value in old_cfg.items():
        if value is False: assert cfg[key] is False
    identity = dict(parent=old_identity, seals=seals, bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        parent.risk.base.previous.require_committed(path)
    return cfg, identity


def training(cfg, identity, pilot=False, resume=False, verify=False):
    parent.check_sources(identity['parent'])
    source = {r['tag']:r['input'] for r in json.loads((parent.PUBLIC/'support_report.json').read_text())['rows']}
    receipts, checks = [], []; started = time.monotonic()
    for v in parent.views(identity['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve 10GiB; resume checkpoints retained')
        x, y, easy, event, hx, frozen, henv, reference, old, record = parent.inputs(v)
        assert source[v['tag']] == record
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0]); states = []
        for arm in (['cap_aux'] if pilot else cfg['arms']):
            home = PRIVATE/'heads'/v['tag']/arm; path = home/'complete.json'
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record, arm=arm, seed=seed)
            original_home = parent.PRIVATE/'heads'/v['tag']/arm
            original_receipt = json.loads((original_home/'complete.json').read_text())
            for a in original_receipt['artifacts'].values(): assert artifact(ROOT/a['path']) == a
            _, original = parent.method.restore(original_home)
            if path.exists():
                receipt = json.loads(path.read_text()); assert receipt['identity'] == hid
                for a in receipt['artifacts'].values(): assert artifact(ROOT/a['path']) == a
                model, state = method.restore(home); fit = receipt['fit']
            else:
                if verify: raise ValueError('Missing completed head')
                model, fit = method.fit(x, y, easy, event, v['sites'], v['outer'], v['env'], v['pr'],
                    arm=arm, seed=seed, settings=cfg['head_training'], identity=hid, directory=home,
                    initialization=cfg['initialization'], resume=resume, stop_at=200 if pilot else None,
                    heartbeat=lambda **kw: beat(tag=v['tag'], arm=arm, **kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                        projected_fit_seconds=fit['seconds']/200*cfg['updates'],
                        wall_seconds=time.monotonic()-started, excludes_ancestry_preflight=True)); return
                _, state = method.restore(home)
            assert state['step'] == 2000 and state['identity'] == hid and state['settings'] == cfg['head_training']
            assert state['initialization'] == 'fitting_cap'
            assert state['initial_prior'] == record['event_prior']
            parent.match_state(state, original)
            for key in state['initial_model']:
                if key != 'membership.bias': assert torch.equal(state['initial_model'][key], original['initial_model'][key])
            np.testing.assert_array_equal(state['auxiliary_target'], original['auxiliary_target'])
            assert not state['draws'][~v['pr']['known']].any()
            native, probability = method.predict(model, hx, henv, v['pr'])
            score = parent.parent.compose(native[:, [1, 3]], frozen)
            raw_fit, _ = method.predict(model, x, v['env'], v['pr'])
            fitting = parent.parent.compose(raw_fit[:, [1, 3]], reference)
            edges = parent.parent.edges_for(fitting, v['env'], v['pr'], cfg)
            diag = dict(edges=edges, training=parent.measure(fitting, y, v['env'], v['sites'], edges))
            if path.exists():
                with np.load(home/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], v['held_ids']); np.testing.assert_array_equal(z['scores'], score)
                    np.testing.assert_array_equal(z['probability'], probability)
                assert json.loads((home/'fit_diagnosis.json').read_text()) == diag
            else:
                parent.risk.base.previous.parent.atomic_npz(home/'scores.npz', ids=v['held_ids'], scores=score, probability=probability)
                immutable_json(home/'fit_diagnosis.json', diag)
                rp, pp = method.predict(method.restore(home)[0], hx, henv, v['pr'])
                np.testing.assert_array_equal(rp, native); np.testing.assert_array_equal(pp, probability)
                immutable_json(path, dict(identity=hid, fit=fit, result_source='fresh_run_native_torch',
                    original_receipt=artifact(original_home/'complete.json'),
                    artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz'),
                        diagnosis=artifact(home/'fit_diagnosis.json'))))
            states.append(state); receipts.append(artifact(path))
            checks.append(dict(tag=v['tag'], arm=arm, initial_prior=state['initial_prior'],
                inherited_prior=original['prevalence'], only_auxiliary_intercept_changed=True,
                matched_draws_and_targets=True))
            beat('head_replayed' if verify else 'head_frozen', completed=len(receipts), tag=v['tag'], arm=arm)
        for key in states[0]['initial_model']: assert torch.equal(states[0]['initial_model'][key], states[1]['initial_model'][key])
    assert len(receipts) == 288
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=receipts, checks=checks,
        heads=288, updates=576000, held_labels_used_for_fitting=False, checkpoints_selected=False)
    if verify:
        assert json.loads((PUBLIC/'prediction_freeze.json').read_text()) == doc
        immutable_json(PUBLIC/'training_replay.json', dict(exact=True, heads=288, prediction_checks=288, initialization_checks=288))
    else: immutable_json(PUBLIC/'prediction_freeze.json', doc)
    beat('training_replay_complete' if verify else 'training_complete', elapsed_seconds=time.monotonic()-started)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; parent.risk.base.previous.require_committed(path)
    doc = json.loads(path.read_text()); assert doc['heads'] == 288 and len(doc['receipts']) == 288
    for ref in doc['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a


def evaluate(cfg, identity, verify=False):
    check_freeze(); groups, cached, direct = {}, {}, 0
    original_groups = {r['group']+'_'+r['pair']:r for r in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']}
    for v in parent.views(identity['parent']):
        _, _, _, _, _, frozen, env, _, _, record = parent.inputs(v)
        name = v['g']['group']+'_'+v['pair']
        if name not in cached: cached = {name:parent.risk.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        y = parent.tail.event_targets(cached[name][v['te']], v['data']['baseline_ade'][v['held_ids'], 1], v['pr']['positive_easy_cut'])
        old = next(f for f in original_groups[name]['folds'] if f['held'] == v['outer'])
        assert old['target_sha256'] == array_hash(y) and old['held_ids_sha256'] == array_hash(v['held_ids'])
        event = parent.cap.method.event_target(y, frozen, env); excess = np.maximum(y[:, 3]-frozen[:, 1], 0)
        metrics, fitting, events = {}, {}, {}
        arms = [('cost_only', parent.PRIVATE, 'cost_only'), ('old_true', parent.PRIVATE, 'cap_aux'),
                ('old_shuffled', parent.PRIVATE, 'shuffled_aux'), ('new_true', PRIVATE, 'cap_aux'),
                ('new_shuffled', PRIVATE, 'shuffled_aux')]
        for name_arm, base, arm in arms:
            home = base/'heads'/v['tag']/arm
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids']); score = z['scores'].copy()
                events[name_arm] = parent.cap.diagnostic.measures(event, z['probability'], excess, record['event_prior'])
            np.testing.assert_array_equal(score[:, [0, 2]], frozen[:, [0, 2]])
            fit = json.loads((home/'fit_diagnosis.json').read_text()); fitting[name_arm] = fit
            metrics[name_arm] = parent.measure(score, y, env, v['data']['sites'][v['held_ids']], fit['edges'])
            if base == parent.PRIVATE: assert metrics[name_arm] == old['metrics'][arm]
            for subset, mask in [('all', np.ones(len(y), bool)), ('envelope_positive', env > 0)]:
                known = np.isfinite(y).all(1) & mask
                np.testing.assert_allclose(metrics[name_arm][subset]['component_MSE'],
                    ((score[known]-y[known])**2).mean(0), rtol=1e-13, atol=1e-13); direct += 1
        group = groups.setdefault(name, dict(group=v['g']['group'], producer=v['g']['producer'], controller=v['g']['controller'],
            seed=int(v['g']['group'].split('_seed')[1].split('_')[0]), pair=v['pair'], folds=[]))
        group['folds'].append(dict(held=v['outer'], metrics=metrics, training=fitting, event_metrics=events,
            target_sha256=array_hash(y), held_ids_sha256=array_hash(v['held_ids'])))
        beat('readout_replayed' if verify else 'readout', views=sum(len(r['folds']) for r in groups.values()), tag=v['tag'])
    assert len(groups) == 36 and direct == 1440
    doc = dict(rows=list(groups.values()), direct_MSE_checks=direct, result_source='fresh_run_exposed_source_development',
        old_controls='cached_verified_exact_metric_replay', independent_selection_read=False,
        reserved_calibration_read=False, confirmation_read=False, new_policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'readout_replay.json', dict(exact=True, views=144, direct_MSE_checks=direct))
    else: immutable_json(PUBLIC/'readout.json', doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['register', 'pilot', 'train', 'evaluate', 'verify_training', 'verify_eval'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, identity = registration(args.phase == 'register')
        if args.phase in ('pilot', 'train', 'verify_training'):
            training(cfg, identity, args.phase == 'pilot', args.resume, args.phase == 'verify_training')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, args.phase == 'verify_eval')


if __name__ == '__main__': main()
