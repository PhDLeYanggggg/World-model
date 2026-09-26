"""Reconstruct the strong cost estimator before adding matched cap supervision."""
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
from scripts import run_m3w_european_cap_auxiliary_cost as parent
from scripts.evaluate_m3w_european_support_fractional import measure
from src.world_model import m3w_strong_cap_auxiliary as method
from src.world_model.m3w_easy_membership_probe import labels
from src.evaluation import m3w_harm_tail_diagnostics as tail
import numpy as np
import torch

cap, risk = parent.parent, parent.parent.risk
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_strong_cap_auxiliary_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_strong_cap_auxiliary_v1'
CONFIG = 'configs/m3w_european_strong_cap_auxiliary_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_strong_cap_auxiliary.py',
    'tests/test_m3w_strong_cap_auxiliary.py', 'scripts/run_m3w_european_strong_cap_auxiliary.py',
    'scripts/report_m3w_european_strong_cap_auxiliary.py', str(PUBLIC.relative_to(ROOT)/'protocol.md'),
    'src/world_model/m3w_membership_auxiliary.py', 'src/world_model/m3w_native_gain_harm.py',
    'src/world_model/m3w_easy_membership_probe.py', 'scripts/evaluate_m3w_european_support_fractional.py',
    'scripts/report_m3w_european_support_fractional.py', 'scripts/report_m3w_european_frozen_harm_readout.py',
    'scripts/plot_m3w_european_strong_cap_auxiliary.py', 'scripts/verify_m3w_european_strong_cap_auxiliary.py']
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    check = json.loads((parent.PUBLIC/'verification.json').read_text()); assert check['all_passed']
    for p, h in check['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in check['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['arms'] == list(method.ARMS) and (cfg['views'], cfg['new_heads'], cfg['updates']) == (144, 432, 864000)
    assert cfg['head_training'] == json.loads((ROOT/'configs/m3w_european_membership_auxiliary_v1.json').read_text())['head_training']
    assert cfg['auxiliary_coefficient'] == 1 and cfg['feature_dimension'] == 383
    assert not any(cfg[k] for k in ('new_forecaster_training', 'new_policy_evaluation', 'threshold_refit',
        'selection_access', 'reserved_calibration_access', 'confirmation_access', 'deployment_changed',
        'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; risk.base.previous.require_committed(path)
    return cfg, identity


def views(identity):
    return cap.views(identity['parent']['parent'])


def check_sources(identity):
    risk.check_sources(identity['parent']['parent']['source_parent'])
    for ref in json.loads((risk.PUBLIC/'support_report.json').read_text())['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values():
            assert artifact(ROOT/a['path']) == a


def inputs(v):
    bank, y, _ = risk.banks(v)
    inner, _, _, producer = risk.nested.method.assemble(bank, v['raw'], v['cv'], v['sites'], v['outer'], 'oof')
    event = cap.method.event_target(y, inner, v['env'])
    easy = labels(v['cv'], v['pr']['positive_easy_cut'])
    x = v['x']; bx, env, _, _ = v['pairs']['B'][v['pair']]; hx = bx[v['te']]
    directory = risk.context_run.frozen_directory(v['tag'], 'original')
    _, old = method.restore(directory)
    for key in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(v['pr'][key], old['preprocess'][key])
    for key in ('cost_scale', 'positive_easy_cut', 'hard_cut'):
        assert v['pr'][key] == old['preprocess'][key]
    prior = json.loads((directory/'complete.json').read_text())
    for key, value in (('train_x_sha256', x), ('held_x_sha256', hx), ('train_y_sha256', y), ('train_easy_sha256', easy)):
        assert prior['input'][key] == array_hash(value)
    with np.load(directory/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['held_ids']); frozen = z['scores'].copy()
    reference_path = risk.PRIVATE/'support'/v['tag']/'scores.npz'
    with np.load(reference_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['ids']); reference = z['scores'].copy()
    method.validate(x, y, easy, event, v['sites'], v['outer'], v['env'], v['pr'])
    known_event = np.isfinite(event); weights = v['pr']['weights']
    event_prior = float(weights[known_event]@event[known_event]/weights[known_event].sum())
    record = dict(tag=v['tag'], outer=v['outer'], training_sites=sorted(set(v['sites'])),
        training_ids_sha256=array_hash(v['ids']), held_ids_sha256=array_hash(v['held_ids']),
        training_features_sha256=array_hash(x), held_features_sha256=array_hash(hx),
        cost_target_sha256=array_hash(y), easy_sha256=array_hash(easy), cap_target_sha256=array_hash(event),
        row_producer_sha256=array_hash(producer), inner_score_sha256=array_hash(inner),
        original=artifact(directory/'complete.json'), fitting_reference=artifact(reference_path),
        feature_dimension=x.shape[1], event_prior=event_prior, cost_scale=v['pr']['cost_scale'],
        outer_targets_used_for_fitting=False, teacher_scores_used_as_inputs=False)
    return x, y, easy, event, hx, frozen, env[v['te']], reference, old, record


def support(cfg, identity):
    check_sources(identity); rows = []
    for v in views(identity):
        x, y, easy, event, _, _, _, _, _, record = inputs(v)
        assert x.shape[1] == cfg['feature_dimension']
        local = cap.diagnostic.support(event, v['sites'], v['data']['recordings'][v['ids']], v['data']['agents'][v['ids']])
        rows.append(dict(tag=v['tag'], pair=v['pair'], input=record, localities=local,
            known_cost_rows=int(v['pr']['known'].sum()), event_rows=int(np.isfinite(event).sum()),
            positive_event_rows=int(np.nansum(event)), known_zero_envelope_rows=int((v['pr']['known'] & (v['env'] == 0)).sum())))
        beat('support', views=len(rows), tag=v['tag'])
    assert len(rows) == 144
    immutable_json(PUBLIC/'support_report.json', dict(rows=rows, training_allowed=True,
        registration=artifact(PUBLIC/'registration_lock.json'), statistical_power_established=False,
        all_original_input_target_preprocessing_hashes_match=True, held_labels_used_for_fitting=False))


def match_state(a, b):
    for key in ('draws', 'fixed_ids', 'loss_scales'):
        np.testing.assert_array_equal(a[key], b[key])
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    for key in ('mean', 'std', 'known', 'weights'):
        np.testing.assert_array_equal(a['preprocess'][key], b['preprocess'][key])


def compare_control(model, old, score, frozen, cfg):
    errors = []
    for key, value in model.state_dict().items():
        np.testing.assert_allclose(value.numpy(), old['model'][key].numpy(), rtol=cfg['control_rtol'], atol=cfg['control_atol'])
        errors.append(float(torch.max(torch.abs(value-old['model'][key]))))
    np.testing.assert_allclose(score, frozen, rtol=cfg['control_rtol'], atol=cfg['control_atol'])
    return dict(state_max_abs_error=max(errors), held_score_max_abs_error=float(np.max(np.abs(score-frozen))),
                exact=all(e == 0 for e in errors) and np.array_equal(score, frozen), tolerance_pass=True)


def training(cfg, identity, pilot=False, resume=False, verify=False):
    support_doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert support_doc['training_allowed'] and support_doc['registration'] == artifact(PUBLIC/'registration_lock.json')
    risk.base.previous.require_committed(PUBLIC/'support_report.json'); check_sources(identity)
    records = {r['tag']: r['input'] for r in support_doc['rows']}; refs, controls = [], []
    matched = 0; started = time.monotonic()
    for v in views(identity):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; retained checkpoints support resume')
        x, y, easy, event, hx, frozen, henv, reference, old, record = inputs(v)
        assert records[v['tag']] == record
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0]); states = []
        for arm in (['cost_only'] if pilot else cfg['arms']):
            home = PRIVATE/'heads'/v['tag']/arm; path = home/'complete.json'
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record, arm=arm, seed=seed)
            if path.exists():
                receipt = json.loads(path.read_text()); assert receipt['identity'] == hid
                for a in receipt['artifacts'].values(): assert artifact(ROOT/a['path']) == a
                model, state = method.restore(home)
                fit = receipt['fit']
            else:
                if verify: raise ValueError('Missing completed head')
                model, fit = method.fit(x, y, easy, event, v['sites'], v['outer'], v['env'], v['pr'],
                    arm=arm, seed=seed, settings=cfg['head_training'], identity=hid, directory=home,
                    resume=resume, stop_at=100 if pilot else None,
                    heartbeat=lambda **kw: beat(tag=v['tag'], arm=arm, **kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                        projected_fit_seconds=fit['seconds']/100*cfg['updates'], wall_seconds=time.monotonic()-started,
                        excludes_preflight=True, auxiliary_runtime_not_yet_measured=True)); return
                _, state = method.restore(home)
            assert state['step'] == 2000 and state['settings'] == cfg['head_training'] and state['identity'] == hid
            match_state(state, old)
            np.testing.assert_array_equal(state['auxiliary_target'], method.auxiliary_target(event, v['sites'], arm, seed))
            assert not state['draws'][~v['pr']['known']].any()
            native, probability = method.predict(model, hx, henv, v['pr'])
            score = parent.compose(native[:, [1, 3]], frozen)
            if arm == 'cost_only':
                control = compare_control(model, old, score, frozen, cfg); controls.append(dict(tag=v['tag'], **control))
            else: control = None
            raw_fit, _ = method.predict(model, x, v['env'], v['pr'])
            fitting = parent.compose(raw_fit[:, [1, 3]], reference)
            edges = parent.edges_for(fitting, v['env'], v['pr'], cfg)
            diag = dict(edges=edges, training=measure(fitting, y, v['env'], v['sites'], edges))
            if path.exists():
                with np.load(home/'scores.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], v['held_ids']); np.testing.assert_array_equal(z['scores'], score)
                    np.testing.assert_array_equal(z['probability'], probability)
                assert json.loads((home/'fit_diagnosis.json').read_text()) == diag
                assert receipt['control_reconstruction'] == control
            else:
                risk.base.previous.parent.atomic_npz(home/'scores.npz', ids=v['held_ids'], scores=score, probability=probability)
                immutable_json(home/'fit_diagnosis.json', diag)
                replay, rp = method.predict(method.restore(home)[0], hx, henv, v['pr'])
                np.testing.assert_array_equal(replay, native); np.testing.assert_array_equal(rp, probability)
                immutable_json(path, dict(identity=hid, fit=fit, control_reconstruction=control,
                    fitting_event_prior=record['event_prior'], result_source='fresh_run_native_torch',
                    artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz'),
                                   diagnosis=artifact(home/'fit_diagnosis.json'))))
            states.append(state); refs.append(artifact(path))
            beat('head_replayed' if verify else 'head_frozen', completed=len(refs), tag=v['tag'], arm=arm)
        for s in states[1:]:
            match_state(states[0], s)
            for key in s['initial_model']: assert torch.equal(s['initial_model'][key], states[0]['initial_model'][key])
            matched += 1
    assert len(refs) == 432 and len(controls) == 144 and matched == 288
    immutable_json(PUBLIC/('training_replay.json' if verify else 'prediction_freeze.json'),
        dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs, controls=controls,
             heads=432, updates=864000, matched_arm_checks=matched, held_labels_used_for_fitting=False))
    beat('training_replay_complete' if verify else 'training_complete', elapsed_seconds=time.monotonic()-started)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; risk.base.previous.require_committed(path)
    doc = json.loads(path.read_text()); assert len(doc['receipts']) == 432 and len(doc['controls']) == 144
    assert all(r['tolerance_pass'] for r in doc['controls'])
    for ref in doc['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a


def evaluate(cfg, identity, verify=False):
    check_freeze(); groups, cached, direct = {}, {}, 0
    old_groups = {r['group']+'_'+r['pair']: r for r in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']}
    for v in views(identity):
        _, _, _, _, _, frozen, env, _, _, record = inputs(v)
        name = v['g']['group']+'_'+v['pair']
        if name not in cached: cached = {name: risk.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        y = tail.event_targets(cached[name][v['te']], v['data']['baseline_ade'][v['held_ids'], 1], v['pr']['positive_easy_cut'])
        old = next(f for f in old_groups[name]['folds'] if f['held'] == v['outer'])
        assert old['target_sha256'] == array_hash(y) and old['held_ids_sha256'] == array_hash(v['held_ids'])
        event = cap.method.event_target(y, frozen, env); excess = np.maximum(y[:, 3]-frozen[:, 1], 0)
        metrics, fitting, events = {}, {}, {}
        for arm in ['original', *cfg['arms']]:
            home = risk.context_run.frozen_directory(v['tag'], 'original') if arm == 'original' else PRIVATE/'heads'/v['tag']/arm
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids']); score = z['scores'].copy()
                if arm != 'original': events[arm] = cap.diagnostic.measures(event, z['probability'], excess, record['event_prior'])
            np.testing.assert_array_equal(score[:, [0, 2]], frozen[:, [0, 2]])
            fit = json.loads((home/'fit_diagnosis.json').read_text()); fitting[arm] = fit
            metrics[arm] = measure(score, y, env, v['data']['sites'][v['held_ids']], fit['edges'])
            for subset, mask in [('all', np.ones(len(y), bool)), ('envelope_positive', env > 0)]:
                known = np.isfinite(y).all(1) & mask
                np.testing.assert_allclose(metrics[arm][subset]['component_MSE'], ((score[known]-y[known])**2).mean(0), rtol=1e-13, atol=1e-13)
                direct += 1
        group = groups.setdefault(name, dict(group=v['g']['group'], producer=v['g']['producer'], controller=v['g']['controller'],
            seed=int(v['g']['group'].split('_seed')[1].split('_')[0]), pair=v['pair'], folds=[]))
        group['folds'].append(dict(held=v['outer'], metrics=metrics, training=fitting, event_metrics=events,
                                  target_sha256=array_hash(y), held_ids_sha256=array_hash(v['held_ids'])))
        beat('readout_replayed' if verify else 'readout', views=sum(len(r['folds']) for r in groups.values()), tag=v['tag'])
    assert len(groups) == 36 and direct == 1152
    doc = dict(rows=list(groups.values()), direct_component_MSE_checks=direct, result_source='fresh_run_source_development',
        independent_selection_read=False, reserved_calibration_read=False, confirmation_read=False, new_policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'readout_replay.json', dict(groups=36, views=144, exact=True, direct_MSE_checks=direct))
    else: immutable_json(PUBLIC/'readout.json', doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['register', 'support', 'pilot', 'train', 'evaluate', 'verify_training', 'verify_eval'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, identity = registration(args.phase == 'register')
        if args.phase == 'support': support(cfg, identity)
        elif args.phase in ('pilot', 'train', 'verify_training'):
            training(cfg, identity, args.phase == 'pilot', args.resume, args.phase == 'verify_training')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, args.phase == 'verify_eval')


if __name__ == '__main__': main()
