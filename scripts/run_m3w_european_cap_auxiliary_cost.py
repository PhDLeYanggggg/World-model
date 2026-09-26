"""Matched cap-event auxiliary cost learning with frozen source lineage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 is required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
from scripts import run_m3w_european_cap_exceedance as parent
from scripts.evaluate_m3w_european_support_fractional import measure
from src.world_model import m3w_cap_auxiliary_cost as method
from src.evaluation import m3w_harm_tail_diagnostics as tail
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cap_auxiliary_cost_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cap_auxiliary_cost_v1'
CONFIG = 'configs/m3w_european_cap_auxiliary_cost_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_cap_auxiliary_cost.py', 'tests/test_m3w_cap_auxiliary_cost.py',
    'scripts/run_m3w_european_cap_auxiliary_cost.py', 'scripts/report_m3w_european_cap_auxiliary_cost.py',
    'tests/test_m3w_cap_auxiliary_cost_reporting.py', str(PUBLIC.relative_to(ROOT)/'protocol.md'),
    'scripts/evaluate_m3w_european_support_fractional.py',
    'scripts/report_m3w_european_support_fractional.py',
    'scripts/report_m3w_european_frozen_harm_readout.py']
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    check = json.loads((parent.PUBLIC/'verification.json').read_text()); assert check['all_passed']
    for p, h in check['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in check['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['arms'] == list(method.ARMS) and (cfg['views'], cfg['new_heads'], cfg['updates']) == (144, 432, 864000)
    assert cfg['head_training'] == json.loads((ROOT/parent.CONFIG).read_text())['head_training']
    assert cfg['auxiliary_coefficient'] == 1
    assert not any(cfg[k] for k in ('new_forecaster_training', 'new_policy_evaluation', 'threshold_refit',
        'selection_access', 'reserved_calibration_access', 'confirmation_access', 'deployment_changed',
        'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create:
        immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        parent.risk.base.previous.require_committed(path)
    return cfg, identity


def inputs(v):
    x, event, hx, held, henv, record = parent.inputs(v)
    y = v['outer_y']
    record = dict(record, cost_target_sha256=array_hash(y), cost_scale=v['pr']['cost_scale'],
                  target_event_not_in_inputs=True, cap_classifier_predictions_not_stacked=True)
    return x, y, event, hx, held, henv, record


def prepare(v, x, y, event):
    return method.prepare(x, y, event, v['sites'], v['outer'], v['env'], v['pr']['cost_scale'])


def support(cfg, identity):
    parent.risk.check_sources(identity['parent']['source_parent']); rows = []
    for v in parent.views(identity['parent']):
        x, y, event, _, _, _, record = inputs(v)
        pr = prepare(v, x, y, event)
        local = parent.diagnostic.support(event, v['sites'], v['data']['recordings'][v['ids']], v['data']['agents'][v['ids']])
        assert len(set(v['sites'])) == 3 and v['outer'] not in v['sites']
        rows.append(dict(tag=v['tag'], pair=v['pair'], input=record, localities=local,
            prevalence=pr['prevalence'], known=int(pr['known'].sum()), positive=int(np.nansum(event)),
            loss_scales=pr['loss_scales'].tolist(), cost_means=pr['cost_means'].tolist(), numerically_supported=True))
        beat('support', views=len(rows), tag=v['tag'])
    assert len(rows) == cfg['views']
    immutable_json(PUBLIC/'support_report.json', dict(registration=artifact(PUBLIC/'registration_lock.json'),
        rows=rows, training_allowed=True, statistical_power_established=False, held_labels_used_for_fitting=False))


def compose(harm, frozen):
    out = frozen.copy(); out[:, [1, 3]] = harm
    np.testing.assert_array_equal(out[:, [0, 2]], frozen[:, [0, 2]])
    return out


def training_reference(v):
    path = parent.risk.PRIVATE/'support'/v['tag']/'scores.npz'
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['ids'])
        return z['scores'].copy()


def edges_for(score, env, pr, cfg):
    return {k: tail.quantiles(s, pr['weights'], cfg['score_bin_quantiles'])
            for k, s in tail.score_columns(score, env).items()}


def assert_matching(a, b):
    for key in ('draws', 'fixed'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in a['initial_model']:
        assert torch.equal(a['initial_model'][key], b['initial_model'][key])
    assert torch.equal(a['sampler_rng'], b['sampler_rng'])
    for key in ('mean', 'std', 'weights', 'known', 'loss_scales', 'cost_means'):
        np.testing.assert_array_equal(a['preprocess'][key], b['preprocess'][key])


def training(cfg, identity, *, pilot=False, resume=False, verify=False):
    support_doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert support_doc['training_allowed'] and support_doc['registration'] == artifact(PUBLIC/'registration_lock.json')
    parent.risk.base.previous.require_committed(PUBLIC/'support_report.json')
    parent.risk.check_sources(identity['parent']['source_parent'])
    refs, matched, started = [], 0, time.monotonic()
    records = {r['tag']: r['input'] for r in support_doc['rows']}
    for v in parent.views(identity['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30:
            raise OSError('Preserve10GiB disk; retained checkpoints can resume')
        x, y, event, hx, frozen, henv, record = inputs(v)
        assert records[v['tag']] == record
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0])
        reference = training_reference(v); states = []
        for arm in (['cost_only'] if pilot else cfg['arms']):
            home = PRIVATE/'heads'/v['tag']/arm; path = home/'complete.json'
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record, arm=arm, seed=seed)
            if path.exists():
                receipt = json.loads(path.read_text()); assert receipt['identity'] == hid
                for ref in receipt['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
                model, state = method.restore(home)
                if verify:
                    assert state['identity'] == hid and state['step'] == cfg['head_training']['steps']
                    assert state['settings'] == cfg['head_training'] and state['seed'] == seed and state['arm'] == arm
                    pr = prepare(v, x, y, event)
                    for key in ('mean', 'std', 'known', 'weights', 'loss_scales', 'cost_means'):
                        np.testing.assert_array_equal(pr[key], state['preprocess'][key])
                    for key in ('cost_scale', 'prevalence', 'mean_envelope'):
                        assert pr[key] == state['preprocess'][key]
                    np.testing.assert_array_equal(state['auxiliary_target'], method.auxiliary_target(event, v['sites'], arm, seed))
                    assert not state['draws'][~pr['known']].any()
                    harm, probability = method.predict(model, hx, henv, pr)
                    with np.load(home/'scores.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], v['held_ids'])
                        np.testing.assert_array_equal(z['scores'], compose(harm, frozen))
                        np.testing.assert_array_equal(z['probability'], probability)
                    harm, _ = method.predict(model, x, v['env'], pr)
                    score = compose(harm, reference); edges = edges_for(score, v['env'], pr, cfg)
                    assert json.loads((home/'fit_diagnosis.json').read_text()) == dict(edges=edges,
                        training=measure(score, y, v['env'], v['sites'], edges))
            else:
                if verify: raise ValueError('Missing completed head')
                model, pr, fit = method.fit(x, y, event, v['sites'], v['outer'], v['env'], v['pr']['cost_scale'],
                    arm=arm, seed=seed, settings=cfg['head_training'], identity=hid, directory=home,
                    resume=resume, stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(tag=v['tag'], **kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                        estimated_cost_only_rate_for_total_updates_seconds=fit['seconds']/100*cfg['updates'],
                        auxiliary_fit_speed_not_yet_measured=True, wall_seconds=time.monotonic()-started,
                        excludes_ancestry_preflight=True))
                    return
                harm, probability = method.predict(model, hx, henv, pr)
                parent.risk.base.previous.parent.atomic_npz(home/'scores.npz', ids=v['held_ids'],
                                                           scores=compose(harm, frozen), probability=probability)
                train_harm, _ = method.predict(model, x, v['env'], pr)
                score = compose(train_harm, reference); edges = edges_for(score, v['env'], pr, cfg)
                immutable_json(home/'fit_diagnosis.json', dict(edges=edges, training=measure(score, y, v['env'], v['sites'], edges)))
                restored, state = method.restore(home)
                a, b = method.predict(restored, hx, henv, state['preprocess'])
                np.testing.assert_array_equal(a, harm); np.testing.assert_array_equal(b, probability)
                immutable_json(path, dict(identity=hid, fit=fit, fitting_prior=pr['prevalence'],
                    result_source='fresh_run_native_torch', artifacts=dict(checkpoint=artifact(home/'checkpoint.pt'),
                        scores=artifact(home/'scores.npz'), diagnosis=artifact(home/'fit_diagnosis.json'))))
            states.append(state); refs.append(artifact(path))
            beat('head_replayed' if verify else 'head_frozen', completed=len(refs), tag=v['tag'], arm=arm)
        for state in states[1:]:
            assert_matching(states[0], state); matched += 1
    assert len(refs) == cfg['new_heads'] and matched == 288
    immutable_json(PUBLIC/('training_replay.json' if verify else 'prediction_freeze.json'),
        dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs, heads=len(refs),
             updates=cfg['updates'], matched_arm_checks=matched, held_labels_used_for_fitting=False))
    beat('training_replay_complete' if verify else 'training_complete', elapsed_seconds=time.monotonic()-started)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; parent.risk.base.previous.require_committed(path)
    doc = json.loads(path.read_text()); assert len(doc['receipts']) == 432
    for ref in doc['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a


def evaluate(cfg, identity, verify=False):
    check_freeze(); groups, cached, direct = {}, {}, 0
    for v in parent.views(identity['parent']):
        _, _, _, _, frozen, env, _ = inputs(v)
        name = v['g']['group']+'_'+v['pair']
        if name not in cached:
            cached = {name: parent.risk.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        cv = v['data']['baseline_ade'][v['held_ids'], 1]
        y = tail.event_targets(cached[name][v['te']], cv, v['pr']['positive_easy_cut'])
        prior = json.loads((parent.risk.PUBLIC/'groups'/(name+'.json')).read_text())
        old = next(f for f in prior['folds'] if f['held'] == v['outer'])
        assert old['target_sha256'] == array_hash(y)
        event = parent.method.event_target(y, frozen, env)
        overshoot = np.maximum(y[:, 3]-frozen[:, 1], 0)
        metrics, fitting, events, sources = {}, {}, {}, {}
        for arm in ['original', *cfg['arms']]:
            home = (parent.risk.context_run.frozen_directory(v['tag'], 'original') if arm == 'original'
                    else PRIVATE/'heads'/v['tag']/arm)
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids']); score = z['scores'].copy()
                if arm != 'original':
                    r = json.loads((home/'complete.json').read_text())
                    events[arm] = parent.diagnostic.measures(event, z['probability'], overshoot, r['fitting_prior'])
            np.testing.assert_array_equal(score[:, [0, 2]], frozen[:, [0, 2]])
            fit = json.loads((home/'fit_diagnosis.json').read_text())
            metrics[arm] = measure(score, y, env, v['data']['sites'][v['held_ids']], fit['edges'])
            fitting[arm] = fit; sources[arm] = artifact(home/'complete.json')
            for subset, mask in [('all', np.ones(len(y), bool)), ('envelope_positive', env > 0)]:
                known = np.isfinite(y).all(1) & mask
                np.testing.assert_allclose(metrics[arm][subset]['component_MSE'],
                    ((score[known]-y[known])**2).mean(0), rtol=1e-13, atol=1e-13); direct += 1
        group = groups.setdefault(name, dict(group=v['g']['group'], producer=v['g']['producer'],
            controller=v['g']['controller'], seed=int(v['g']['group'].split('_seed')[1].split('_')[0]),
            pair=v['pair'], folds=[], result_source='fresh_run_source_development_cost_readout'))
        group['folds'].append(dict(held=v['outer'], metrics=metrics, training=fitting, event_metrics=events,
            heads=sources, target_sha256=array_hash(y), held_ids_sha256=array_hash(v['held_ids'])))
        beat('readout_replayed' if verify else 'readout', views=sum(len(r['folds']) for r in groups.values()), tag=v['tag'])
    assert len(groups) == 36 and direct == 1152
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), rows=list(groups.values()),
        direct_component_MSE_checks=direct, independent_selection_read=False, reserved_calibration_read=False,
        confirmation_read=False, new_policy_evaluated=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'readout_replay.json', dict(groups=36, views=144, exact=True,
            direct_component_MSE_checks=direct, readout=artifact(PUBLIC/'readout.json')))
    else:
        immutable_json(PUBLIC/'readout.json', doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['register', 'support', 'pilot', 'train',
                                                         'evaluate', 'verify_training', 'verify_eval'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, identity = registration(args.phase == 'register')
        if args.phase == 'support': support(cfg, identity)
        elif args.phase in ['pilot', 'train', 'verify_training']:
            training(cfg, identity, pilot=args.phase == 'pilot', resume=args.resume, verify=args.phase == 'verify_training')
        elif args.phase in ['evaluate', 'verify_eval']: evaluate(cfg, identity, verify=args.phase == 'verify_eval')


if __name__ == '__main__': main()
