"""Train and freeze causal cap-event probes before source-held readout."""
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
from scripts import run_m3w_european_fixed_cap_diagnostic as parent
from src.world_model import m3w_cap_exceedance as method
from src.evaluation import m3w_cap_exceedance as diagnostic
import numpy as np
import torch

risk = parent.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cap_exceedance_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_cap_exceedance_v1'
CONFIG = 'configs/m3w_european_cap_exceedance_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_cap_exceedance.py', 'src/evaluation/m3w_cap_exceedance.py',
         'tests/test_m3w_cap_exceedance.py', 'scripts/run_m3w_european_cap_exceedance.py',
         str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    risk.base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    check = json.loads((parent.PUBLIC/'verification.json').read_text()); assert check['all_passed']
    for path, h in check['artifacts'].items():
        assert digest(parent.PUBLIC/path) == h
    for path, h in check['source_bindings'].items():
        assert digest(ROOT/path) == h
    assert cfg['arms'] == list(method.ARMS) and (cfg['views'], cfg['new_heads'], cfg['updates']) == (144, 288, 576000)
    assert not any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'selection_access',
                                  'reserved_calibration_access', 'confirmation_access',
                                  'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, source_parent=pid['parent'], parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={f: digest(ROOT/f) for f in FILES})
    path = PUBLIC/'registration_lock.json'
    if create:
        immutable_json(path, identity)
    else:
        registered = json.loads(path.read_text())
        amended = PUBLIC/'implementation_amendment.json'
        if amended.exists():
            change = json.loads(amended.read_text())
            assert change['registration_sha256'] == digest(path) and change['scientific_changes'] is False
            for file, hashes in change['bindings'].items():
                assert registered['bindings'][file] == hashes['old_sha256']
                registered['bindings'][file] = hashes['new_sha256']
            risk.base.previous.require_committed(amended)
        assert registered == identity
        risk.base.previous.require_committed(path)
    return cfg, identity


def views(identity):
    return risk.nested.views(identity['source_parent']['parent']['parent'])


def inputs(v):
    bank, y, _ = risk.banks(v)
    inner, _, _, producer = risk.nested.method.assemble(bank, v['raw'], v['cv'], v['sites'], v['outer'], 'oof')
    target = method.event_target(y, inner, v['env'])
    x = method.causal_features(v['x'], v['context'], inner, v['env'])
    directory = risk.context_run.frozen_directory(v['tag'], 'original')
    with np.load(directory/'scores.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], v['held_ids']); held = z['scores'].copy()
    bx, env, _, _ = v['pairs']['B'][v['pair']]
    hx = method.causal_features(bx[v['te']], v['held_context'], held, env[v['te']])
    record = dict(tag=v['tag'], outer=v['outer'], training_sites=sorted(set(v['sites'])),
                  training_ids_sha256=array_hash(v['ids']), held_ids_sha256=array_hash(v['held_ids']),
                  training_features_sha256=array_hash(x), held_features_sha256=array_hash(hx),
                  target_sha256=array_hash(target), inner_score_sha256=array_hash(inner),
                  row_producer_sha256=array_hash(producer), outer_scores=artifact(directory/'scores.npz'),
                  easy_cut=v['pr']['positive_easy_cut'], held_outcomes_used_for_fitting=False,
                  feature_dimension=x.shape[1])
    return x, target, hx, held, env[v['te']], record


def support(cfg, identity):
    risk.check_sources(identity['source_parent']); rows = []
    for v in views(identity):
        x, target, _, _, _, record = inputs(v)
        pr = method.preprocess(x, target, v['sites'], v['outer'])
        local = diagnostic.support(target, v['sites'], v['data']['recordings'][v['ids']], v['data']['agents'][v['ids']])
        rows.append(dict(tag=v['tag'], pair=v['pair'], input=record, localities=local,
                         prevalence=pr['prevalence'], known=int(pr['known'].sum()),
                         positive=int(np.nansum(target)), numerically_supported=True))
        beat('support', views=len(rows), tag=v['tag'])
    assert len(rows) == cfg['views']
    immutable_json(PUBLIC/'support_report.json', dict(rows=rows, training_allowed=True,
                   registration=artifact(PUBLIC/'registration_lock.json'),
                   outer_held_outcomes_read=False, power_established=False))


def training(cfg, identity, *, pilot=False, resume=False, verify=False):
    doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert doc['training_allowed'] and doc['registration'] == artifact(PUBLIC/'registration_lock.json')
    risk.base.previous.require_committed(PUBLIC/'support_report.json')
    risk.check_sources(identity['source_parent'])
    supporting = {r['tag']: r for r in doc['rows']}; refs = []; started = time.monotonic()
    for v in views(identity):
        if shutil.disk_usage(PRIVATE).free < 10*2**30:
            raise OSError('Preserve10GiB disk reserve; checkpoint resume available')
        x, target, hx, _, _, record = inputs(v)
        assert record == supporting[v['tag']]['input']
        seed = int(v['g']['group'].split('_seed')[1].split('_')[0])
        for arm in (['mlp'] if pilot else cfg['arms']):
            home = PRIVATE/'heads'/v['tag']/arm; path = home/'complete.json'
            hid = dict(registration=artifact(PUBLIC/'registration_lock.json'), input=record, seed=seed, arm=arm,
                       implementation_amendment=artifact(PUBLIC/'implementation_amendment.json'))
            if path.exists():
                receipt = json.loads(path.read_text()); assert receipt['identity'] == hid
                for ref in receipt['artifacts'].values():
                    assert artifact(ROOT/ref['path']) == ref
                if verify:
                    model, state = method.restore(home)
                    assert state['identity'] == hid and state['step'] == cfg['head_training']['steps']
                    assert state['settings'] == cfg['head_training'] and state['seed'] == seed and state['arm'] == arm
                    pr = method.preprocess(x, target, v['sites'], v['outer'])
                    for k in ('mean', 'std', 'weights', 'known'):
                        np.testing.assert_array_equal(pr[k], state['preprocess'][k])
                    assert not state['draws'][~pr['known']].any()
                    with np.load(home/'scores.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], v['held_ids'])
                        np.testing.assert_array_equal(z['probability'], method.predict(model, hx, pr))
            else:
                if verify:
                    raise ValueError('Missing completed head')
                model, pr, fit = method.fit(x, target, v['sites'], v['outer'], arm=arm, seed=seed,
                    settings=cfg['head_training'], identity=hid, directory=home, resume=resume,
                    stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(tag=v['tag'], **kw))
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(home/'checkpoint.pt'),
                        projected_MLP_only_upper_approx_seconds=fit['seconds']/100*cfg['updates'],
                        wall_seconds=time.monotonic()-started, excludes_ancestry_preflight=True))
                    return
                probability = method.predict(model, hx, pr)
                risk.base.previous.parent.atomic_npz(home/'scores.npz', ids=v['held_ids'], probability=probability)
                restored, state = method.restore(home)
                np.testing.assert_array_equal(method.predict(restored, hx, state['preprocess']), probability)
                immutable_json(path, dict(identity=hid, fit=fit, fitting_prior=pr['prevalence'],
                    result_source='fresh_run_native_torch', artifacts=dict(
                        checkpoint=artifact(home/'checkpoint.pt'), scores=artifact(home/'scores.npz'))))
            refs.append(artifact(path)); beat('head_replayed' if verify else 'head_frozen', completed=len(refs), tag=v['tag'], arm=arm)
    assert len(refs) == cfg['new_heads']
    immutable_json(PUBLIC/('training_replay.json' if verify else 'prediction_freeze.json'),
                   dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs,
                        heads=len(refs), updates=cfg['updates'], held_outcomes_read=False))
    beat('training_replay_complete' if verify else 'training_complete', elapsed_seconds=time.monotonic()-started)


def check_freeze():
    path = PUBLIC/'prediction_freeze.json'; risk.base.previous.require_committed(path)
    doc = json.loads(path.read_text()); assert len(doc['receipts']) == 288
    for ref in doc['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values():
            assert artifact(ROOT/a['path']) == a


def evaluate(cfg, identity, verify=False):
    check_freeze(); rows = []; cached = {}
    for v in views(identity):
        x, target, hx, frozen, env, record = inputs(v)
        name = v['g']['group']+'_'+v['pair']
        if name not in cached:
            cached = {name: risk.base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        cv = v['data']['baseline_ade'][v['held_ids'], 1]
        y = risk.source.tail.diagnostic.event_targets(cached[name][v['te']], cv, v['pr']['positive_easy_cut'])
        prior_row = json.loads((risk.PUBLIC/'groups'/(name+'.json')).read_text())
        prior_fold = next(f for f in prior_row['folds'] if f['held'] == v['outer'])
        assert prior_fold['target_sha256'] == array_hash(y)
        event = method.event_target(y, frozen, env)
        overshoot = np.maximum(y[:, 3]-frozen[:, 1], 0)
        measured = {}
        for arm in cfg['arms']:
            home = PRIVATE/'heads'/v['tag']/arm
            prior = json.loads((home/'complete.json').read_text())['fitting_prior']
            with np.load(home/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids']); probability = z['probability'].copy()
            measured[arm] = diagnostic.measures(event, probability, overshoot, prior)
        for key, scores in dict(envelope=env, frozen_easy_fraction=np.divide(frozen[:, 3], env, out=np.zeros(len(env)), where=env>0),
                                frozen_harm_fraction=method.risk_features(frozen, env)[:, 0]).items():
            measured[key] = diagnostic.measures(event, scores, overshoot, None)
        rows.append(dict(tag=v['tag'], group=v['g']['group'], pair=v['pair'], outer=v['outer'],
                         producer=v['g']['producer'], controller=v['g']['controller'],
                         seed=int(v['g']['group'].split('_seed')[1].split('_')[0]),
                         metrics=measured, target_sha256=array_hash(event),
                         support=diagnostic.support(event, v['data']['sites'][v['held_ids']],
                             v['data']['recordings'][v['held_ids']], v['data']['agents'][v['held_ids']]),
                         result_source='fresh_run_source_development_readout'))
        beat('readout_replayed' if verify else 'readout', views=len(rows), tag=v['tag'])
    assert len(rows) == cfg['views']
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), rows=rows,
               independent_selection_read=False, reserved_calibration_read=False, confirmation_read=False)
    if verify:
        assert json.loads((PUBLIC/'readout.json').read_text()) == doc
        immutable_json(PUBLIC/'readout_replay.json', dict(views=len(rows), exact=True, readout=artifact(PUBLIC/'readout.json')))
    else:
        immutable_json(PUBLIC/'readout.json', doc)


def contrasts(rows, cfg):
    assignments = sorted({(r['producer'], r['controller']) for r in rows})
    out = []
    for producer, controller in assignments:
        for pair in cfg['pairs']:
            selected = [r for r in rows if (r['producer'], r['controller'], r['pair']) == (producer, controller, pair)]
            sites = sorted({r['outer'] for r in selected})
            assert len(selected) == 12 and len(sites) == 4
            for arm in cfg['arms']:
                for contrast in ('BCE_gain_prior', 'Brier_gain_prior', 'AP_gain_envelope', 'capture_gain_envelope',
                                 'AP_gain_frozen_easy_fraction', 'AP_gain_frozen_harm_fraction',
                                 'BCE_gain_linear', 'AP_gain_linear'):
                    values = np.full((3, 4), np.nan)
                    for r in selected:
                        m = r['metrics'][arm]; other = r['metrics']
                        if m.get('status') != 'measured':
                            continue
                        if contrast == 'BCE_gain_prior': val = m['prior_BCE']-m['BCE']
                        elif contrast == 'Brier_gain_prior': val = m['prior_Brier']-m['Brier']
                        elif contrast == 'BCE_gain_linear': val = other['linear']['BCE']-m['BCE']
                        else:
                            field = 'top10_overshoot_mass' if contrast.startswith('capture') else 'AP'
                            comparator = contrast.split('_gain_', 1)[1]
                            a, b = m.get(field), other[comparator].get(field)
                            val = a-b if a is not None and b is not None else np.nan
                        values[cfg['seeds'].index(r['seed']), sites.index(r['outer'])] = val
                    out.append(dict(producer=producer, controller=controller, pair=pair, arm=arm, contrast=contrast,
                                    **diagnostic.paired_interval(values, seed=cfg['bootstrap_seed'], draws=cfg['bootstrap'])))
    return out


def report(cfg):
    rows = json.loads((PUBLIC/'readout.json').read_text())['rows']; cs = contrasts(rows, cfg)
    summary = {}
    for pair in cfg['pairs']:
        summary[pair] = {}
        for arm in cfg['arms']:
            summary[pair][arm] = {}
            for contrast in sorted({r['contrast'] for r in cs}):
                group = [r for r in cs if (r['pair'], r['arm'], r['contrast']) == (pair, arm, contrast)]
                signs = [r.get('sign', 'not_estimable') for r in group]
                points = [r['point'] for r in group if 'point' in r]
                summary[pair][arm][contrast] = dict(counts={k: signs.count(k) for k in ('positive', 'negative', 'overlap', 'not_estimable')},
                                                   point_range=[min(points), max(points)] if points else None)
    required = ('BCE_gain_prior', 'AP_gain_envelope', 'capture_gain_envelope')
    signal = all(summary['full']['mlp'][k]['counts']['positive'] == 6 for k in required)
    doc = dict(contrasts=cs, summary=summary, gates=dict(cap_event_transfer_signal=signal,
        magnitude_calibration_proven=False, policy_utility_proven=False, deployment_changed=False,
        independent_confirmation=False, stage5c_executed=False, smc_enabled=False))
    immutable_json(PUBLIC/'aggregate_metrics.json', doc)
    lines = ['# Cap-Event Learnability Results', '', '## Material Passport',
        'fresh_run:288 native-Torch heads,576000 fixed updates and144 source-held readouts.',
        'cached_verified:frozen source forecasters, inner OOF risk teachers and outer risk estimators.',
        'not_run:new trajectory training, policy evaluation and independent selection/calibration/confirmation.', '',
        '| Inputs / arm / contrast | Positive / negative / overlap / missing intervals | Point range |', '|---|---|---|']
    for pair, arms in summary.items():
        for arm, metrics in arms.items():
            for contrast, value in metrics.items():
                lines.append(f"| {pair}/{arm}/{contrast} | {list(value['counts'].values())} | {value['point_range']} |")
    lines += ['', 'Positive means improvement. Ranges are six point estimates, not one confidence interval.',
        'Each interval averages three seeds within locality before3000 paired resamples of four localities.',
        'The six assignments overlap. These source-development intervals are exploratory and not multiplicity-adjusted.',
        'Source outcomes have prior development exposure. No independent scene or window-count significance claim.',
        'The target transports a two-locality inner producer to a three-locality outer producer.',
        'The top10% is a fixed ranking diagnostic, not a deployment threshold. Event probability is not expected harm.',
        '', '```json', json.dumps(doc['gates'], indent=2), '```', '',
        'Obs8/pred12 annotation steps, detector pixels only; no metric/seconds/physical-safety/foundation claim.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(doc['gates'], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['register', 'support', 'pilot', 'train', 'evaluate', 'report', 'verify_training', 'verify_eval'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, identity = registration(args.phase == 'register')
        if args.phase == 'support': support(cfg, identity)
        elif args.phase in ('pilot', 'train', 'verify_training'):
            training(cfg, identity, pilot=args.phase == 'pilot', resume=args.resume, verify=args.phase == 'verify_training')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, verify=args.phase == 'verify_eval')
        elif args.phase == 'report': report(cfg)


if __name__ == '__main__':
    main()
