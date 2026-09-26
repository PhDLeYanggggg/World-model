"""Fitting-only closed-form context probes, frozen before source-held readout."""
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
    raise RuntimeError('Use native arm64 Python')
from scripts import run_m3w_european_severity_transport as parent
from src.world_model import m3w_context_residual as method
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
import numpy as np
import torch
source = parent.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_context_residual_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_context_residual_v1'
CONFIG = 'configs/m3w_european_context_residual_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_context_residual.py', 'tests/test_m3w_context_residual.py',
         'scripts/run_m3w_european_context_residual.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = source.artifact, source.digest, source.immutable_json, source.array_hash


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    v = json.loads((parent.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p, h in v['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in v['source_bindings'].items(): assert digest(ROOT/p) == h
    assert (cfg['frozen_views'], cfg['closed_form_fits'], cfg['ridge']) == (144, 864, .1)
    assert cfg['probe_arms'] == list(method.ARMS) and cfg['new_neural_updates'] == 0
    assert not any(cfg[k] for k in ('threshold_refit', 'selection_access', 'reserved_calibration_access',
        'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p:digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity; source.base.previous.require_committed(path)
    return cfg, identity


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    source.base.base.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def frozen_directory(tag, arm):
    if arm == 'severity_aux': return source.PRIVATE/'heads'/tag
    return source.parent.PRIVATE/'heads'/tag/('cost_only' if arm == 'original' else 'membership_aux')


def check_frozen(identity):
    si = identity['parent']['parent']
    source.checked_training(si); source.parent.checked_training(si['parent']['parent'])


def fit(cfg, identity, *, pilot=False, verify=False):
    check_frozen(identity); refs = []; started = time.monotonic()
    for g, data, pairs in source.base.contexts(identity['parent']['parent']['source']):
        name = g['group']; bi = pairs['B']['ids']; sites = data['sites'][bi]
        for pair in cfg['pairs']:
            bx, be, reference, candidate = pairs['B'][pair]
            context = method.features(data['geometry'][bi], data['width'][bi], reference, candidate)
            for held in sorted(set(sites)):
                if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume completed views')
                tr, te = sites != held, sites == held; tag = name+'_'+pair+'_'+held
                x, env, y, easy, ss, ids, pr = source.diagnosis.fitting_inputs(g, data, pairs, pair, held)
                np.testing.assert_array_equal(ids, bi[tr])
                w = np.where(env > 0, pr['weights'], 0)
                for arm in cfg['frozen_arms']:
                    out = PRIVATE/'probes'/tag/arm; receipt = out/'complete.json'
                    if receipt.exists() and not verify:
                        row = json.loads(receipt.read_text())
                        assert row['registration'] == artifact(PUBLIC/'registration_lock.json')
                        for a in row['artifacts'].values(): assert artifact(ROOT/a['path']) == a
                        refs.append(artifact(receipt)); continue
                    directory = frozen_directory(tag, arm); cp = artifact(directory/'checkpoint.pt')
                    old = json.loads((directory/'complete.json').read_text())
                    for k, a in (('train_x_sha256', x), ('train_y_sha256', y), ('train_ids_sha256', ids)):
                        assert old['input'][k] == array_hash(a)
                    model, state = source.method.restore(directory)
                    for k in ('mean', 'std', 'known', 'weights'):
                        np.testing.assert_array_equal(state['preprocess'][k], pr[k])
                    fitted, _ = source.method.predict(model, x, env, pr)
                    with np.load(directory/'scores.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], bi[te]); base = z['scores'].copy()
                    replay, _ = source.method.predict(model, bx[te][:4096], be[te][:4096], pr)
                    np.testing.assert_array_equal(replay[:, [1, 3]], base[:4096, [1, 3]])
                    probes = method.fit(context[tr], fitted, y, w, ss, held, ridge=cfg['ridge'])
                    analysis_target = y.copy(); analysis_target[env <= 0] = np.nan
                    training = method.context_table(context[tr], probes['context_bias']['cuts'], fitted,
                        analysis_target, ss, data['recordings'][ids], data['agents'][ids], probes['context_bias']['rms'])
                    values = {a:method.predict(probes[a], context[te], base) for a in method.ARMS}
                    doc = dict(probes=probes, training_contexts=training, held=held, arm=arm,
                        fitting_ids_sha256=array_hash(ids), fitting_targets_sha256=array_hash(y),
                        fitting_context_sha256=array_hash(context[tr]), held_context_sha256=array_hash(context[te]),
                        held_ids_sha256=array_hash(bi[te]), checkpoint=cp, base_scores=artifact(directory/'scores.npz'),
                        future_held_labels_used=False, positive_disagreement_fitting_rows=int((w > 0).sum()))
                    if verify:
                        assert json.loads((out/'model.json').read_text()) == doc
                        with np.load(out/'scores.npz', allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['ids'], bi[te])
                            for a in method.ARMS: np.testing.assert_array_equal(z[a], values[a])
                    else:
                        immutable_json(out/'model.json', doc)
                        source.base.previous.parent.atomic_npz(out/'scores.npz', ids=bi[te], **values)
                        immutable_json(receipt, dict(registration=artifact(PUBLIC/'registration_lock.json'),
                            artifacts=dict(model=artifact(out/'model.json'), scores=artifact(out/'scores.npz')),
                            closed_form_fits=2, new_neural_updates=0))
                    assert artifact(directory/'checkpoint.pt') == cp
                    refs.append(artifact(receipt))
                    beat('probe_replayed' if verify else 'probe_fitted', completed=len(refs), group=name, pair=pair, held=held, arm=arm)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', dict(frozen_view=tag, closed_form_fits=6,
                        elapsed_seconds=time.monotonic()-started, excludes_registration_and_parent_preflight=True))
                    return
    assert len(refs) == 432; check_frozen(identity)
    immutable_json(PUBLIC/('fit_replay.json' if verify else 'prediction_freeze.json'),
        dict(registration=artifact(PUBLIC/'registration_lock.json'), receipts=refs, closed_form_fits=864,
             new_neural_updates=0, held_labels_used_for_fit=False, parent_checkpoints_unchanged=True))
    beat('fit_replay_complete' if verify else 'fit_complete', elapsed_seconds=time.monotonic()-started)


def evaluate(cfg, identity, verify=False):
    freeze = json.loads((PUBLIC/'prediction_freeze.json').read_text())
    source.base.previous.require_committed(PUBLIC/'prediction_freeze.json')
    for ref in freeze['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    refs = []
    for g, data, pairs in source.base.contexts(identity['parent']['parent']['source']):
        bi = pairs['B']['ids']; sites = data['sites'][bi]; cv = data['baseline_ade'][bi, 1]
        for pair in cfg['pairs']:
            name = g['group']; path = PUBLIC/'groups'/(name+'_'+pair+'.json')
            receipt = PRIVATE/'readout'/path.name
            if path.exists() and not verify:
                assert artifact(path) == json.loads(receipt.read_text()); refs.append(artifact(path)); continue
            _, be, by, *_ = source.base.pair_inputs(g, data, pairs, pair)
            _, _, reference, candidate = pairs['B'][pair]
            context = method.features(data['geometry'][bi], data['width'][bi], reference, candidate)
            old = json.loads((source.PUBLIC/'groups'/path.name).read_text()); folds = []
            for held in sorted(set(sites)):
                te = sites == held; tag = name+'_'+pair+'_'+held
                prior = next(f for f in old['folds'] if f['held'] == held)
                target = source.tail.diagnostic.event_targets(by[te], cv[te], prior['easy_cut'])
                assert array_hash(target) == prior['target_sha256'] and array_hash(bi[te]) == prior['held_ids_sha256']
                metrics, patterns = {}, {}
                for arm in cfg['frozen_arms']:
                    key = {'original':'cost_only', 'ordinary_aux':'membership_aux', 'severity_aux':'severity_aux'}[arm]
                    metrics[arm] = prior['metrics'][key]; edges = prior['training'][key]['edges']
                    out = PRIVATE/'probes'/tag/arm; doc = json.loads((out/'model.json').read_text())
                    assert doc['held_context_sha256'] == array_hash(context[te])
                    with np.load(frozen_directory(tag, arm)/'scores.npz', allow_pickle=False) as z: base = z['scores'].copy()
                    with np.load(out/'scores.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['ids'], bi[te])
                        for probe in cfg['probe_arms']:
                            pred = z[probe]; np.testing.assert_array_equal(pred[:, :3], base[:, :3])
                            assert (pred[:, 3] >= 0).all() and (pred[:, 3] <= pred[:, 1]).all()
                            metrics[arm+'__'+probe] = source.parent.parent.measure(pred, target, be[te], sites[te], edges)
                    analysis_target = target.copy(); analysis_target[be[te] <= 0] = np.nan
                    contexts = method.context_table(context[te], doc['probes']['context_bias']['cuts'], base,
                        analysis_target, sites[te], data['recordings'][bi[te]], data['agents'][bi[te]], doc['probes']['context_bias']['rms'])
                    patterns[arm] = method.repeated_contexts(doc['training_contexts'], contexts)
                folds.append(dict(held=held, metrics=metrics, context_patterns=patterns,
                    held_ids_sha256=array_hash(bi[te]), target_sha256=array_hash(target)))
            row = dict(group=name, producer=g['producer'], controller=g['controller'],
                seed=int(name.split('_seed')[1].split('_')[0]), pair=pair, folds=folds,
                result_source='fresh_run_closed_form_probe_readout_cached_verified_frozen_estimators_source_development')
            if verify: assert json.loads(path.read_text()) == row
            else: immutable_json(path, row); immutable_json(receipt, artifact(path))
            refs.append(artifact(path)); beat('readout_replayed' if verify else 'readout', groups=len(refs), group=name, pair=pair)
    assert len(refs) == 36
    immutable_json(PUBLIC/('eval_replay.json' if verify else 'completion_checks.json'),
        dict(registration=artifact(PUBLIC/'registration_lock.json'), groups=refs, all_passed=True))


def aggregate(rows, cfg):
    comparisons = {}
    for arm in cfg['frozen_arms']:
        comparisons[arm+'_global_vs_self'] = (arm+'__global_bias', arm)
        comparisons[arm+'_context_vs_self'] = (arm+'__context_bias', arm)
        comparisons[arm+'_context_vs_global'] = (arm+'__context_bias', arm+'__global_bias')
        if arm != 'original': comparisons[arm+'_context_vs_original'] = (arm+'__context_bias', 'original')
    cs = {}
    for key, (a, b) in comparisons.items():
        proxy = [dict(r, folds=[dict(f, metrics={'fractional':f['metrics'][a], 'mean':f['metrics'][b]})
                    for f in r['folds']]) for r in rows]
        cs[key] = paired_contrasts(proxy, cfg)
    summary = summarize_contrasts(cs)
    patterns = {pair:{arm:{name:{k:sum(f['context_patterns'][arm][j][k] for r in rows if r['pair'] == pair for f in r['folds'])
        for k in ('supported_cells', 'repeated_fitting_sign', 'held_same_sign', 'held_opposite_sign')}
        for j, name in enumerate(method.NAMES)} for arm in cfg['frozen_arms']} for pair in cfg['pairs']}
    primary = [summary[k]['full'] for k in ('original_context_vs_self', 'original_context_vs_global')]
    positive = all(v['envelope_positive__harm_MSE_gain_percent']['positive'] == 6 for v in primary)
    guards = all(v['envelope_positive__'+k]['negative'] == v['envelope_positive__'+k]['not_estimable'] == 0
        for v in primary for k in ('top10_gain_pp', 'coverage_log_error_reduction'))
    return dict(contrasts=cs, summary=summary, patterns=patterns,
        gates=dict(primary_six_positive_vs_original_and_global=positive, tail_coverage_guards=guards,
            diagnostic_context_signal=positive and guards, independent_calibration=False,
            new_neural_training=False, policy_evaluated=False, deployment_changed=False,
            independent_confirmation=False, stage5c_executed=False, smc_enabled=False))


def report(cfg, identity):
    checks = json.loads((PUBLIC/'completion_checks.json').read_text()); rows = []
    for ref in checks['groups']:
        assert artifact(ROOT/ref['path']) == ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    a = aggregate(rows, cfg); immutable_json(PUBLIC/'aggregate_metrics.json', a)
    lines = ['# Causal Context Residual Results', '', '## Material Passport',
        '864 fresh closed-form probes on432 frozen estimators /144 aligned views. Zero new Torch updates.',
        'Cached_verified producers, predictors and exposed source development. Not independent calibration.', '',
        '| Comparison / inputs | Positive / negative / overlap / missing MSE intervals | Point range (%) |',
        '|---|---|---|']
    for c, ps in a['summary'].items():
        for p, v in ps.items():
            q = v['envelope_positive__harm_MSE_gain_percent']
            lines.append(f"| {c} / {p} | {q['positive']} / {q['negative']} / {q['overlap']} / {q['not_estimable']} | {q['point_range']} |")
    lines += ['', 'Three seeds averaged within locality;3000 resamples of four localities per assignment.',
        'All six assignments retained; dependent exploratory comparisons, no multiplicity adjustment.',
        'Training residuals are in-sample for the frozen base. Any positive probe needs separate nested-OOF validation.',
        '', '```json', json.dumps(a['gates'], indent=2), '```', '',
        'Detector pixels and annotation steps only. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(dict(summary=a['summary'], gates=a['gates']), indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', required=True,
        choices=['register', 'pilot', 'fit', 'evaluate', 'report', 'verify_fit', 'verify_eval'])
    args = ap.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); cfg, identity = registration(args.phase == 'register')
        if args.phase in ('pilot', 'fit', 'verify_fit'): fit(cfg, identity, pilot=args.phase == 'pilot', verify=args.phase == 'verify_fit')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, verify=args.phase == 'verify_eval')
        elif args.phase == 'report': report(cfg, identity)


if __name__ == '__main__': main()
