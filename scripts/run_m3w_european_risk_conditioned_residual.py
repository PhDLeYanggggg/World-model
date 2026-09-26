"""Risk-conditioned fixed residual study; source development, not deployment."""
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
from scripts import run_m3w_european_event_transport as parent
from src.world_model import m3w_risk_conditioned_residual as method
import numpy as np
import torch

nested, context_run = parent.parent, parent.parent.parent
base, source = parent.base, parent.source
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_risk_conditioned_residual_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_risk_conditioned_residual_v1'
CONFIG = 'configs/m3w_european_risk_conditioned_residual_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_risk_conditioned_residual.py',
         'tests/test_m3w_risk_conditioned_residual.py',
         'scripts/run_m3w_european_risk_conditioned_residual.py',
         str(PUBLIC.relative_to(ROOT)/'protocol.md')]
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text()); _, pid = parent.registration()
    check = json.loads((parent.PUBLIC/'verification.json').read_text())
    assert check['all_passed']
    for p, h in check['artifacts'].items(): assert digest(parent.PUBLIC/p) == h
    for p, h in check['source_bindings'].items(): assert digest(ROOT/p) == h
    assert cfg['variants'] == list(nested.method.VARIANTS) and cfg['arms'] == list(method.ARMS)
    assert (cfg['views'], cfg['fits'], cfg['ridge'], cfg['new_neural_updates']) == (144, 864, .1, 0)
    assert not any(cfg[k] for k in ('threshold_refit', 'selection_access', 'reserved_calibration_access',
                                  'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent=pid, parent_verification=artifact(parent.PUBLIC/'verification.json'),
                    bindings={p: digest(ROOT/p) for p in FILES})
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        base.previous.require_committed(path)
    return cfg, identity


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def check_sources(identity):
    nested.checked_training(); context_run.check_frozen(identity['parent']['parent']['parent'])
    parent.check_freeze()
    for ref in json.loads((context_run.PUBLIC/'prediction_freeze.json').read_text())['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a


def banks(v):
    bank = {}
    for inner in nested.method.fitting_sites(v['sites'], v['outer']):
        home = nested.PRIVATE/'heads'/v['tag']/inner
        row = json.loads((home/'complete.json').read_text())
        with np.load(home/'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], v['ids']); pred = z['scores'].copy()
        bank[inner] = dict(prediction=pred, training_sites=row['input']['training_sites'], cut=row['input']['cut'])
    common = parent.method.common_event_target(v['raw'], v['cv'], v['sites'], v['outer'], v['pr']['positive_easy_cut'])
    np.testing.assert_array_equal(common, v['outer_y'])
    return bank, common, np.where(v['env'] > 0, v['pr']['weights'], 0)


def support(cfg, identity, verify=False):
    check_sources(identity); rows = []; receipts = []
    for v in nested.views(identity['parent']['parent']):
        out = PRIVATE/'support'/v['tag']; receipt = out/'complete.json'
        if receipt.exists() and not verify:
            r = json.loads(receipt.read_text())
            assert r['registration'] == artifact(PUBLIC/'registration_lock.json')
            for a in r['artifacts'].values(): assert artifact(ROOT/a['path']) == a
            rows.extend(json.loads((out/'diagnostic.json').read_text())['rows'])
            receipts.append(artifact(receipt)); continue
        bank, y, w = banks(v)
        model, state = nested.neural.restore(context_run.frozen_directory(v['tag'], 'original'))
        for k in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(state['preprocess'][k], v['pr'][k])
        for k in ('cost_scale', 'positive_easy_cut', 'hard_cut'): assert state['preprocess'][k] == v['pr'][k]
        original, _ = nested.neural.predict(model, v['x'], v['env'], v['pr'])
        old = json.loads((context_run.PRIVATE/'probes'/v['tag']/'original'/'model.json').read_text())
        common_models = json.loads((parent.PRIVATE/'probes'/v['tag']/'models.json').read_text())
        group = []
        for variant in cfg['variants']:
            p, inner_y, cuts, producers = nested.method.assemble(bank, v['raw'], v['cv'], v['sites'], v['outer'], variant)
            assert array_hash(p) == common_models['alignment'][variant]['prediction_sha256']
            diagnostic = method.producer_diagnostic(v['context'], p, original, v['env'], y, w,
                common_models['models'][variant]['context_bias'], old['probes']['context_bias'])
            use = w > 0; positive = int((y[use, 3] > 0).sum())
            supported = positive > 0 and all(s > 1e-8 for k in ('inner', 'outer') for s in diagnostic[k]['risk_feature_std'])
            group.append(dict(tag=v['tag'], pair=v['pair'], variant=variant, known=int(use.sum()),
                easy_harm_rows=positive, numerically_supported=bool(supported),
                original_fitting_score_sha256=array_hash(original), inner_score_sha256=array_hash(p),
                target_sha256=array_hash(y), fitting_ids_sha256=array_hash(v['ids']), **diagnostic))
        doc = dict(rows=group, outer_held_labels_used=False)
        if verify:
            assert json.loads((out/'diagnostic.json').read_text()) == doc
            with np.load(out/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['ids']); np.testing.assert_array_equal(z['scores'], original)
        else:
            immutable_json(out/'diagnostic.json', doc)
            base.previous.parent.atomic_npz(out/'scores.npz', ids=v['ids'], scores=original)
            immutable_json(receipt, dict(registration=artifact(PUBLIC/'registration_lock.json'),
                artifacts=dict(diagnostic=artifact(out/'diagnostic.json'), scores=artifact(out/'scores.npz'))))
        rows.extend(group); receipts.append(artifact(receipt)); beat('support_replayed' if verify else 'support', views=len(receipts), tag=v['tag'])
    assert len(rows) == 432 and len(receipts) == 144
    doc = dict(registration=artifact(PUBLIC/'registration_lock.json'), rows=rows, receipts=receipts,
        training_allowed=all(r['numerically_supported'] for r in rows), projection_checks=432,
        fresh_Torch_fitting_inferences=144, new_neural_updates=0, outer_held_labels_used=False,
        statistical_power_established=False, causal_mechanism_proven=False)
    immutable_json(PUBLIC/('support_replay.json' if verify else 'support_report.json'), doc)
    if not verify:
        lines = ['# Fitting-Only Producer Diagnostic', '', 'Fresh144 frozen-Torch fitting predictions;432 exact fixed-design producer projections.',
                 'This is not new neural training, causal attribution or held-out accuracy.', '',
                 '| Inputs/bank | Views | Median score-gap RMS | Minimum easy-harm rows | Median outer outside inner central90% (two risk scores) |', '|---|---:|---:|---:|---|']
        for pair in cfg['pairs']:
            for variant in cfg['variants']:
                r = [q for q in rows if q['pair'] == pair and q['variant'] == variant]
                lines.append(f"| {pair}/{variant} | {len(r)} | {np.median([q['prediction_difference_RMS'] for q in r]):.6g} | {min(q['easy_harm_rows'] for q in r)} | {np.median([q['outer_outside_inner_central90_fraction'] for q in r],axis=0).tolist()} |")
        lines += ['', 'Training allowed: '+str(doc['training_allowed'])+'. All views retained. Central90% is descriptive, not an OOD or safety bound.',
                  'Original versus inner scores differ on matched fitting rows. Identity isolates a producer-dependent term but does not establish the source of held errors.',
                  'No fitting support threshold was selected from held outcomes. Obs8/pred12 annotation steps, pixels only.']
        (PUBLIC/'support_report.md').write_text('\n'.join(lines)+'\n')


def fit(cfg, identity, verify=False):
    check_sources(identity)
    support_doc = json.loads((PUBLIC/'support_report.json').read_text())
    assert support_doc['training_allowed'] and support_doc['registration'] == artifact(PUBLIC/'registration_lock.json')
    base.previous.require_committed(PUBLIC/'support_report.json'); refs = []; seconds = []
    for v in nested.views(identity['parent']['parent']):
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; per-view resume available')
        out = PRIVATE/'probes'/v['tag']; receipt = out/'complete.json'
        if receipt.exists() and not verify:
            row = json.loads(receipt.read_text()); assert row['registration'] == artifact(PUBLIC/'registration_lock.json')
            for a in row['artifacts'].values(): assert artifact(ROOT/a['path']) == a
            refs.append(artifact(receipt)); seconds.append(row['fit_seconds']); continue
        start = time.monotonic(); bank, y, w = banks(v)
        with np.load(context_run.frozen_directory(v['tag'], 'original')/'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], v['held_ids']); frozen = z['scores'].copy()
        held_env = v['pairs']['B'][v['pair']][1][v['te']]
        models, alignment, scores = {}, {}, {}
        for variant in cfg['variants']:
            p, inner_y, cuts, producers = nested.method.assemble(bank, v['raw'], v['cv'], v['sites'], v['outer'], variant)
            models[variant] = {}; alignment[variant] = dict(prediction_sha256=array_hash(p),
                common_target_sha256=array_hash(y), row_producer_sha256=array_hash(producers))
            for arm in cfg['arms']:
                m = method.fit(v['context'], p, v['env'], y, w, v['sites'], v['outer'], arm=arm, variant=variant, ridge=cfg['ridge'])
                models[variant][arm] = m
                scores[variant+'__'+arm] = method.predict(m, v['held_context'], frozen, held_env)
        doc = dict(models=models, alignment=alignment, ids_sha256=array_hash(v['ids']),
            held_context_sha256=array_hash(v['held_context']), outer_cut=v['pr']['positive_easy_cut'],
            outer_held_labels_used=False, common_response_used_only_for_fitting=True)
        if verify:
            assert json.loads((out/'models.json').read_text()) == doc
            with np.load(out/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], v['held_ids'])
                for k, p in scores.items(): np.testing.assert_array_equal(p, z[k])
            seconds.append(json.loads(receipt.read_text())['fit_seconds'])
        else:
            immutable_json(out/'models.json', doc)
            base.previous.parent.atomic_npz(out/'scores.npz', ids=v['held_ids'], **scores)
            elapsed = time.monotonic()-start; seconds.append(elapsed)
            immutable_json(receipt, dict(registration=artifact(PUBLIC/'registration_lock.json'), fit_seconds=elapsed,
                artifacts=dict(models=artifact(out/'models.json'), scores=artifact(out/'scores.npz'))))
        refs.append(artifact(receipt)); beat('fits_replayed' if verify else 'fits_frozen', views=len(refs), tag=v['tag'])
    assert len(refs) == 144
    immutable_json(PUBLIC/('fit_replay.json' if verify else 'prediction_freeze.json'),
        dict(receipts=refs, closed_form_fits=864, new_neural_updates=0, outer_held_labels_used=False, summed_fit_seconds=sum(seconds)))


def check_freeze():
    base.previous.require_committed(PUBLIC/'prediction_freeze.json')
    for ref in json.loads((PUBLIC/'prediction_freeze.json').read_text())['receipts']:
        assert artifact(ROOT/ref['path']) == ref
        for a in json.loads((ROOT/ref['path']).read_text())['artifacts'].values(): assert artifact(ROOT/a['path']) == a


def evaluate(cfg, identity, verify=False):
    check_freeze(); groups = {}; by_cache = {}; refs = []; direct = 0
    for v in nested.views(identity['parent']['parent']):
        name = v['g']['group']+'_'+v['pair']; path = PUBLIC/'groups'/(name+'.json')
        if name not in by_cache: by_cache = {name: base.pair_inputs(v['g'], v['data'], v['pairs'], v['pair'])[2]}
        cv = v['data']['baseline_ade'][v['held_ids'], 1]
        y = source.tail.diagnostic.event_targets(by_cache[name][v['te']], cv, v['pr']['positive_easy_cut'])
        old = json.loads((parent.PUBLIC/'groups'/(name+'.json')).read_text())
        prior = next(f for f in old['folds'] if f['held'] == v['outer'])
        assert prior['target_sha256'] == array_hash(y) and prior['held_ids_sha256'] == array_hash(v['held_ids'])
        metrics = dict(prior['metrics'])
        orig = json.loads((source.PUBLIC/'groups'/(name+'.json')).read_text())
        edges = next(f for f in orig['folds'] if f['held'] == v['outer'])['training']['cost_only']['edges']
        env = v['pairs']['B'][v['pair']][1][v['te']]
        with np.load(context_run.frozen_directory(v['tag'], 'original')/'scores.npz', allow_pickle=False) as z: frozen = z['scores'].copy()
        with np.load(PRIVATE/'probes'/v['tag']/'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], v['held_ids'])
            for variant in cfg['variants']:
                for arm in cfg['arms']:
                    k = variant+'__'+arm; p = z[k]
                    np.testing.assert_array_equal(p[:, :3], frozen[:, :3])
                    assert np.isfinite(p).all() and (p[:, 3] >= 0).all() and (p[:, 3] <= p[:, 1]).all()
                    metric = source.parent.parent.measure(p, y, env, np.repeat(v['outer'], len(y)), edges)
                    for subset, use in (('all', np.ones(len(y), bool)), ('envelope_positive', env > 0)):
                        use &= np.isfinite(y).all(1)
                        np.testing.assert_allclose(np.mean((p[use, 3]-y[use, 3])**2), metric[subset]['harm_MSE'], rtol=1e-12, atol=1e-12)
                        direct += 1
                    metrics['risk__'+k] = metric
        groups.setdefault(name, []).append(dict(held=v['outer'], metrics=metrics,
            target_sha256=array_hash(y), held_ids_sha256=array_hash(v['held_ids'])))
        if len(groups[name]) == 4:
            row = dict(group=v['g']['group'], producer=v['g']['producer'], controller=v['g']['controller'],
                seed=int(v['g']['group'].split('_seed')[1].split('_')[0]), pair=v['pair'], folds=groups.pop(name),
                result_source='fresh_run_risk_conditioned_readout_cached_verified_controls_source_development')
            if verify: assert json.loads(path.read_text()) == row
            else: immutable_json(path, row)
            refs.append(artifact(path)); beat('readout_replayed' if verify else 'readout', groups=len(refs), group=name)
    assert len(refs) == 36 and direct == 1728 and not groups
    immutable_json(PUBLIC/('eval_replay.json' if verify else 'completion_checks.json'), dict(all_passed=True, groups=refs, direct_MSE_checks=direct))


def report(cfg, identity):
    rows = []
    for a in json.loads((PUBLIC/'completion_checks.json').read_text())['groups']:
        assert artifact(ROOT/a['path']) == a; rows.append(json.loads((ROOT/a['path']).read_text()))
    controls = dict(original='original', common_context='common__oof__context_bias',
        inner_event='oof__context_bias', outer_context='outer_in_sample__context_bias',
        risk_only='risk__oof__risk_only', risk_next='risk__in_sample_next__risk_context', risk_prev='risk__in_sample_prev__risk_context')
    comparisons = {'risk_oof_vs_'+k: ('risk__oof__risk_context', a) for k, a in controls.items()}
    for variant in cfg['variants']:
        for arm in cfg['arms']: comparisons[variant+'_'+arm+'_vs_original'] = ('risk__'+variant+'__'+arm, 'original')
    contrasts = {}
    for key, (a, b) in comparisons.items():
        proxy = [dict(r, folds=[dict(f, metrics={'fractional': f['metrics'][a], 'mean': f['metrics'][b]}) for f in r['folds']]) for r in rows]
        contrasts[key] = context_run.paired_contrasts(proxy, cfg)
    summary = context_run.summarize_contrasts(contrasts)
    primary = [summary['risk_oof_vs_'+k]['full'] for k in controls]
    mechanism = summary['risk_oof_vs_common_context']['full']['envelope_positive__harm_MSE_gain_percent']['positive'] == 6
    positive = all(v['envelope_positive__harm_MSE_gain_percent']['positive'] == 6 for v in primary)
    guards = all(v['envelope_positive__'+k]['negative'] == v['envelope_positive__'+k]['not_estimable'] == 0
        for v in primary for k in ('top10_gain_pp', 'coverage_log_error_reduction'))
    gates = dict(risk_conditioning_signal=mechanism, primary_six_positive_all_controls=positive,
        tail_coverage_guards=guards, risk_conditioned_repair_signal=positive and guards,
        policy_evaluated=False, deployment_changed=False, independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC/'aggregate_metrics.json', dict(contrasts=contrasts, summary=summary, gates=gates))
    lines = ['# Risk-Conditioned Residual Results', '', 'fresh_run:864 fixed ridge probes and144 fitting-only frozen-Torch inferences.',
        'cached_verified:432 inner risk heads, original outer estimators and prior controls. No new neural or trajectory training.',
        'Source development only; independent selection, calibration and confirmation remain unopened.', '',
        '| Comparison / inputs | Positive / negative / overlapping / missing MSE intervals | Point range (%) |', '|---|---|---|']
    for key, ps in summary.items():
        for pair, v in ps.items():
            q = v['envelope_positive__harm_MSE_gain_percent']
            lines.append(f"| {key} / {pair} | {q['positive']} / {q['negative']} / {q['overlap']} / {q['not_estimable']} | {q['point_range']} |")
    lines += ['', 'Range is six point estimates, not one CI. Three seeds averaged per locality;3000 paired resamples of four localities per assignment.',
        'Six assignments overlap. No multiplicity adjustment or window-independent claims. No negative interval is not a safety proof.',
        'Only predicted H_E changes. This is cost estimation, not a trajectory gain, independent calibration or deployment.', '',
        '```json', json.dumps(gates, indent=2), '```', '',
        'Obs8/pred12 annotation steps, detector pixels. No metric/seconds, human-gold, physical-safety, true3D or foundation claim.']
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n'); print(json.dumps(gates, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['register', 'support', 'fit', 'evaluate', 'report', 'verify_support', 'verify_fit', 'verify_eval'])
    args = ap.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); cfg, identity = registration(args.phase == 'register')
        if args.phase in ('support', 'verify_support'): support(cfg, identity, args.phase == 'verify_support')
        elif args.phase in ('fit', 'verify_fit'): fit(cfg, identity, args.phase == 'verify_fit')
        elif args.phase in ('evaluate', 'verify_eval'): evaluate(cfg, identity, args.phase == 'verify_eval')
        elif args.phase == 'report': report(cfg, identity)


if __name__ == '__main__': main()
