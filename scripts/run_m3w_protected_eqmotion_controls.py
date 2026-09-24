"""Reuse pair-excluded EqMotion and complete its matched full-forecast controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch

from scripts import run_m3w_protected_motion_controls as motion
from scripts import run_m3w_eqmotion_cost_refit as eq
from scripts.run_m3w_bounded_cost import training_data, features, read_arrays
from scripts.run_m3w_native_forecast import array_hash, file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_bounded_cost_head import build, predict
from src.training.m3w_external_cost_bank import fit_forest, predict_forest
from src.evaluation.m3w_protected_motion_controls import causal_candidate, protected_decisions, match_strict_count
from src.evaluation.m3w_protected_eqmotion_controls import check_reused_head, verify_same_population
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast

CONFIG = 'configs/m3w_protected_eqmotion_controls_v1.json'
CODE = ('scripts/run_m3w_protected_eqmotion_controls.py',
    'src/evaluation/m3w_protected_eqmotion_controls.py', 'tests/test_m3w_protected_eqmotion_controls.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    ec, data, views, _, predictions, _, eid = eq.load()
    mc, md, mv, _, mid, refs = motion.load()
    verify_same_population(data, md)
    del md
    if (cfg['sites'] != mc['sites'] or cfg['seeds'] != mc['seeds'] or cfg['comparators'] != mc['actions']
            or cfg['heads'] != mc['heads'] or cfg['forest'] != mc['forest']
            or cfg['policies'] != mc['policies'] or ec['training'] != mc['training']
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'new_forecast_training',
                'new_neural_head_training', 'external_readout', 'risk_calibration',
                'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed matched full-EqMotion extension required')
    bindings = dict(eid['source_bindings'])
    for path, sha in mid['source_bindings'].items():
        if path in bindings and bindings[path] != sha:
            raise ValueError('Parent identities conflict')
        bindings[path] = sha
    def bind(path, expected=None):
        path = str(path)
        sha = file_digest(ROOT/path)
        if (expected is not None and sha != expected) or (path in bindings and bindings[path] != sha):
            raise ValueError('Changed source '+path)
        bindings[path] = sha
    ap, ep = Path(mc['reports'])/'analysis.json', Path(ec['reports'])/'analysis.json'
    bind(ap, cfg['motion_analysis_sha256']); bind(ep, cfg['eqmotion_analysis_sha256'])
    ma, ea = json.loads((ROOT/ap).read_text()), json.loads((ROOT/ep).read_text())
    assert ma['identity'] == mid and ea['identity'] == eid
    for report, analysis, files in ((mc['reports'], ap, ('verification.json', 'independent_verification.json')),
                                    (ec['reports'], ep, ('replay.json', 'independent_verification.json'))):
        for name in files:
            path = Path(report)/name; r = json.loads((ROOT/path).read_text())
            assert r['all_checks_passed'] and r['analysis_sha256'] == file_digest(ROOT/analysis)
            bind(path)
    neural = {}
    for key in views:
        path = Path(ec['output'])/'trials'/key/'bounded_fraction'/'complete.json'
        r = json.loads((ROOT/path).read_text())
        assert r['identity']['identity'] == eid and r['fit']['complete']
        bind(path); bind(r['checkpoint'], r['checkpoint_sha256'])
        neural[key] = r
    for r in ma['fits']:
        bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in ma['decision_archives'] + ea['archives']:
        bind(r['path'], r['sha256'])
    cuts_path = Path(mc['output'])/'decisions_complete.json'
    bind(cuts_path)
    cuts = json.loads((ROOT/cuts_path).read_text())['cuts']
    for path in (CONFIG, cfg['registration'], *CODE): bind(path)
    identity = dict(source_bindings=bindings, config=cfg, torch=torch.__version__,
        numpy=np.__version__, sklearn=__import__('sklearn').__version__,
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        source_roles='four_design_exposed_sites_with_per_fit_exclusion')
    assert_current(identity)
    return cfg, data, views, predictions, mv, neural, refs, ma, ea, cuts, identity


def train(cfg, data, views, neural, refs, identity, args, beat):
    for key, meta in views.items():
        if args.view and args.view != key:
            continue
        ids, x, y, d, pr = training_data(meta, data)
        r = neural[key]
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        ref = torch.load(ROOT/refs[key]['checkpoint'], map_location='cpu', weights_only=False)
        check_reused_head(cp, r, settings=ref['settings'], seed=meta['seed'],
            input_hash=array_hash(ids, x, d), label_hash=array_hash(y), pr=pr,
            reference_draws=ref['draws'], held=meta['outer_site'])
        ti = dict(identity=identity, view=key, inputs_sha256=array_hash(ids, x, d),
            labels_sha256=array_hash(y), draws_sha256=array_hash(cp['draws']),
            neural_checkpoint_sha256=r['checkpoint_sha256'], training_sites=pr['training_sites'],
            producers=meta['training_producers'])
        folder = ROOT/cfg['output']/'trials'/key
        receipt = folder/'complete.json'
        if receipt.exists():
            motion.verified_receipt(receipt, ti)
            beat(state='cached_verified_forest', view=key)
            continue
        _, result = fit_forest(x, y, d, pr, cp['draws'], settings=cfg['forest'], seed=meta['seed'],
            identity=ti, directory=folder, resume=args.resume, stop_at=args.stop_at,
            heartbeat=lambda **v:beat(view=key, **v))
        if not result['complete']:
            beat(state='pilot_checkpoint_only_not_complete_matrix', view=key)
            return
        path = folder/'checkpoint.joblib'
        immutable_json(receipt, dict(identity=ti, checkpoint=str(path.relative_to(ROOT)),
            checkpoint_sha256=file_digest(path), fit=result, result_source='fresh_run'))
    assert_current(identity)
    beat(state='requested_training_complete')


def evaluate(cfg, data, views, predictions, mv, neural, ma, ea, cuts, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    forests = {}
    for key in views:
        path = root/'trials'/key/'complete.json'
        r = json.loads(path.read_text())
        assert r['identity']['identity'] == identity and r['fit']['trees'] == cfg['forest']['trees']
        motion.verified_receipt(path, r['identity'])
        forests[key] = r
    archives, cached, eqp, parent = [], {}, {}, {}
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
        old = next(r for r in ma['decision_archives'] if r['view'] == key)
        with np.load(ROOT/old['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); parent[key] = {k:z[k].copy() for k in z.files}
        old_eq = next(r for r in ea['archives'] if r['view'] == key)
        with np.load(ROOT/old_eq['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            expected = {k:z[k].copy() for k in ('refit_bounded_fraction', 'refit_bounded_fraction_strict_stop', 'refit_bounded_fraction_net_stop')}
        x, d, _ = features(data['geometry'][ids], p, data['scale'][ids])
        eqp[key] = p
        arrays = dict(ids=ids)
        for head in cfg['heads']:
            if head == 'neural':
                cp = torch.load(ROOT/neural[key]['checkpoint'], map_location='cpu', weights_only=False)
                assert cp['identity'] == neural[key]['identity']
                model = build(x.shape[1], cp['settings']['width'], meta['seed']); model.load_state_dict(cp['model'])
                score = predict(model, x, d, cp['preprocess'], 'bounded_fraction')
                np.testing.assert_array_equal(score, expected['refit_bounded_fraction'])
            else:
                cp = joblib.load(ROOT/forests[key]['checkpoint'])
                assert cp['identity'] == forests[key]['identity'] and cp['settings'] == cfg['forest']
                score = predict_forest(cp['model'], x, d, cp['preprocess'])
            assert meta['outer_site'] not in cp['preprocess']['training_sites']
            bits = protected_decisions(score, data['geometry'][ids, :16].reshape(-1, 8, 2), d)
            arrays[head+'_score'] = score
            for policy, selected in bits.items():
                arrays[head+'_'+policy] = selected
                if head == 'neural':
                    np.testing.assert_array_equal(selected, expected['refit_bounded_fraction_'+policy])
            for action in cfg['comparators']:
                old_score = parent[key][action+'__'+head+'__score']
                old_bits = parent[key][action+'__'+head+'__strict_stop']
                a, b = match_strict_count(score, old_score, bits['strict_stop'], old_bits, ids)
                arrays[action+'__'+head+'__matched_eqmotion'] = a
                arrays[action+'__'+head+'__matched_comparator'] = b
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists(): raise ValueError('Missing replay archive')
        write_arrays(path, arrays)
        cached[key] = arrays
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        beat(state='decisions_before_outcomes', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        all_forests_complete=True, neural_scores_and_choices_reproduce=True,
        target_arrays_loaded_for_decisions=False, cuts=cuts))
    n = len(data['sites'])
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale'])
    masks = dict(complete=valid.all(1), zero_CV=valid.all(1) & (cv == 0),
        positive_easy=np.zeros(n, bool), hard=np.zeros(n, bool))
    actions = ['eqmotion']+cfg['comparators']
    err, fde, lo, hi = ({(s,a):np.empty(n) for s in cfg['seeds'] for a in actions} for _ in range(4))
    for key, meta in views.items():
        ids, seed = cached[key]['ids'], meta['seed']
        masks['hard'][ids] = cv[ids] >= cuts[key]['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= cuts[key]['positive_easy_cut'])
        with np.load(ROOT/mv[key]['outer_prediction']['prediction']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); t = z['prediction'].copy()
        for action in actions:
            p = eqp[key] if action == 'eqmotion' else t if action == 'transformer' else causal_candidate(data['geometry'][ids], action)
            err[seed,action][ids], fde[seed,action][ids] = native_errors(p, y[ids], valid[ids], data['scale'][ids])
            bounds = partial_gain_bounds(p, baseline[ids], y[ids], valid[ids], data['scale'][ids])
            lo[seed,action][ids], hi[seed,action][ids] = bounds['lower'], bounds['upper']
    def metric(e, ref, mask=None, bootstrap=True):
        mask = np.ones(n, bool) if mask is None else mask
        return paired_scene_metrics(e[mask], ref[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'] if bootstrap else 0)
    def choices(action, head, policy, matched=None):
        choices = {s:np.zeros(n, bool) for s in cfg['seeds']}
        for key, meta in views.items():
            ids = cached[key]['ids']
            if matched:
                selected = cached[key][matched+'__'+head+'__matched_'+('eqmotion' if action == 'eqmotion' else 'comparator')]
            elif policy == 'uncontrolled': selected = np.ones(len(ids), bool)
            elif action == 'eqmotion': selected = cached[key][head+'_'+policy]
            else: selected = parent[key][action+'__'+head+'__'+policy]
            choices[meta['seed']][ids] = selected
        return choices
    def summarize(action, selected):
        errors, endpoints, seeds = [], [], {}
        for seed in cfg['seeds']:
            use = selected[seed]
            e, f = np.where(use, err[seed,action], cv), np.where(use, fde[seed,action], cf)
            errors.append(e); endpoints.append(f)
            seeds[str(seed)] = dict(ADE=metric(e, cv, bootstrap=False), FDE=metric(f, cf, bootstrap=False),
                subsets={k:metric(e, cv, m, bootstrap=False) for k,m in masks.items()},
                selected=int(use.sum()), selected_unknown=int((use & ~valid.any(1)).sum()),
                selected_incomplete=int((use & ~valid.all(1)).sum()), zero_CV_harmed=int((e[masks['zero_CV']] > 0).sum()),
                full_grid_gain_bounds={s:[float(np.where(use, b[seed,action], 0)[data['sites'] == s].mean()) for b in (lo, hi)] for s in cfg['sites']})
        mean = np.mean(errors, axis=0)
        return dict(ADE=metric(mean, cv), FDE=metric(np.mean(endpoints, axis=0), cf),
            subsets={k:metric(mean, cv, m) for k,m in masks.items()}, seeds=seeds,
            worst_site_seed_easy_degradation_percent=max(0., max(-r['subsets']['positive_easy']['worst_scene_gain_percent'] for r in seeds.values())))
    def contrast(a, b):
        out = {}
        for key in ('ADE', 'hard', 'positive_easy'):
            aa = a['ADE'] if key == 'ADE' else a['subsets'][key]
            bb = b['ADE'] if key == 'ADE' else b['subsets'][key]
            av = [aa['by_scene'][s]['gain_percent'] for s in cfg['sites']]
            bv = [bb['by_scene'][s]['gain_percent'] for s in cfg['sites']]
            out[key] = None if None in av+bv else paired_scene_contrast(av, bv, resamples=cfg['bootstrap_resamples'])
        return out
    summaries = {'eqmotion__uncontrolled':summarize('eqmotion', choices('eqmotion', 'none', 'uncontrolled'))}
    for head in cfg['heads']:
        for policy in cfg['policies']:
            summaries['eqmotion__'+head+'__'+policy] = summarize('eqmotion', choices('eqmotion', head, policy))
    comparisons, matched = {}, {}
    for action in cfg['comparators']:
        for head in cfg['heads']:
            key = action+'__'+head
            c = summarize(action, choices(action, head, 'strict_stop'))
            old = ma['summaries'][action+'__'+head+'__strict_stop']
            for field in ('ADE', 'FDE', 'subsets', 'seeds'):
                assert c[field] == old[field]
            comparisons[key] = dict(eqmotion_minus_comparator=contrast(summaries['eqmotion__'+head+'__strict_stop'], c), comparator=c)
            a = summarize('eqmotion', choices('eqmotion', head, 'strict_stop', matched=action))
            b = summarize(action, choices(action, head, 'strict_stop', matched=action))
            assert [v['selected'] for v in a['seeds'].values()] == [v['selected'] for v in b['seeds'].values()]
            matched[key] = dict(eqmotion=a, comparator=b, eqmotion_minus_comparator=contrast(a,b))
    report = dict(identity=identity, result_source='fresh_forests_and_common_comparison_cached_verified_forecasters_and_neural_heads',
        rows=n, complete_rows=int(valid.all(1).sum()), unknown_rows=int((~valid.any(1)).sum()),
        new_forests=12, reused_neural_heads=12, new_forecasters=0, summaries=summaries,
        comparisons=comparisons, matched_count=matched,
        neural_minus_forest=contrast(summaries['eqmotion__neural__strict_stop'], summaries['eqmotion__forest__strict_stop']),
        decision_archives=archives, fits=[dict(view=k, checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
            result_source='fresh_run', fit=r['fit']) for k,r in forests.items()],
        no_leakage_scope='per_fit_exclusion_and_past_indexed_access_not_online_annotation_provenance',
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(public/'analysis.json', report)
    render(cfg, report)
    if verify:
        immutable_json(public/'replay.json', dict(all_checks_passed=True, analysis_sha256=file_digest(public/'analysis.json'),
            views=len(views), rows=n, model_score_and_choice_replay_exact=True, aggregate_replay_exact=True))
    beat(state='replay_complete' if verify else 'evaluation_complete')


def render(cfg, result):
    lines = ['# Full EqMotion Protected Motion Controls', '',
        'Four design-exposed SDD sites, three seeds, observation8/prediction12 native annotation steps.',
        'Annotation pixels; no verified metric/seconds, independent confirmation or deployment.',
        'Twelve new forests; twelve cached-verified neural heads. No new forecasting models.', '',
        '| EqMotion policy | ADE gain % [CI95] | Hard gain % | Worst site/seed easy degradation % |',
        '|---|---:|---:|---:|']
    for name, r in result['summaries'].items():
        lines.append(f"| {name} | {r['ADE']['equal_scene_gain_percent']:.4f} {r['ADE']['scene_bootstrap_ci95']} | "
            f"{r['subsets']['hard']['equal_scene_gain_percent']:.4f} | {r['worst_site_seed_easy_degradation_percent']:.4f} |")
    lines += ['', '## Strict and Matched-Count Contrasts', '',
        'Positive difference favors EqMotion. All comparisons reported; no selected winner.', '',
        '| Comparator | Head | Strict gain difference pp [CI95] | Matched difference pp [CI95] | Matched EqMotion easy degradation % | Matched comparator easy degradation % |',
        '|---|---|---:|---:|---:|---:|']
    for key, r in result['comparisons'].items():
        a,h = key.split('__'); m = result['matched_count'][key]
        c, d = r['eqmotion_minus_comparator']['ADE'], m['eqmotion_minus_comparator']['ADE']
        lines.append(f"| {a} | {h} | {c['mean_gain_difference_pp']:.4f} {c['ci95_pp']} | "
            f"{d['mean_gain_difference_pp']:.4f} {d['ci95_pp']} | {m['eqmotion']['worst_site_seed_easy_degradation_percent']:.4f} | "
            f"{m['comparator']['worst_site_seed_easy_degradation_percent']:.4f} |")
    lines += ['', 'CIs are nominal four-site development bootstraps, not multiplicity-adjusted confirmation.',
        'Matched counts are outcome-blind offline diagnostics, not online safety certificates.',
        'Training-only easy/hard cutoffs are shared with the prior protected-motion study.',
        'Unknown outcomes, complete paths, zero-CV harms, tails and missing-label gain bounds remain in analysis.json.', '']
    path = ROOT/cfg['reports']/'results.md'; text = '\n'.join(lines)
    if path.exists() and path.read_text() != text: raise ValueError('Changed completed report')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    for name in ('audit-only', 'resume', 'evaluate', 'verify'): parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--view'); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if (sum((args.audit_only,args.evaluate,args.verify)) > 1 or
        ((args.audit_only or args.evaluate or args.verify) and (args.view or args.stop_at is not None or args.resume)) or
        (args.stop_at is not None and not args.view)):
        raise ValueError('Separate full-readout phases and explicitly scoped pilot required')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**values):
        row = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/'heartbeat.json',row)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    signal.signal(signal.SIGTERM, lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,data,views,predictions,mv,neural,refs,ma,ea,cuts,identity = load()
        if args.view and args.view not in views: raise ValueError('Unknown registered view')
        immutable_json(root/'identity.json',identity)
        if args.audit_only: beat(state='preflight_pass',views=len(views),bindings=len(identity['source_bindings']))
        elif args.evaluate or args.verify:
            evaluate(cfg,data,views,predictions,mv,neural,ma,ea,cuts,identity,beat,args.verify)
        else: train(cfg,data,views,neural,refs,identity,args,beat)


if __name__ == '__main__':
    main()
