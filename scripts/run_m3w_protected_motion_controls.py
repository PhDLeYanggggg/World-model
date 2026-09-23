"""Fixed source-only learned protection for all six causal motion alternatives."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from scripts.run_m3w_bounded_cost import load as parent_load, features, read_arrays, training_data
from scripts.run_m3w_native_joint_controls import view_scores
from scripts.run_m3w_native_forecast import array_hash, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_forecast_cost_bounds import bounded_fractions, partial_gain_bounds
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_protected_motion_controls import (
    ACTIONS, causal_candidate, protected_decisions, match_strict_count, supported_costs)
from src.world_model.m3w_native_gain_harm import preprocess
from src.world_model.m3w_bounded_cost_head import build, fit, predict
from src.training.m3w_external_cost_bank import fit_forest, predict_forest

CONFIG = 'configs/m3w_protected_motion_controls_v1.json'
CODE = ('scripts/run_m3w_protected_motion_controls.py',
        'src/evaluation/m3w_protected_motion_controls.py',
        'tests/test_m3w_protected_motion_controls.py',
        'src/training/m3w_external_cost_bank.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    parent, data, views, gain, previous = parent_load()
    if (cfg['actions'] != list(ACTIONS) or cfg['sites'] != parent['sites']
            or cfg['seeds'] != parent['seeds'] or cfg['training'] != parent['training']
            or cfg['heads'] != ['neural', 'forest'] or cfg['policies'] != ['net_stop', 'strict_stop']
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'external_readout',
                'independent_confirmation', 'risk_calibration', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only matched experiment required')
    bindings = dict(previous['source_bindings'])
    ap = Path(parent['reports'])/'analysis.json'
    assert file_digest(ROOT/ap) == cfg['reference_analysis_sha256']
    for p in (str(ap), CONFIG, cfg['registration'], *CODE):
        bindings[p] = file_digest(ROOT/p)
    references = {}
    for key in views:
        path = Path(parent['output'])/'trials'/key/'bounded_fraction'/'complete.json'
        r = json.loads((ROOT/path).read_text())
        assert (r['identity']['identity'] == previous and r['fit']['complete']
                and r['fit']['step'] == cfg['training']['steps'])
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        bindings[str(path)], bindings[r['checkpoint']] = file_digest(ROOT/path), r['checkpoint_sha256']
        references[key] = r
    identity = dict(source_bindings=bindings, config=cfg, runtime=previous['runtime'],
                    torch=torch.__version__, numpy=np.__version__, sklearn=__import__('sklearn').__version__,
                    parent_identity_sha256=file_digest(ROOT/parent['output']/'identity.json'))
    assert_current(identity)
    return cfg, data, views, gain, identity, references


def folder(cfg, key, action, head):
    return ROOT/cfg['output']/'trials'/key/action/head


def verified_receipt(path, identity):
    r = json.loads(path.read_text())
    if r['identity'] != identity or not r['fit']['complete'] or file_digest(ROOT/r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Changed or incomplete fit receipt')
    return r


def train(cfg, data, views, identity, refs, args, beat):
    for key, meta in views.items():
        if args.view and key != args.view:
            continue
        ids, ref_x, ref_y, ref_d, ref_pr = training_data(meta, data)
        reference = torch.load(ROOT/refs[key]['checkpoint'], map_location='cpu', weights_only=False)
        assert reference['identity'] == refs[key]['identity'] and reference['settings'] == cfg['training']
        assert reference['arm'] == 'bounded_fraction' and reference['seed'] == meta['seed']
        assert refs[key]['identity']['inputs_sha256'] == array_hash(ids, ref_x, ref_d)
        assert refs[key]['identity']['labels_sha256'] == array_hash(ref_y)
        for field in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(reference['preprocess'][field], ref_pr[field])
        assert reference['preprocess']['cost_scale'] == ref_pr['cost_scale']
        valid = read_arrays(data, ids, 'valid')
        target = read_arrays(data, ids, 'target')
        geometry, scale = data['geometry'][ids], data['scale'][ids]
        cv, _ = native_errors(geometry[:, 332:356].reshape(-1, 12, 2), target, valid, scale)
        for action in cfg['actions']:
            if args.action and action != args.action:
                continue
            if action == 'transformer':
                x, y, d, pr = ref_x, ref_y, ref_d, ref_pr
            else:
                candidate = causal_candidate(geometry, action)
                x, d, _ = features(geometry, candidate, scale)
                err, _ = native_errors(candidate, target, valid, scale)
                y, train_cv = supported_costs(err, cv, valid.all(1))
                bounded_fractions(y, d, valid.all(1))
                pr = preprocess(x, y, train_cv, data['sites'][ids], meta['outer_site'])
            np.testing.assert_array_equal(pr['known'], ref_pr['known'])
            ti = dict(experiment=identity, view=key, action=action,
                inputs_sha256=array_hash(ids, x, d), labels_sha256=array_hash(y),
                support_sha256=array_hash(pr['known']), training_sites=pr['training_sites'],
                producer='past_only_fixed_formula' if action != 'transformer' else meta['training_producers'])
            nn_dir = folder(cfg, key, action, 'neural')
            receipt_path = nn_dir/'complete.json'
            if receipt_path.exists():
                nn_r = verified_receipt(receipt_path, ti)
                beat(state='cached_verified_fit', view=key, action=action, head='neural')
            else:
                if action == 'transformer':
                    cp, result, source = ROOT/refs[key]['checkpoint'], refs[key]['fit'], 'cached_verified'
                else:
                    _, result = fit(x, y, d, data['sites'][ids], pr, arm='bounded_fraction', seed=meta['seed'],
                        settings=cfg['training'], identity=ti, directory=nn_dir, resume=args.resume,
                        stop_at=args.stop_at, heartbeat=lambda **v:beat(view=key, action=action, head='neural', **v))
                    if not result['complete']:
                        beat(state='pilot_checkpoint_complete_not_full_matrix', view=key, action=action)
                        return
                    cp, source = nn_dir/'checkpoint.pt', 'fresh_run'
                nn_r = dict(identity=ti, checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                            fit=result, result_source=source)
                immutable_json(receipt_path, nn_r)
            state = torch.load(ROOT/nn_r['checkpoint'], map_location='cpu', weights_only=False)
            np.testing.assert_array_equal(state['draws'], reference['draws'])
            assert state['draws'].sum() == cfg['training']['steps']*cfg['training']['batch_size']
            assert not state['draws'][~pr['known']].any()
            forest_dir = folder(cfg, key, action, 'forest')
            fp = forest_dir/'complete.json'
            if fp.exists():
                verified_receipt(fp, ti)
                beat(state='cached_verified_fit', view=key, action=action, head='forest')
                continue
            _, result = fit_forest(x, y, d, pr, state['draws'], settings=cfg['forest'], seed=meta['seed'],
                identity=ti, directory=forest_dir, resume=args.resume,
                heartbeat=lambda **v:beat(view=key, action=action, head='forest', **v))
            cp = forest_dir/'checkpoint.joblib'
            immutable_json(fp, dict(identity=ti, checkpoint=str(cp.relative_to(ROOT)),
                checkpoint_sha256=file_digest(cp), fit=result, result_source='fresh_run'))
    assert_current(identity)
    beat(state='requested_training_complete')


def evaluate(cfg, data, views, gain, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']
    # Require the entire fixed matrix, not just a convenient finished subset.
    fits = {}
    for key in views:
        for action in cfg['actions']:
            for head in cfg['heads']:
                r = json.loads((folder(cfg, key, action, head)/'complete.json').read_text())
                assert r['identity']['experiment'] == identity and r['identity']['action'] == action
                verified_receipt(folder(cfg, key, action, head)/'complete.json', r['identity'])
                fits[key, action, head] = r
    n = len(data['sites'])
    scores, bits, predictions, cuts, archives = {}, {}, {}, {}, []
    for key, meta in views.items():
        ids, transformer, _, _, _, pr = view_scores(key, meta, gain, data)
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        cuts[key] = dict(positive_easy_cut=pr['positive_easy_cut'], hard_cut=pr['hard_cut'])
        arrays = dict(ids=ids)
        geometry, scale = data['geometry'][ids], data['scale'][ids]
        for action in cfg['actions']:
            p = transformer if action == 'transformer' else causal_candidate(geometry, action)
            predictions[key, action] = p
            x, d, _ = features(geometry, p, scale)
            for head in cfg['heads']:
                r = fits[key, action, head]
                if head == 'neural':
                    cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
                    model = build(x.shape[1], cfg['training']['width'], meta['seed'])
                    model.load_state_dict(cp['model'])
                    score = predict(model, x, d, cp['preprocess'], 'bounded_fraction')
                else:
                    cp = joblib.load(ROOT/r['checkpoint'])
                    assert cp['identity'] == r['identity'] and cp['settings'] == cfg['forest']
                    score = predict_forest(cp['model'], x, d, cp['preprocess'])
                assert meta['outer_site'] not in cp['preprocess']['training_sites']
                policy = protected_decisions(score, geometry[:, :16].reshape(-1, 8, 2), d)
                scores[key, action, head] = score
                for name, selected in policy.items():
                    bits[key, action, head, name] = selected
                    arrays[action+'__'+head+'__'+name] = selected
                arrays[action+'__'+head+'__score'] = score
        for action in cfg['actions'][:-1]:
            for head in cfg['heads']:
                a, b = match_strict_count(scores[key, 'transformer', head], scores[key, action, head],
                    bits[key, 'transformer', head, 'strict_stop'], bits[key, action, head, 'strict_stop'], ids)
                bits[key, action, head, 'matched_transformer'] = a
                bits[key, action, head, 'matched_causal'] = b
                arrays[action+'__'+head+'__matched_transformer'] = a
                arrays[action+'__'+head+'__matched_causal'] = b
        path = root/'decisions'/f'{key}.npz'
        if verify and not path.exists():
            raise ValueError('Cannot replay missing decisions')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        beat(state='decisions_saved_before_outcomes', view=key)
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        target_arrays_loaded_for_decisions=False, all_fits_complete=True, cuts=cuts))
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = native_errors(b, y, valid, data['scale'])
    masks = dict(complete=valid.all(1), zero_CV=valid.all(1) & (cv == 0),
                 positive_easy=np.zeros(n, bool), hard=np.zeros(n, bool))
    err, end, low, high = ({(s, a):np.full(n, np.nan) for s in cfg['seeds'] for a in cfg['actions']} for _ in range(4))
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site'])
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= cuts[key]['positive_easy_cut'])
        masks['hard'][ids] = cv[ids] >= cuts[key]['hard_cut']
        for action in cfg['actions']:
            p, idx = predictions[key, action], (meta['seed'], action)
            err[idx][ids], end[idx][ids] = native_errors(p, y[ids], valid[ids], data['scale'][ids])
            bounds = partial_gain_bounds(p, b[ids], y[ids], valid[ids], data['scale'][ids])
            low[idx][ids], high[idx][ids] = bounds['lower'], bounds['upper']
    def metric(e, ref, mask=None, bootstrap=True):
        m = np.ones(n, bool) if mask is None else mask
        return paired_scene_metrics(e[m], ref[m], data['sites'][m], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'] if bootstrap else 0)
    all_errors, summaries = {}, {}
    for action in cfg['actions']:
        for head, policy in [('none', 'uncontrolled')]+[(h, p) for h in cfg['heads'] for p in cfg['policies']]:
            name = '__'.join((action, head, policy))
            by_seed, errors, ends = {}, [], []
            for seed in cfg['seeds']:
                chosen = np.ones(n, bool) if head == 'none' else np.zeros(n, bool)
                if head != 'none':
                    for key, meta in views.items():
                        if meta['seed'] == seed:
                            chosen[data['sites'] == meta['outer_site']] = bits[key, action, head, policy]
                e, f = np.where(chosen, err[seed, action], cv), np.where(chosen, end[seed, action], cf)
                errors.append(e); ends.append(f)
                by_seed[str(seed)] = dict(ADE=metric(e, cv, bootstrap=False), FDE=metric(f, cf, bootstrap=False),
                    subsets={k:metric(e, cv, m, bootstrap=False) for k,m in masks.items()},
                    selected=int(chosen.sum()), selected_unknown=int((chosen & ~valid.any(1)).sum()),
                    selected_incomplete=int((chosen & ~valid.all(1)).sum()),
                    zero_CV_harmed=int((e[masks['zero_CV']] > 0).sum()),
                    full_grid_gain_bounds={site:[float(np.where(chosen, bound[seed, action], 0)[data['sites'] == site].mean())
                        for bound in (low, high)] for site in cfg['sites']})
            mean = np.mean(errors, axis=0)
            all_errors[name] = errors
            summaries[name] = dict(ADE=metric(mean, cv), FDE=metric(np.mean(ends, axis=0), cf),
                subsets={k:metric(mean, cv, m) for k,m in masks.items()}, seeds=by_seed,
                seed_gain_std=float(np.std([r['ADE']['equal_scene_gain_percent'] for r in by_seed.values()], ddof=1)))
    def contrast(left, right, mask=None):
        a, c = metric(left, cv, mask), metric(right, cv, mask)
        av = [a['by_scene'][s]['gain_percent'] for s in cfg['sites']]
        bv = [c['by_scene'][s]['gain_percent'] for s in cfg['sites']]
        return None if None in av+bv else paired_scene_contrast(av, bv, resamples=cfg['bootstrap_resamples'])
    contrasts, matched = {}, {}
    for action in cfg['actions'][:-1]:
        for head in cfg['heads']:
            a = np.mean(all_errors['transformer__'+head+'__strict_stop'], axis=0)
            c = np.mean(all_errors[action+'__'+head+'__strict_stop'], axis=0)
            name = 'transformer_minus_'+action+'__'+head
            contrasts[name] = dict(ADE=contrast(a, c), hard=contrast(a, c, masks['hard']),
                                   positive_easy=contrast(a, c, masks['positive_easy']))
            aa, cc, counts = [], [], []
            for seed in cfg['seeds']:
                left, right = np.zeros(n, bool), np.zeros(n, bool)
                for key, meta in views.items():
                    if meta['seed'] == seed:
                        m = data['sites'] == meta['outer_site']
                        left[m] = bits[key, action, head, 'matched_transformer']
                        right[m] = bits[key, action, head, 'matched_causal']
                assert left.sum() == right.sum()
                counts.append(int(left.sum()))
                aa.append(np.where(left, err[seed, 'transformer'], cv))
                cc.append(np.where(right, err[seed, action], cv))
            ma, mc = np.mean(aa, axis=0), np.mean(cc, axis=0)
            matched[name] = dict(counts=counts, transformer=metric(ma, cv), causal=metric(mc, cv),
                ADE=contrast(ma, mc), hard=contrast(ma, mc, masks['hard']),
                positive_easy=contrast(ma, mc, masks['positive_easy']))
    head_contrasts = {a:contrast(np.mean(all_errors[a+'__neural__strict_stop'], axis=0),
                                 np.mean(all_errors[a+'__forest__strict_stop'], axis=0)) for a in cfg['actions']}
    report = dict(identity=identity, status='fresh_run_complete_source_development_matrix',
        rows=n, sites=cfg['sites'], seeds=cfg['seeds'], complete_rows=int(valid.all(1).sum()),
        unknown_rows=int((~valid.any(1)).sum()), summaries=summaries,
        transformer_minus_each_causal=contrasts, matched_strict_count=matched,
        neural_minus_forest=head_contrasts, decision_archives=archives,
        fits=[dict(view=k, action=a, head=h, result_source=r['result_source'],
            checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
            seconds=r['fit']['seconds'], fit_budget=r['fit'].get('step', r['fit'].get('trees')))
            for (k,a,h),r in fits.items()],
        restrictions=dict(independent_confirmation=False, external_readout=False, deployment=False,
                          stage5c_executed=False, smc_enabled=False))
    assert_current(identity)
    immutable_json(public/'analysis.json', report)
    render(cfg, report)
    if verify:
        json_write(public/'verification.json', dict(all_checks_passed=True, decision_replay_exact=True,
            analysis_replay_exact=True, views=len(views), rows=n,
            analysis_sha256=file_digest(public/'analysis.json')))
    beat(state='evaluation_replay_complete' if verify else 'evaluation_complete', rows=n)


def render(cfg, result):
    lines = ['# Protected Causal Motion Controls', '',
        'Fresh source-development readout; no external selection or deployment. Twelve cached-verified',
        'Transformer heads; 72 fresh causal heads; 84 fresh matched forests. Native annotation pixels,',
        '8/12 native steps. Four previously exposed sites. CIs bootstrap sites, not overlapping windows.', '',
        '| Action | Head | Strict ADE gain % [CI95] | Hard gain % | Worst site/seed easy degradation % | Switch % |',
        '|---|---|---:|---:|---:|---:|']
    for action in cfg['actions']:
        for head in cfg['heads']:
            v = result['summaries'][action+'__'+head+'__strict_stop']
            worst = max(-q['subsets']['positive_easy']['worst_scene_gain_percent'] for q in v['seeds'].values())
            selected = np.mean([q['selected'] for q in v['seeds'].values()])/result['rows']*100
            ci = v['ADE']['scene_bootstrap_ci95']
            lines.append(f"| {action} | {head} | {v['ADE']['equal_scene_gain_percent']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"{v['subsets']['hard']['equal_scene_gain_percent']:.4f} | {max(0., worst):.4f} | {selected:.3f} |")
    lines += ['', '## Paired Incremental Forecast Contribution', '',
        'Positive means full Transformer improves more than the protected simple action.',
        'No best arm is selected from this table. Small four-site intervals are exploratory.', '',
        '| Simple action | Head | Transformer minus action pp [CI95] | Matched-count pp [CI95] |',
        '|---|---|---:|---:|']
    for action in cfg['actions'][:-1]:
        for head in cfg['heads']:
            key = 'transformer_minus_'+action+'__'+head
            a, m = result['transformer_minus_each_causal'][key]['ADE'], result['matched_strict_count'][key]['ADE']
            lines.append(f"| {action} | {head} | {a['mean_gain_difference_pp']:.4f} {a['ci95_pp']} | {m['mean_gain_difference_pp']:.4f} {m['ci95_pp']} |")
    lines += ['', '## Boundaries', '',
        '- Equal mean scene-relative available-label ADE; mean seed errors are not an ensemble forecast.',
        '- Complete, partial, zero-CV, easy/hard, FDE, tails, counts and missing-label bounds are in analysis.json.',
        '- Easy mean preservation does not establish individual or population safety.',
        '- Cost learner is neural even for a simple action; this separates forecasting from routing contribution.',
        '- No images, goal contribution, joint-scene contribution, metric calibration or seconds claim.',
        '- DUT is not re-evaluated; DroneCrowd remains closed; nested EqMotion controls are not_run here.',
        '- Existing deployment is unchanged. Stage5C execution and SMC remain disabled.', '']
    path = ROOT/cfg['reports']/'results.md'
    text = '\n'.join(lines)
    if path.exists() and path.read_text() != text:
        raise ValueError('Changed frozen report')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    for option in ('audit-only', 'evaluate', 'verify', 'resume'):
        parser.add_argument('--'+option, action='store_true')
    parser.add_argument('--view')
    parser.add_argument('--action', choices=ACTIONS)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if sum((args.audit_only, args.evaluate, args.verify)) > 1:
        raise ValueError('One phase per invocation')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg, data, views, gain, identity, refs = load()
    if args.view and args.view not in views:
        raise ValueError('Unknown fixed view')
    root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    immutable_json(root/'identity.json', identity)
    def beat(**values):
        row = dict(pid=os.getpid(), updated_unix=time.time(), **values)
        json_write(root/'heartbeat.json', row)
        print(json.dumps(row), flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.audit_only:
            beat(state='preflight_pass', rows=len(data['sites']), bindings=len(identity['source_bindings']))
        elif args.evaluate or args.verify:
            evaluate(cfg, data, views, gain, identity, beat, args.verify)
        else:
            train(cfg, data, views, identity, refs, args, beat)


if __name__ == '__main__':
    main()
