"""Frozen-budget score-ranking controls; no refitting or held-out threshold search."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_forest_cost as parent
from scripts.run_m3w_native_forecast import assert_current, immutable_json, json_write
from scripts.run_m3w_native_nested import write_arrays
from scripts.run_m3w_bounded_cost import read_arrays
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_risk_ranking import choices, overlap, POLICIES, MATCHED
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.evaluation.m3w_forecast_cost_bounds import partial_gain_bounds
from src.evaluation.m3w_temporal_fit_support import population_summary
from src.evaluation.m3w_fixed_choice_context import audit as context_audit
from src.world_model.m3w_temporal_intervention import candidates, second_difference
import numpy as np
import torch

CONFIG = 'configs/m3w_risk_ranking_v1.json'
CODE = ('scripts/run_m3w_risk_ranking.py', 'src/evaluation/m3w_risk_ranking.py',
        'tests/test_m3w_risk_ranking.py')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    pc, old, data, views, predictions, refs, previous, prior, pid = parent.load()
    records, states = parent.load_states(pc, views, pid)
    public = ROOT/pc['reports']; report = json.loads((public/'analysis.json').read_text())
    if (report['identity'] != pid or cfg['sites'] != pc['sites'] or cfg['seeds'] != pc['seeds']
            or cfg['arm'] != 'ramp' or tuple(cfg['policies']) != POLICIES
            or any(cfg[k] for k in ('threshold_search', 'model_selection', 'risk_calibration',
                'closed_role_readout', 'independent_confirmation', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed source-only diagnostic contract required')
    bindings = dict(pid['source_bindings'])
    def bind(path, expected=None):
        actual = file_digest(ROOT/path)
        if (expected is not None and actual != expected) or (path in bindings and bindings[path] != actual):
            raise ValueError('Changed bound artifact: '+path)
        bindings[path] = actual
    for path in (CONFIG, cfg['registration'], *CODE): bind(path)
    for name in ('analysis.json', 'replay.json', 'separate_verification.json', 'fit_held_diagnosis.json'):
        bind(str((public/name).relative_to(ROOT)))
    digest = file_digest(public/'analysis.json')
    for name in ('replay.json', 'separate_verification.json'):
        r = json.loads((public/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256'] == digest
    assert json.loads((public/'fit_held_diagnosis.json').read_text())['analysis_sha256'] == digest
    for r in records.values(): bind(r['checkpoint'], r['checkpoint_sha256'])
    for r in report['archives']: bind(r['path'], r['sha256'])
    bind(str((ROOT/pc['output']/'decisions_complete.json').relative_to(ROOT)), report['decision_manifest_sha256'])
    identity = dict(source_bindings=bindings, config=cfg, parent_analysis_sha256=digest,
        architecture=platform.machine(), no_new_fit=True, all_sites_development_exposed=True)
    assert_current(identity)
    return cfg, data, views, predictions, states, report, identity


def descriptive_checks(summary):
    return dict(each_seed_positive_cv=all(s['ADE']['equal_scene_gain_percent'] > 0 for s in summary['seeds'].values()),
        aggregate_easy=all(s['subsets']['positive_easy']['equal_scene_gain_percent'] >= -2 for s in summary['seeds'].values()),
        each_scene_seed_easy=all(r['gain_percent'] >= -2 for s in summary['seeds'].values()
            for r in s['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=not any(s['zero_CV_harmed'] for s in summary['seeds'].values()))


def evaluate(cfg, data, views, predictions, states, previous, identity, beat, verify=False):
    root, public = ROOT/cfg['output'], ROOT/cfg['reports']; n = len(data['sites'])
    selected = {s: {p: np.zeros(n, bool) for p in POLICIES} for s in cfg['seeds']}
    proposals = {s: {'ramp': np.empty((n, 12, 2))} for s in cfg['seeds']}
    distances = {s: np.empty(n) for s in cfg['seeds']}
    scores, archives, overlaps = {}, [], []
    # Freeze every decision using only existing causal scores/history before label readout.
    for key, meta in views.items():
        archive = next(r for r in previous['archives'] if r['view'] == key)
        with np.load(ROOT/archive['path'], allow_pickle=False) as z:
            old = {k: z[k].copy() for k in z.files}
        ids = old['ids']; np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == meta['outer_site']))
        with np.load(ROOT/predictions[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); p = z['prediction'].copy()
        base = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        proposal = candidates(base, p)[0]['ramp']
        past = data['geometry'][ids, :16].reshape(-1, 8, 2)
        f, nn, d = (old['ramp_'+v] for v in ('forest_score', 'neural_score', 'distance'))
        bits = choices(f, nn, past, d, ids, old['ramp_forest'], old['ramp_neural'])
        np.testing.assert_array_equal(bits['neural_gain'], old['ramp_neural_matched'])
        arrays = dict(ids=ids, **bits)
        path = root/'decisions'/(key+'.npz')
        if verify and not path.exists(): raise ValueError('Missing decisions cannot be replayed')
        write_arrays(path, arrays)
        archives.append(dict(view=key, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
        for name, use in bits.items(): selected[meta['seed']][name][ids] = use
        proposals[meta['seed']]['ramp'][ids] = proposal; distances[meta['seed']][ids] = d
        scores[key] = f, nn
        pairs = (('forest_ratio', 'neural_ratio'), ('neural_ratio', 'neural_gain'),
                 ('forest_ratio', 'forest_gain'), ('neural_ratio', 'neural_strict'))
        for a, b in pairs:
            overlaps.append(dict(view=key, policy_a=a, policy_b=b, counts=overlap(bits[a], bits[b])))
        beat(state='decisions_replayed' if verify else 'decisions_frozen', view=key,
             count=int(bits['forest_strict'].sum()))
    immutable_json(root/'decisions_complete.json', dict(identity=identity, archives=archives,
        future_targets_used=False, future_masks_used=False, score_models_refitted=False))
    interaction = context_audit(cfg, data, proposals, selected, {p: 'ramp' for p in POLICIES}, beat)
    y, valid = read_arrays(data, np.arange(n), 'target'), read_arrays(data, np.arange(n), 'valid')
    baseline = data['geometry'][:, 332:356].reshape(n, 12, 2)
    past = data['geometry'][:, :16].reshape(n, 8, 2)
    cv, cf = native_errors(baseline, y, valid, data['scale']); full = valid.all(1)
    masks = dict(complete=full, zero_CV=full & (cv == 0), hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ix = data['sites'] == meta['outer_site']; pr = states[key, 'ramp']['preprocess']
        masks['hard'][ix] = cv[ix] >= pr['hard_cut']
        masks['positive_easy'][ix] = (cv[ix] > 0) & (cv[ix] <= pr['positive_easy_cut'])
    errors, bounds, smooth = {}, {}, {}
    cv_smooth = second_difference(past, baseline, data['scale'])
    for seed in cfg['seeds']:
        p = proposals[seed]['ramp']
        errors[seed] = native_errors(p, y, valid, data['scale'])
        bounds[seed] = partial_gain_bounds(p, baseline, y, valid, data['scale'])
        smooth[seed] = second_difference(past, p, data['scale'])
    def metric(a, r, mask=None):
        if mask is None: mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], r[mask], data['sites'][mask], expected_scenes=cfg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=cfg['bootstrap_resamples'])
    summaries = {}
    for name in POLICIES:
        seeds, ades, fdes = {}, [], []
        for seed in cfg['seeds']:
            use = selected[seed][name]
            ade, fde = np.where(use, errors[seed][0], cv), np.where(use, errors[seed][1], cf)
            sm = np.where(use, smooth[seed], cv_smooth); ades.append(ade); fdes.append(fde)
            seeds[str(seed)] = dict(ADE=metric(ade, cv), FDE=metric(fde, cf), selected=int(use.sum()),
                selected_unknown=int((use & ~valid.any(1)).sum()), selected_incomplete=int((use & ~full).sum()),
                zero_CV_harmed=int((ade[masks['zero_CV']] > 0).sum()),
                full_grid_absolute_gain_bounds={s: [float(np.where(use, bounds[seed][k], 0)[data['sites']==s].mean())
                    for k in ('lower', 'upper')] for s in cfg['sites']},
                causal_displacement_sum={s: float(np.where(use, distances[seed], 0)[data['sites']==s].sum()) for s in cfg['sites']},
                smoothness={s: dict(mean=float(sm[data['sites']==s].mean()), p95=float(np.quantile(sm[data['sites']==s], .95)),
                    cv_mean=float(cv_smooth[data['sites']==s].mean())) for s in cfg['sites']},
                subsets={g: metric(ade, cv, m) for g, m in masks.items()})
        summaries[name] = dict(ADE=metric(np.mean(ades, 0), cv), FDE=metric(np.mean(fdes, 0), cf), seeds=seeds,
            subsets={g: metric(np.mean(ades, 0), cv, m) for g, m in masks.items()})
        mapping = dict(forest_strict='ramp_forest', neural_strict='ramp_neural', neural_gain='ramp_neural_matched',
                       forest_ratio='ramp_forest')
        if name in mapping:
            assert summaries[name] == previous['summaries'][mapping[name]]
        beat(state='metrics_complete', policy=name)
    def contrast(a, b):
        return paired_scene_contrast([summaries[a]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']],
                                     [summaries[b]['ADE']['by_scene'][s]['gain_percent'] for s in cfg['sites']])
    contrasts = {a+'_minus_'+b: contrast(a, b) for a, b in
        (('forest_strict', 'neural_ratio'), ('neural_ratio', 'neural_gain'), ('forest_ratio', 'forest_gain'))}
    quality, differences = [], []
    for key, meta in views.items():
        ids = np.flatnonzero(data['sites'] == meta['outer_site']); seed = meta['seed']
        delta = cv[ids]-errors[seed][0][ids]
        target = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0))); target[~full[ids]] = np.nan
        for estimator, score in zip(('forest', 'neural'), scores[key]):
            for name in MATCHED:
                use = selected[seed][name][ids]
                quality.append(dict(view=key, estimator=estimator, chosen_by=name,
                    costs=population_summary(target, score, distances[seed][ids], use, np.ones(len(ids)))))
        for a, b in (('forest_ratio', 'neural_ratio'), ('neural_ratio', 'neural_gain'), ('forest_ratio', 'forest_gain')):
            ma, mb = selected[seed][a][ids], selected[seed][b][ids]
            regions = {}
            for label, m in dict(common=ma & mb, a_only=ma & ~mb, b_only=mb & ~ma).items():
                known = m & full[ids]; easy = m & masks['positive_easy'][ids]; observed = m & np.isfinite(delta)
                regions[label] = dict(rows=int(m.sum()), complete=int(known.sum()),
                    incomplete=int((m & ~full[ids]).sum()), unknown=int((m & ~valid[ids].any(1)).sum()),
                    complete_benefit_sum=float(target[known, 0].sum()), complete_harm_sum=float(target[known, 1].sum()),
                    observed_gain_sum=float(delta[observed].sum()), observed_easy_rows=int(easy.sum()),
                    observed_easy_gain_sum=float(delta[easy].sum()),
                    complete_harm_event_count=int((target[known, 1] > 0).sum()))
            differences.append(dict(view=key, policy_a=a, policy_b=b, regions=regions))
    result = dict(identity=identity, result_source='fresh_fixed_ranking_diagnostic',
        reused_scores_and_forecasts='cached_verified_no_refit', summaries=summaries, contrasts=contrasts,
        overlaps=overlaps, turnover=differences, conditional_quality=quality, interaction=interaction, archives=archives,
        descriptive_checks={name: descriptive_checks(r) for name, r in summaries.items()},
        decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
        original_primary_superiority_gate_unchanged=previous['primary_gates'],
        independent_confirmation=False, risk_calibrated=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'replay.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            all_checks_passed=True, fixed_decisions_replayed=len(views)*len(POLICIES), new_model_fit=False))
    beat(state='verified' if verify else 'evaluated', descriptive_checks=result['descriptive_checks'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--audit-only', action='store_true'); group.add_argument('--verify', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg = json.loads((ROOT/CONFIG).read_text()); root = ROOT/cfg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**row):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **row)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, data, views, predictions, states, previous, identity = load()
        immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', source_bindings=len(identity['source_bindings']), no_new_fit=True)
        else:
            evaluate(cfg, data, views, predictions, states, previous, identity, beat, args.verify)


if __name__ == '__main__': main()
