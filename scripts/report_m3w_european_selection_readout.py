"""Summarize every frozen selection view without choosing a favorable seed."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_selection_readout as run
from scripts.report_m3w_european_floor_relative import dump
from src.world_model.m3w_frozen_selection import POLICIES


def aggregate_seeds(rows, policy, field='ADE_vs_incumbent', subset='all'):
    if sorted(r['metadata']['seed'] for r in rows) != [17, 29, 43]:
        raise ValueError('Exactly the three registered seeds required')
    metrics = [r['views'][policy][field][subset] for r in rows]
    roster = metrics[0]['expected_scenes']
    if any(m['expected_scenes'] != roster for m in metrics): raise ValueError('Scene mismatch')
    values = [[m['by_scene'][s]['gain_percent'] for m in metrics] for s in roster]
    if any(v is None for vv in values for v in vv):
        return dict(mean=None, ci=None, by_scene=None, seeds=3, localities=len(roster), status='undefined_support')
    means = np.mean(values, axis=1)
    draws = np.random.default_rng(39271).choice(means, size=(3000, len(means))).mean(1)
    return dict(mean=float(means.mean()), ci=np.quantile(draws, [.025, .975]).tolist(),
        by_scene=dict(zip(roster, means.tolist())), seeds=3, localities=len(roster),
        status='conditional_model_selection', estimator='mean_seedwise_relative_gain_not_trajectory_ensemble')


def summarize(rows):
    if len(rows) != 36: raise ValueError('Complete frozen family required')
    groups = {}
    for r in rows:
        m = r['metadata']; key = f"producer{m['fold']}_controller{m['controller']}_{m['event']}"
        groups.setdefault(key, []).append(r)
    result, seedwise = {}, {}
    for p in POLICIES:
        result[p] = {k: {s: aggregate_seeds(rr, p, subset=s) for s in ('all', 'hard', 'easy')}
                     for k, rr in sorted(groups.items())}
        ss = [r['summary'][p] for r in rows]
        def bounds(key):
            vv = [s[key] for s in ss if s[key] is not None]
            return [min(vv), max(vv)] if vv else None
        seedwise[p] = dict(gain_range=bounds('gain_vs_incumbent'), hard_range=bounds('hard_gain_vs_incumbent'),
            training_selected_gain_range=bounds('gain_vs_training_selected'),
            easy_degradation_range=bounds('worst_easy_degradation'), easy_pass_views=sum(s['easy_pass'] for s in ss),
            zero_CV_harm_views=sum(s['zero_CV_harmed_rows'] > 0 for s in ss), switch_range=bounds('switch_rate'),
            positive_CI=sum(s['CI'] is not None and s['CI'][0] > 0 for s in ss),
            negative_CI=sum(s['CI'] is not None and s['CI'][1] < 0 for s in ss))
    return dict(seed_averaged=result, seedwise=seedwise, no_candidate_selected=True,
        no_new_training=True, no_calibration_or_confirmation=True,
        independent_scene_groups=6, dependent_model_views=36,
        confidence_intervals_not_multiplicity_adjusted=True)


def format_range(v):
    return 'undefined' if v is None else f'{v[0]:+.6f}% to {v[1]:+.6f}%'


def main():
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['all_passed']
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        r = json.loads((ROOT/ref['path']).read_text()); assert r['verified']; rows.append(r)
    summary = summarize(rows); dump(run.PUBLIC/'aggregate_metrics.json', summary)
    lines = ['# Frozen Six-Locality Readout', '',
        'Fresh inference/readout of frozen, hash-verified checkpoints; no new training.',
        'Model-selection evidence, not final confirmation. All 36 dependent views share the same six localities.', '',
        '| Policy | All ADE gain vs old incumbent | Hard gain | Worst-locality easy degradation range | Easy-pass views | Positive/negative CIs |',
        '|---|---:|---:|---:|---:|---:|']
    for p, r in summary['seedwise'].items():
        lines.append(f"| {p} | {format_range(r['gain_range'])} | {format_range(r['hard_range'])} | {format_range(r['easy_degradation_range'])} | {r['easy_pass_views']}/36 | {r['positive_CI']}/{r['negative_CI']} |")
    lines += ['', 'A positive average is not an easy-risk certificate. Exact-zero CV harm is assessed separately.',
        'CIs are paired locality bootstrap (3,000 draws), exploratory and not multiplicity-adjusted.',
        'The table reports ranges over all source/controller/event/seed views, not independent replications.',
        'No favorable seed or candidate is selected. Calibration and confirmation remain closed.', '',
        '## Three-Seed Means', '',
        'These estimates average seedwise relative gains within each locality before bootstrapping six localities.',
        'They are not the performance of an averaged-trajectory ensemble.', '',
        '| Policy | Producer/controller/event | All gain | 95% locality CI | Hard gain |',
        '|---|---|---:|---:|---:|']
    for p, groups in summary['seed_averaged'].items():
        for k, r in groups.items():
            a, h = r['all'], r['hard']
            gain = 'undefined' if a['mean'] is None else f"{a['mean']:+.6f}%"
            hard = 'undefined' if h['mean'] is None else f"{h['mean']:+.6f}%"
            lines.append(f"| {p} | {k} | {gain} | {format_range(a['ci'])} | {hard} |")
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    risk_rows = []
    for r in rows:
        for p, view in r['views'].items():
            for site, risk in view['risk_reliability'].items():
                risk_rows.append(dict(group=r['group'], policy=p, site=site, **risk))
    dump(run.PUBLIC/'risk_reliability.json', dict(rows=risk_rows, purpose='diagnosis_only_no_recalibration'))
    print(json.dumps(summary['seedwise'], indent=2))


if __name__ == '__main__': main()
