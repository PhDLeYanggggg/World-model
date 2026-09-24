"""Compact verified source evidence without fitting or selecting a policy."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/protected_motion_controls_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT/'configs/m3w_protected_motion_controls_v1.json').read_text())
    ap = PUBLIC/'analysis.json'
    analysis = json.loads(ap.read_text())
    replay = json.loads((PUBLIC/'verification.json').read_text())
    independent = json.loads((PUBLIC/'independent_verification.json').read_text())
    for evidence in (replay, independent):
        if not evidence['all_checks_passed'] or evidence['analysis_sha256'] != digest(ap):
            raise ValueError('Verified current analysis required')
    if independent['verifier_sha256'] != digest(ROOT/'scripts/verify_m3w_protected_motion_controls.py'):
        raise ValueError('Verifier source changed')
    fits = analysis['fits']
    fresh = [r for r in fits if r['result_source'] == 'fresh_run']
    cached = [r for r in fits if r['result_source'] == 'cached_verified']
    assert len(fresh) == 156 and len(cached) == 12 and len(fits) == 168
    records = []
    for action in cfg['actions']:
        for head in cfg['heads']:
            v = analysis['summaries'][action+'__'+head+'__strict_stop']
            seeds = list(v['seeds'].values())
            row = dict(action=action, cost_head=head, ADE_gain_percent=v['ADE']['equal_scene_gain_percent'],
                ADE_ci95_low=v['ADE']['scene_bootstrap_ci95'][0], ADE_ci95_high=v['ADE']['scene_bootstrap_ci95'][1],
                FDE_gain_percent=v['FDE']['equal_scene_gain_percent'],
                complete_ADE_gain_percent=v['subsets']['complete']['equal_scene_gain_percent'],
                hard_ADE_gain_percent=v['subsets']['hard']['equal_scene_gain_percent'],
                worst_site_seed_easy_degradation_percent=max(0., max(-s['subsets']['positive_easy']['worst_scene_gain_percent'] for s in seeds)),
                switch_percent=100*sum(s['selected'] for s in seeds)/len(seeds)/analysis['rows'],
                zero_CV_harmed_rows_repeated=sum(s['zero_CV_harmed'] for s in seeds),
                seed_gain_std=v['seed_gain_std'])
            records.append(row)
    with (PUBLIC/'strict_controls.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = dict(status='complete_verified_source_development_not_confirmation',
        analysis_sha256=digest(ap), replay_sha256=digest(PUBLIC/'verification.json'),
        independent_verification_sha256=digest(PUBLIC/'independent_verification.json'),
        fresh_fits=156, fresh_neural_heads=72, fresh_forests=84, cached_verified_neural_heads=12,
        fitting_seconds_fresh=sum(r['seconds'] for r in fresh),
        fitting_time_excludes_loading_interfit_overhead_and_wall_clock_interruptions=True,
        rows=analysis['rows'], complete_rows=analysis['complete_rows'], unknown_rows=analysis['unknown_rows'],
        physical_sites=4, seeds=cfg['seeds'], strict_controls=records,
        source_forecasting_superiority_over_protected_damping_established=False,
        new_deployment=False, external_readout=False, independent_confirmation=False,
        stage5c_executed=False, smc_enabled=False)
    (PUBLIC/'evidence_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Training Loss Records', '',
        'Fixed-budget fits, not convergence certificates. Neural values are first/last logged',
        'minibatch bounded-fraction MSE; forests use weighted fitting-set MSE. Different action',
        'targets and loss-estimation samples make cross-row training-loss rankings inappropriate.',
        'All neural budgets are 3,000 updates and all forests contain 128 trees.', '',
        '| View | Action | Head | Result source | First loss | Last loss | Fit seconds |',
        '|---|---|---|---|---:|---:|---:|']
    for row in fits:
        path = ROOT/cfg['output']/'trials'/row['view']/row['action']/row['head']/'complete.json'
        r = json.loads(path.read_text())
        key = 'loss' if row['head'] == 'neural' else 'fitting_fraction_MSE'
        trace = r['fit']['trace']
        lines.append(f"| {row['view']} | {row['action']} | {row['head']} | {row['result_source']} | "
            f"{trace[0][key]:.6f} | {trace[-1][key]:.6f} | {row['seconds']:.3f} |")
    (PUBLIC/'training_losses.md').write_text('\n'.join(lines)+'\n')
    lines = ['# Matched-Count Safety Supplement', '',
        'Completes the registered subset analysis of the already frozen count-matched choices.',
        'No thresholds or selections change. T = Transformer; C = the indicated causal action.',
        'The same smaller strict count is used per outer fold/seed, not a deployable online budget.', '',
        '| Action | Head | T gain % | C gain % | T worst easy degradation % | C worst easy degradation % |',
        '|---|---|---:|---:|---:|---:|']
    for key, v in independent['matched_count_safety_supplement'].items():
        action, head = key.split('__'); t, c = v['transformer'], v['causal']
        lines.append(f"| {action} | {head} | {t['mean_equal_scene_ADE_gain_percent']:.4f} | "
            f"{c['mean_equal_scene_ADE_gain_percent']:.4f} | {t['worst_site_seed_easy_degradation_percent']:.4f} | "
            f"{c['worst_site_seed_easy_degradation_percent']:.4f} |")
    lines += ['', 'Reducing selection count does not monotonically reduce subgroup harm. In particular,',
        'damping005 with the neural head has worse worst-site/seed easy degradation after this',
        'count matching (3.166%) than under its original strict choices (2.494%). Selecting fewer',
        'rows is not a safety certificate. All comparisons remain four-site development evidence.', '']
    (PUBLIC/'matched_count_safety.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k != 'strict_controls'}, indent=2))


if __name__ == '__main__':
    main()
