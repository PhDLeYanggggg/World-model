"""Reproducible aggregate reporting; no policy search or individual-row export."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_moment_crossfit as run


def fmt(value):
    return 'undefined' if value is None else f'{value:.4f}'


def interval(value):
    c = value['ci95']
    return fmt(value['point'])+(f' [{fmt(c[0])}, {fmt(c[1])}]' if c is not None else ' [undefined]')


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    rows = ['# Cost-Moment Cross-Fit Results', '',
        'Source: fresh_run for144 controller heads and diagnostics; cached_verified for the frozen',
        'forecaster bank. This is three-source fitting versus one inner-held source, not independent',
        'confirmation or the previous complete selection policy. Outer readout outcomes are unused.', '',
        '## Primary Reference-Cost Diagnostic', '',
        'MSE skill is100*(1-head MSE/fitting-constant MSE); positive is better. Each locality first',
        'averages its dependent role/seed views. Intervals bootstrap12localities with3000 draws.', '',
        '| Candidate | Fitting reference skill % | Held reference skill % | Held minus fit points |',
        '|---|---:|---:|---:|']
    for arm, s in d['by_candidate'].items():
        rows.append(f"| {arm} | {interval(s['fit_reference_MSE_skill_percent'])} | {interval(s['held_reference_MSE_skill_percent'])} | {interval(s['gap_reference_MSE_skill_percent'])} |")
    rows += ['', '## Harm and Fixed2% Screen', '',
        '| Candidate | Fitting harm skill % | Held harm skill % | Held screen rate | Held actual harm/reference |',
        '|---|---:|---:|---:|---:|']
    for arm, s in d['by_candidate'].items():
        rows.append(f"| {arm} | {interval(s['fit_harm_MSE_skill_percent'])} | {interval(s['held_harm_MSE_skill_percent'])} | {interval(s['held_screen_rate'])} | {interval(s['held_screen_actual_harm_ratio'])} |")
    rows += ['', '| Candidate | Fit predicted ratio | Fit actual ratio | Held predicted ratio | Held actual ratio |',
        '|---|---:|---:|---:|---:|']
    for arm, s in d['by_candidate'].items():
        keys = ['fit_screen_predicted_harm_ratio','fit_screen_actual_harm_ratio',
                'held_screen_predicted_harm_ratio','held_screen_actual_harm_ratio']
        rows.append('| '+arm+' | '+' | '.join(interval(s[k]) for k in keys)+' |')
    rows += ['', 'The all-risk screen omits utility/easy heads on purpose, preventing the other heads',
        'from seeing the held controller source. It is not a proposed deployable policy. An empirical',
        'ratio over2% diagnoses a miscalibrated predicted screen; it does not invalidate a nonexistent',
        'conformal guarantee. A zero true reference denominator remains undefined, with absolute harm retained.', '',
        '## Numerator Versus Denominator', '',
        'Actual/predicted reference <1 means an inflated predicted risk budget. Actual/predicted harm',
        '>1 means underestimated harm. These are equal-locality means of dependent-view ratios, not',
        'a pooled probability or a factorization using independent medians.', '',
        '| Candidate | Held reference actual/pred | Held harm actual/pred | Screen reference actual/pred | Screen harm actual/pred |',
        '|---|---:|---:|---:|---:|']
    for arm, s in d['by_candidate'].items():
        keys = ['held_reference_actual_over_predicted','held_harm_actual_over_predicted',
                'held_screen_reference_actual_over_predicted','held_screen_harm_actual_over_predicted']
        rows.append('| '+arm+' | '+' | '.join(interval(s[k]) for k in keys)+' |')
    rows += ['', '## Seed Breakdown', '', '| Candidate | Seed | Held reference skill % | Held harm skill % |', '|---|---:|---:|---:|']
    for arm, seeds in d['by_seed'].items():
        for seed, s in seeds.items():
            rows.append(f"| {arm} | {seed} | {interval(s['held_reference_MSE_skill_percent'])} | {interval(s['held_harm_MSE_skill_percent'])} |")
    rows += ['', '## Scope', '',
        'No trajectory deployment gain is estimated by this experiment. No model/threshold was selected',
        'from these held diagnostics. Unknown labels are excluded, zero-CV cases retained. Per-locality',
        'fixed score bins and absolute costs are preserved privately; public summaries contain aggregates.',
        'Independent selection/calibration/confirmation remain closed. Detector-derived silver image-local',
        'tracks; obs8/pred12 at raw-frame stride12. No metric, seconds, human-gold, true3D or foundation',
        'claim. No Stage5C execution, SMC or deployment change.', '']
    (run.PUBLIC/'results.md').write_text('\n'.join(rows))
    frozen = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    records = [json.loads((ROOT/r['path']).read_text()) for r in frozen['heads']]
    training = dict(heads=len(records), updates=sum(r['fit']['step'] for r in records),
        fit_seconds=sum(r['fit']['seconds'] for r in records),
        unknown_training_draws=sum(r['fit']['unknown_rows_sampled'] for r in records),
        parameters_per_head=sorted(set(r['fit']['parameters'] for r in records)),
        result_source='fresh_run', forecaster_source='cached_verified')
    run.immutable_json(run.PUBLIC/'training.json', training)
    gates = dict(diagnostic_training_complete=len(records)==144,
        unknown_training_draws_zero=training['unknown_training_draws']==0,
        held_reference_prediction_beats_constant={}, held_harm_prediction_beats_constant={},
        calibration_certificate=False, policy_benefit_tested=False, independent_confirmation=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    for arm, s in d['by_candidate'].items():
        for name in ('reference','harm'):
            ci = s[f'held_{name}_MSE_skill_percent']['ci95']
            gates[f'held_{name}_prediction_beats_constant'][arm] = ci is not None and ci[0] > 0
    run.immutable_json(run.PUBLIC/'gates.json', gates)
    print(json.dumps(dict(training=training, diagnostic_gates=gates)))


if __name__ == '__main__': main()
