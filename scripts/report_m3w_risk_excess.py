"""Report the pre-specified loss contrast without changing a model or threshold."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_excess as run


def fmt(x): return 'undefined' if x is None else f'{x:.4f}'


def interval(x, scale=1):
    c = x['ci95']; value = x['point']
    return fmt(value*scale if value is not None else None)+(f' [{fmt(c[0]*scale)}, {fmt(c[1]*scale)}]' if c else ' [undefined]')


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    rows = ['# Direct Risk-Budget Objective: Matched Results', '',
        'Fresh_run:144 risk-score heads with288000 updates and comparative scoring.',
        'Cached_verified: the two-moment control weights and frozen forecasting bank; all controls',
        'have fresh inference verified against their old predictions. No new forecaster was fitted.', '',
        '## Primary Mechanistic Contrast', '',
        'Signed score is positive harm minus0.02CV ADE. Positive MSE improvement means lower error',
        'than the matched two-moment controller. Roles, preprocessing, sampling, initialization,',
        'architecture and update budget are unchanged. Only the objective differs. Each source',
        'averages dependent producer/seed views before the3000-draw12-locality bootstrap.', '',
        '| Candidate | Fitting MSE improvement % | Held MSE improvement % |', '|---|---:|---:|']
    for arm, s in d['by_candidate'].items():
        rows.append(f"| {arm} | {interval(s['fit_MSE_gain_vs_control_percent'])} | {interval(s['held_MSE_gain_vs_control_percent'])} |")
    rows += ['', '## Fixed All-Risk Screen', '',
        'Both objectives use score<=0. These all-risk-only screens omit full utility/easy guards',
        'and are not proposed deployable selectors. Positive-harm/reference is not net ADE degradation.', '',
        '| Candidate | Objective | Held coverage % | Held screened positive harm % | Held all ADE gain vs CV % | Held easy gain % | Held hard gain % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm, s in d['by_candidate'].items():
        for m, name in [('control','two_moment_MSE'),('new','signed_excess_MSE')]:
            keys = [('screen_rate',100),('screen_positive_harm_ratio',100),('all_ADE_gain_vs_CV_percent',1),
                    ('easy_ADE_gain_vs_CV_percent',1),('hard_ADE_gain_vs_CV_percent',1)]
            rows.append('| '+arm+' | '+name+' | '+' | '.join(interval(s['held_'+m+'_'+k], factor) for k, factor in keys)+' |')
    rows += ['', '## Source Fitting Versus Holdout', '',
        '| Candidate | Objective | Fitting positive harm % | Held positive harm % |', '|---|---|---:|---:|']
    for arm, s in d['by_candidate'].items():
        for m in ('control','new'):
            rows.append(f"| {arm} | {m} | {interval(s['fit_'+m+'_screen_positive_harm_ratio'],100)} | {interval(s['held_'+m+'_screen_positive_harm_ratio'],100)} |")
    rows += ['', '## Three Seeds', '', '| Candidate | Seed | Held signed-MSE improvement % |', '|---|---:|---:|']
    for arm, seeds in d['by_seed'].items():
        for seed, s in seeds.items(): rows.append(f"| {arm} | {seed} | {interval(s['held_MSE_gain_vs_control_percent'])} |")
    rows += ['', '## Individual-View Failures', '',
        'The locality-mean easy gate averages dependent producer/seed views; it does not certify every view.', '',
        '| Candidate | Worst view easy gain % | Views with easy degradation >2% | Zero-reference harmed row-views |',
        '|---|---:|---:|---:|']
    for arm in ('dimensionless','damped'):
        rr = [r['metric'] for r in d['rows'] if r['candidate'] == arm]
        if rr:
            rows.append(f"| {arm} | {min(r['held_new_easy_ADE_gain_vs_CV_percent'] for r in rr):.4f} | {sum(r['held_new_easy_ADE_gain_vs_CV_percent'] < -2 for r in rr)}/{len(rr)} | {sum(r['held_new_zero_reference_harmed'] for r in rr)} |")
    rows += ['', '## Screens and Limitations', '', *[f'- {k}: {v}' for k, v in d['gates'].items()], '',
        'The new two components are an internal parameterization: only their signed combination',
        'is supervised, so interpreting either component as a calibrated moment is invalid.',
        'A loss improvement does not establish reliable accepted-subset risk, independent calibration,',
        'a neural full-policy advantage, or scene-joint benefit. No held outcome selected a threshold',
        'or checkpoint. Unknown labels remain excluded; zero-reference harm stays explicit.',
        'Outer readout and independent roles remain unused. Silver image-local obs8/pred12 at',
        'raw-frame stride12, not metric, seconds, human gold, physical safety, true3D or foundation.',
        'No deployment change, Stage5C execution or SMC.', '']
    (run.PUBLIC/'results.md').write_text('\n'.join(rows))
    frozen = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    records = [json.loads((ROOT/r['path']).read_text()) for r in frozen['heads']]
    training = dict(heads=len(records), updates=sum(r['fit']['step'] for r in records),
        fit_seconds=sum(r['fit']['seconds'] for r in records), unknown_training_draws=sum(r['fit']['unknown_rows_sampled'] for r in records),
        matched_sampler_checks=sum(r['matched_sampling_exact'] for r in records),
        fresh_control_inferences_exact=sum(r['control_inference_exact'] for r in records),
        result_source='fresh_run', changed_factor='objective_only', component_moments_identified=False)
    run.immutable_json(run.PUBLIC/'training.json', training)
    print(json.dumps(dict(training=training, gates=d['gates'])))


if __name__ == '__main__': main()
