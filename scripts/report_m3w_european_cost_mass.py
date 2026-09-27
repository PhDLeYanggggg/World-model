"""Registered harm-mass contrasts and descriptive fitting loss decomposition."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cost_mass as run
from scripts.report_m3w_european_support_fractional import paired_contrasts
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY, GUARDS
import numpy as np

COMPARISONS = {f'mass_{arm}_vs_{mode}_{arm}':(arm+'_mass',arm+'_'+mode)
    for arm in ('cost_only','cap_aux','shuffled_aux') for mode in ('raw','scaled')}
COMPARISONS.update(mass_true_vs_mass_cost=('cap_aux_mass','cost_only_mass'),
    mass_true_vs_mass_shuffled=('cap_aux_mass','shuffled_aux_mass'))


def summarize(values):
    a = [v for v in values if v is not None and np.isfinite(v)]
    return dict(records=len(values),not_estimable=len(values)-len(a),
        minimum=float(min(a)) if a else None,median=float(np.median(a)) if a else None,
        maximum=float(max(a)) if a else None)


def fitting_summary(records):
    out = {}
    for family in ('full','motion_only'):
        rows = [r for r in records if r['pair'] == family]; assert len(rows) == 72
        out[family] = {}
        for arm in ('cost_only','cap_aux','shuffled_aux'):
            models = [r['models'][arm] for r in rows]; details = [r['details'][arm] for r in rows]
            report = dict(views=len(rows),mass_preserved=sum(m['mass_preserved'] for m in models),
                bound_components=sum(a == 8 for m in models for a in m['slopes']),
                zero_slope_components=sum(a == 0 for m in models for a in m['slopes']),
                H_all_slope=summarize([m['slopes'][0] for m in models]),
                H_easy_slope=summarize([m['slopes'][1] for m in models]))
            for weighting in ('equal_locality','row_weighted'):
                report[weighting] = {}
                for component in ('H_all','H_easy'):
                    z = [r[weighting] for r in details]
                    values = {mode:{key:summarize([r[mode][component][key] for r in z])
                        for key in ('MSE','zero_target_SSE','positive_target_SSE','mass_ratio',
                            'LS_denominator_zero_share','unprojected_LS_slope','unprojected_mean_slope','projection_mass_loss')}
                        for mode in ('raw','unprojected_L2','scaled','mass')}
                    def tradeoff(r):
                        a,b = r['raw'][component],r['scaled'][component]
                        ratios = (a['mass_ratio'],b['mass_ratio'])
                        return all(v is not None and v > 0 for v in ratios) and b['MSE'] < a['MSE'] and abs(np.log(ratios[1])) > abs(np.log(ratios[0]))
                    values['L2_MSE_improves_but_mass_log_error_worsens'] = sum(tradeoff(r) for r in z)
                    report[weighting][component] = values
            out[family][arm] = report
    return out


def aggregate(rows,cfg,supported):
    contrasts = {}
    for key,(new,old) in COMPARISONS.items():
        proxy = [dict(row,folds=[dict(f,metrics=dict(fractional=f['metrics'][new],mean=f['metrics'][old]))
            for f in row['folds']]) for row in rows]
        contrasts[key] = paired_contrasts(proxy,cfg)
    summary = summarize_contrasts(contrasts)
    def screen(names):
        primary = all(summary[n]['full'][PRIMARY]['positive'] == 6 for n in names)
        guards = all(summary[n]['full'][g]['negative'] == summary[n]['full'][g]['not_estimable'] == 0 for n in names for g in GUARDS)
        return primary,guards
    primary,guards = screen(['mass_cost_only_vs_raw_cost_only','mass_cost_only_vs_scaled_cost_only'])
    info,info_guards = screen(['mass_true_vs_mass_cost','mass_true_vs_mass_shuffled'])
    return dict(contrasts=contrasts,summary=summary,gates=dict(
        fitting_mass_constraints_supported=supported,primary_readout_gate=primary,guard_gate=guards,
        auxiliary_information_gate=info and info_guards,
        development_readout_screen=supported and primary and guards,
        independent_confirmation=False,new_policy_evaluated=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False))


def main():
    cfg,_ = run.registration(); freeze = run.check_freeze()
    records = [json.loads((ROOT/ref['path']).with_name('models.json').read_text()) for ref in freeze['readouts']]
    fs = fitting_summary(records); run.immutable_json(run.PUBLIC/'fitting_diagnostics.json',fs)
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    out = aggregate(rows,cfg,fs['full']['cost_only']['mass_preserved'] == 72)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',out)
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(fresh_scalar_readouts=432,
        parameters_per_readout=2,new_neural_heads=0,new_optimizer_updates=0,
        root_iterations_per_component=80,full_views=144,direct_MSE_checks=2592,
        cached_inner_neural_heads=1296,cached_outer_neural_heads=432,
        threads=4,interop=1,workers=0,remote_jobs_submitted=0))
    lines = ['# Expected Cost and Harm-Mass Readout','','## Material Passport',
        'fresh_run:432 scalar moment readouts,loss decomposition and144 source-held views.',
        'cached_verified:neural heads,forecasts,causal features and raw/L2 predictions.',
        'not_run:new neural training,new policy,independent selection/calibration/confirmation.','',
        'Three seeds average within locality;3000 paired four-locality resamples. Overlapping exposed-source assignments;unadjusted intervals.',
        'Primary remains expected easy-harm MSE,not trajectory accuracy or physical safety.','',
        '| Contrast/family | Positive/negative/overlap/missing CIs | Primary point range (%) |','|---|---|---|']
    for name,families in out['summary'].items():
        for family,metrics in families.items():
            m = metrics[PRIMARY]
            lines.append(f"| {name}/{family} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## Every Primary Interval','','| Contrast/family/assignment | Point (%) | 95% CI |','|---|---:|---|']
    for name,families in out['contrasts'].items():
        for family,groups in families.items():
            for group,metrics in groups.items():
                m = metrics[PRIMARY]; lines.append(f"| {name}/{family}/{group} | {m.get('point')} | {m.get('CI','not_estimable')} |")
    lines += ['', '## Boundaries',
        'Fitting moment equality is not conditional or held-scene calibration. Bound hits remain included.',
        'Primary and all original guards remain unchanged;coverage gain cannot replace failed MSE.',
        'Full/motion families have different forecast/event populations,not a matched feature ablation.',
        'Obs8/pred12 native steps,detector pixels;no metric/seconds,human-gold,true3D or foundation claim.',
        '```json',json.dumps(out['gates'],indent=2),'```']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    diag = ['# Fitting Loss Decomposition','','Descriptive dependent-view summaries;not held evaluation,selection or independent evidence.',
        'Equal-locality weighting matches fitting. Row weighting is reported separately in JSON.',
        'MSE decomposes into zero-target and positive-target contributions;the per-locality identities are checked.', '',
        '| Family/arm | Matched mass views | H_easy median mass ratio raw/L2/mass | L2 MSE improves but log-mass worsens | Zero-target LS denominator share |',
        '|---|---:|---|---:|---:|']
    for family,arms in fs.items():
        for arm,v in arms.items():
            h = v['equal_locality']['H_easy']
            diag.append(f"| {family}/{arm} | {v['mass_preserved']}/72 | {[h[m]['mass_ratio']['median'] for m in ('raw','scaled','mass')]} | {h['L2_MSE_improves_but_mass_log_error_worsens']} | {h['raw']['LS_denominator_zero_share']['median']} |")
    diag += ['', 'No outer outcomes are used in these fits. Slope/cap feasibility failures stay visible.',
        'A restricted origin-L2 estimator plus projection need not preserve mass. This does not refute squared error as a proper mean score.',
        'The fixed projected moment readout tests this specific restriction,not a universal calibration repair.']
    (run.PUBLIC/'fitting_diagnostics.md').write_text('\n'.join(diag)+'\n')
    print(json.dumps(out['gates'],indent=2))


if __name__ == '__main__': main()
