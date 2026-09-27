"""Prespecified crossed-regime contrasts, replica-aware locality bootstrap."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_regime_transport as run
from scripts.report_m3w_european_support_fractional import differences
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from scripts.report_m3w_european_cap_auxiliary_cost import PRIMARY, GUARDS
import numpy as np

COMPARISONS = {}
for mode in ('raw','scaled'):
    for name,a,b in [('cut_at_two','two_cut3','two_cut2'),('cut_at_three','three_cut3','three_cut2'),
                     ('regime_at_cut2','three_cut2','two_cut2'),('regime_at_cut3','three_cut3','two_cut3')]:
        COMPARISONS[name+'_'+mode] = (a+'_'+mode,b+'_'+mode)
for cell in run.method.CELLS:
    COMPARISONS['scale_'+cell] = (cell+'_scaled',cell+'_raw')


def view_contrasts(row, label):
    out = {}
    for key,(a,b) in COMPARISONS.items():
        out[key] = run.method.mean_replica_deltas([differences(dict(metrics=dict(
            fractional=r['metrics'][label][a],mean=r['metrics'][label][b]))) for r in row['replicas']])
    for mode in ('raw','scaled'):
        replicas = []
        for r in row['replicas']:
            m = r['metrics'][label]
            values = [m[c+'_'+mode]['envelope_positive'].get('harm_MSE') for c in run.method.CELLS]
            a,b,c,d = values
            value = 100*((a-b)-(c-d))/d if all(v is not None and np.isfinite(v) for v in values) and d > 0 else None
            replicas.append(dict(easy_harm_MSE_interaction_pp=value))
        out['interaction_'+mode] = run.method.mean_replica_deltas(replicas)
    return out


def aggregate(rows, cfg):
    result = {}
    for label in ('outer_cut3','inner_cut2_diagnostic'):
        prepared = [(r,view_contrasts(r,label)) for r in rows]; cs = {}
        for comp in prepared[0][1]:
            cs[comp] = {}
            for family in cfg['pairs']:
                cs[comp][family] = {}
                for producer,controller in sorted({(r['producer'],r['controller']) for r in rows}):
                    group = [(r,d[comp]) for r,d in prepared if r['pair'] == family
                             and (r['producer'],r['controller']) == (producer,controller)]
                    sites = sorted({r['held'] for r,d in group}); assert len(sites) == 4 and len(group) == 12
                    metrics = {}
                    for key in group[0][1]:
                        values = []
                        for site in sites:
                            selected = [(r,d) for r,d in group if r['held'] == site]
                            assert sorted(r['seed'] for r,d in selected) == cfg['seeds']
                            numbers = [d[key] for r,d in selected]
                            if any(n is None or not np.isfinite(n) for n in numbers): break
                            values.append(float(np.mean(numbers)))
                        if len(values) != 4:
                            metrics[key] = dict(status='not_estimable',reason='missing_replica_seed_or_locality_support_not_dropped'); continue
                        v = np.asarray(values); rng = np.random.default_rng(cfg['bootstrap_seed'])
                        draws = v[rng.integers(0,4,size=(cfg['bootstrap_resamples'],4))].mean(1)
                        metrics[key] = dict(point=float(v.mean()),CI=np.quantile(draws,[.025,.975]).tolist(),
                            locality_points=dict(zip(sites,values)),localities=4,seeds=3,replicas_per_view=3)
                    cs[comp][family][f'producer{producer}_controller{controller}'] = metrics
        result[label] = dict(contrasts=cs,summary=summarize_contrasts(cs))
    s = result['outer_cut3']['summary']
    def screen(names):
        return all(s[n]['full'][PRIMARY]['positive'] == 6 and all(
            s[n]['full'][g]['negative'] == s[n]['full'][g]['not_estimable'] == 0 for g in GUARDS) for n in names)
    result['gates'] = dict(cut_explanation_screen=screen(['cut_at_two_raw','cut_at_three_raw']),
        larger_fitting_regime_screen=screen(['regime_at_cut2_raw','regime_at_cut3_raw']),
        size_matched_magnitude_screen=screen(['scale_two_cut2']),
        interaction_consistency={mode:any(s['interaction_'+mode]['full']['easy_harm_MSE_interaction_pp'][k] == 6
            for k in ('positive','negative')) for mode in ('raw','scaled')},
        new_auxiliary_failure_cause_proven=False,development_advance_gate=False,
        new_policy_evaluated=False,independent_confirmation=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False)
    return result


def main():
    cfg,_ = run.registration(); frozen = run.check_freeze()
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    out = aggregate(rows,cfg); run.immutable_json(run.PUBLIC/'aggregate_metrics.json',out)
    fits = [json.loads((ROOT/r['path']).read_text()) for r in frozen['heads']]
    bridge = json.loads((ROOT/frozen['pilot_bridge']['path']).read_text())
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(fresh_crossed_heads=len(fits),
        crossed_updates=sum(r['fit']['step'] for r in fits),native_bridge_updates=bridge['fit']['step'],
        fitting_seconds=sum(r['fit']['seconds'] for r in fits)+bridge['fit']['seconds'],
        unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits)+bridge['fit']['unknown_rows_sampled'],
        cached_two_site_controls=432,cached_three_site_controls=144,new_readout_fits=0,
        matched_cut_checks=frozen['matched_cut_checks'],threads=4,interop=1,workers=0,
        excludes_IO_preflight_evaluation=True,remote_jobs_submitted=0))
    lines = ['# Fitting Regime / Easy-Cut Transport','','## Material Passport',
        'fresh_run:864 crossed Torch cost heads and one native bridge;cached_verified:576 native controls and frozen readouts.',
        'not_run:new forecasting/policy,independent selection/calibration/confirmation.','',
        'Primary uses unchanged outer_cut3. inner_cut2_diagnostic is not a replacement endpoint.',
        'Replicas average within locality/seed;three seeds then average inside locality;3000 four-locality resamples.',
        'Previously exposed source development;overlapping assignments;unadjusted CIs.','',
        '| Target/contrast/family | Positive/negative/overlap/missing intervals | Point range |','|---|---|---|']
    for label in ('outer_cut3','inner_cut2_diagnostic'):
        for comp,families in out[label]['summary'].items():
            key = 'easy_harm_MSE_interaction_pp' if comp.startswith('interaction_') else PRIMARY
            for family,metrics in families.items():
                m = metrics[key]
                lines.append(f"| {label}/{comp}/{family} | {[m[k] for k in ('positive','negative','overlap','not_estimable')]} | {m['point_range']} |")
    lines += ['', '## Every Primary-Cut Interval','','| Contrast/family/assignment | Point | 95% CI |','|---|---:|---|']
    for comp,families in out['outer_cut3']['contrasts'].items():
        key = 'easy_harm_MSE_interaction_pp' if comp.startswith('interaction_') else PRIMARY
        for family,groups in families.items():
            for group,metrics in groups.items():
                m = metrics[key]; lines.append(f"| {comp}/{family}/{group} | {m.get('point')} | {m.get('CI','not_estimable')} |")
    lines += ['', '## Boundaries',
        'The regime includes row composition,preprocessing and sampling,not pure sample count.',
        'two_cut3 uses its omitted row site for its training label definition;it cannot serve as inner-OOF for that site.',
        'No held target is an inference input. All controller fitting/definition sites exclude the outer held locality.',
        'Cost-only mechanisms do not establish why the auxiliary failed. No causal-world or deployment claim.',
        'Expected-cost MSE is not trajectory ADE/FDE or easy degradation. Full/motion are not matched ablations.',
        'Obs8/pred12 native steps,detector pixels;no metric/seconds,human-gold,true3D,foundation or physical-safety claim.',
        '```json',json.dumps(out['gates'],indent=2),'```']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out['gates'],indent=2))


if __name__ == '__main__': main()
