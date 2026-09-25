"""Matched B/C fitting reductions, never used to change the frozen decisions."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_reference_protection as run


def compare(records, arm, comparator, population):
    if len(records)!=18: raise ValueError('All eighteen matched settings required')
    pairs=[(r[arm][population],r[comparator][population]) for r in records]
    a=np.asarray([x['component_mse'] for x,_ in pairs],float)
    b=np.asarray([y['component_mse'] for _,y in pairs],float)
    if (a.shape!=(18,4) or b.shape!=a.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or (a<0).any() or (b<=0).any()):
        raise ValueError('Positive matched denominator errors required')
    coverage={k:[r[k][population]['easy_harm_fitted_over_actual'] for r in records]
        for k in (arm,comparator)}
    return dict(component_MSE_ratio_median=np.median(a/b,axis=0).tolist(),
        component_improved_count=np.sum(a<b,axis=0).tolist(),
        component_equal_count=np.sum(a==b,axis=0).tolist(),
        easy_harm_coverage={k:dict(estimable=sum(x is not None for x in v),
            median=float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None)
            for k,v in coverage.items()})


def main():
    cfg,identity=run.registration(); run.checked_training(identity)
    replay=json.loads((run.PUBLIC/'replay_receipt.json').read_text()); assert replay['all_passed']
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    old=json.loads((run.prior.PUBLIC/'matched_transport_audit.json').read_text())
    oldfits={(r['group'],r['pair']):r['fits']['uniform'] for r in old['groups']}
    bfits={(r['group'],r['pair']):r['arms'] for r in replay['full_B_fitting']}
    populations={p:{'B':[],'C':[]} for p in cfg['pairs']}
    selection={p:{a:[] for a in cfg['arms']} for p in cfg['pairs']}
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref
        row=json.loads((ROOT/ref['path']).read_text()); pair=row['pair']; key=(row['group'],pair)
        populations[pair]['B'].append(bfits[key])
        populations[pair]['C'].append(dict(row['matched_action_diagnostics'],uniform=oldfits[key]))
        for arm,values in row['query_budget'].items():
            for site,value in values.items():
                e=value['events']['easy']
                selection[pair][arm].append(dict(group=row['group'],site=site,**e))
    out={}
    for pair,roles in populations.items():
        out[pair]={role:{arm+'_vs_'+comparator:{population:compare(records,arm,comparator,population)
            for population in ('population','old_raw_selected')}
            for arm,comparator in (('protected','continued'),('protected','uniform'),('continued','uniform'))}
            for role,records in roles.items()}
    text=['# Reference Preservation and Harm Transport','','Posthoc diagnosis only; no refitting or decision changes.',
        'B uses fresh frozen-model replay; C uses checked frozen predictions. Cost/loss scales remain B-only.',
        'Rows summarize eighteen dependent settings; output order is D_all, H_all, D_easy, H_easy.',
        'A ratio below one means lower normalized squared error, not calibrated selected harm.','',
        '| Pair / role | Comparison | Population | Median component MSE ratios | Improved counts /18 |',
        '|---|---|---|---|---|']
    for pair,roles in out.items():
        for role,comparisons in roles.items():
            for contrast,pops in comparisons.items():
                for pop,value in pops.items():
                    ratios=', '.join(f'{v:.5f}' for v in value['component_MSE_ratio_median'])
                    counts=', '.join(str(v) for v in value['component_improved_count'])
                    text.append(f'| {pair} / {role} | {contrast} | {pop} | {ratios} | {counts} |')
    text+=['','## Same-Action Easy-Harm Coverage','',
        'Old raw-neural action masks are identical across all models. Values are median predicted/actual easy-harm mass.',
        '| Pair / role | Uniform | Continued | Protected |','|---|---:|---:|---:|']
    for pair,roles in out.items():
        for role,comparisons in roles.items():
            cov=comparisons['protected_vs_uniform']['old_raw_selected']['easy_harm_coverage']
            cc=comparisons['continued_vs_uniform']['old_raw_selected']['easy_harm_coverage']
            text.append(f"| {pair} / {role} | {cov['uniform']['median']:.5f} | {cc['continued']['median']:.5f} | {cov['protected']['median']:.5f} |")
    text+=['','## Boundaries','',
        'C is historically opened development, not independent confirmation. Independent calibration/confirmation remain closed.',
        'Net easy preservation does not imply control of positive harm. Own-policy selected sets differ across arms.',
        'Freezing an inaccurate reference estimate does not calibrate it. Population MSE is not selected-group risk.',
        'Image pixels, annotation steps, detector-derived labels; no metric/seconds, physical-safety or foundation claim.',
        'No deployment change. Stage5C and SMC remain off.']
    run.immutable_json(run.PUBLIC/'fitting_transport_audit.json',dict(summary=out,
        own_policy_easy_query_mass=selection,source_binding=run.artifact(Path(__file__)),
        inputs=[run.artifact(run.PUBLIC/'replay_receipt.json'),run.artifact(run.PUBLIC/'completion_checks.json'),
            run.artifact(run.prior.PUBLIC/'matched_transport_audit.json')],
        result_source='fresh_run_posthoc_reduction_checked_frozen_B_C_outputs',
        changed_fit=False,changed_decisions=False,independent_confirmation=False))
    (run.PUBLIC/'fitting_transport_analysis.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
