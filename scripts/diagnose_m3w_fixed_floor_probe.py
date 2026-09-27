"""Post-readout conditional moment audit; no policy/model/threshold changes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_probe as run
import numpy as np


def main():
    cfg,data,jobs,identity,bound=run.load(); rows=[]
    for c in run.contexts(data,jobs,identity):
        cv,_,(floor,_),(neural,_)=run.costs(c,data,np.arange(len(c['ids'])))
        for pair,(_,held_sites) in enumerate(c['pairs']):
            name=c['name']+f'_pair{pair}'; home=run.PRIVATE/'fits'/name
            with np.load(home/'scores.npz',allow_pickle=False) as z:
                ids=z['ids']; pos=np.searchsorted(c['ids'],ids)
                np.testing.assert_array_equal(c['ids'][pos],ids)
                for site in held_sites:
                    local=data['sites'][ids]==site; ix=pos[local]
                    for offset,ref in enumerate(('cv','floor')):
                        take=z[ref+'_safe'][local]; scores=z['scores'][local,5*offset:5*offset+5]
                        y=run.api.targets(cv[ix],cv[ix] if ref=='cv' else floor[ix],neural[ix],c['job']['design']['easy_cut'])
                        known=np.isfinite(cv[ix]); chosen=known & take
                        pred=scores[chosen].astype(float); actual=y[chosen]
                        def ratio(a,b): return float(a/b) if b>0 else None
                        zero_pred_harm=chosen & (scores[:,1]==0)
                        rows.append(dict(site=site,policy=ref,metric=dict(
                            selected_rows=int(chosen.sum()),
                            predicted_harm_ratio=ratio(pred[:,1].sum(),pred[:,2].sum()),
                            actual_harm_ratio=ratio(actual[:,1].sum(),actual[:,2].sum()),
                            predicted_to_actual_harm=ratio(pred[:,1].sum(),actual[:,1].sum()),
                            predicted_to_actual_reference=ratio(pred[:,2].sum(),actual[:,2].sum()),
                            predicted_to_actual_easy_harm=ratio(pred[:,4].sum(),actual[:,4].sum()),
                            zero_predicted_harm_selected_fraction=ratio(zero_pred_harm.sum(),chosen.sum()),
                            harmed_among_zero_predicted_harm_fraction=ratio((zero_pred_harm & (y[:,1]>0)).sum(),zero_pred_harm.sum()))))
    sites=sorted(set(data['sites']))
    summary={p:{k:run.inter.paired_localities([r for r in rows if r['policy']==p],sites,k,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
                for k in rows[0]['metric']} for p in ('cv','floor')}
    run.immutable_json(run.PUBLIC/'conditional_moment_diagnosis.json',dict(result_source='fresh_run_post_readout_diagnostic',
        policy_changed=False,summary=summary,not_new_confirmatory_hypothesis=True))
    from scripts.report_m3w_fixed_floor_probe import number
    lines=['# Why Predicted Risk Does Not Match Selected Harm','',
        'Post-readout diagnosis of the frozen216 linear heads. No tuning, refitting or deployment.',
        'Each row aggregates equally over12 locality means, with dependent fits averaged first.','',
        '| Target | Predicted selected harm/reference | Actual selected harm/reference | Predicted/actual harm | Predicted/actual reference | Selected zero-harm prediction fraction | Actually harmed among zero-harm predictions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p,m in summary.items():
        lines.append('| '+p+' | '+' | '.join(number(m[k]) for k in ('predicted_harm_ratio','actual_harm_ratio',
            'predicted_to_actual_harm','predicted_to_actual_reference','zero_predicted_harm_selected_fraction',
            'harmed_among_zero_predicted_harm_fraction'))+' |')
    lines+=['','Undefined means at least one fixed source/view has no valid denominator; empty coverage is not zero harm.',
        'Ratios above are fractions, not percentage points. They concern the reference used by each target.',
        'The main report evaluates BOTH policies relative to the actual damping floor.',
        'This identifies moment underestimation and clipping behavior, not a fitted causal explanation or risk guarantee.']
    (run.PUBLIC/'conditional_moment_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary))


if __name__=='__main__': main()
