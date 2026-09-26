"""Registered expected-cost comparisons; no policy readout or favorable-role choice."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_membership_cost as run
from scripts.report_m3w_european_support_fractional import paired_contrasts,summarize_population
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
from src.evaluation.m3w_easy_membership_diagnosis import summary as membership_summary
COMPARISONS={'conditional_vs_direct':('conditional','direct'),
    'conditional_vs_original':('conditional','original_mean'),
    'conditional_vs_constant':('conditional','constant'),'direct_vs_original':('direct','original_mean')}


def gates_for(summary):
    primary={k:summary[k]['full']['envelope_positive__harm_MSE_gain_percent']['positive']==6
             for k in ('conditional_vs_direct','conditional_vs_original','conditional_vs_constant')}
    required=[summary[k]['full'] for k in ('conditional_vs_direct','conditional_vs_original')]
    guards=all(x[key]['negative']==x[key]['not_estimable']==0 for x in required for key in
        ('envelope_positive__top10_gain_pp','envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent'))
    return dict(primary=primary,tail_coverage_all_harm_guards=guards,development_cost_signal=all(primary.values()) and guards,
        new_policy_evaluated=False,independent_confirmation=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)


def aggregates(rows,cfg):
    cs={}
    for key,(new,old) in COMPARISONS.items():
        proxy=[dict(r,folds=[dict(f,metrics=dict(fractional=f['metrics'][new],mean=f['metrics'][old])) for f in r['folds']]) for r in rows]
        cs[key]=paired_contrasts(proxy,cfg)
    summary=summarize_contrasts(cs)
    pop={pair:{subset:{arm:summarize_population([f['metrics'][arm][subset] for r in rows if r['pair']==pair for f in r['folds']])
         for arm in ('original_mean','direct','conditional','constant')} for subset in ('all','envelope_positive')} for pair in cfg['pairs']}
    membership={a:membership_summary([dict(r,folds=[dict(diagnosis=f['membership'][a]) for f in r['folds']]) for r in rows])
                for a in ('direct','conditional','constant')}
    transport={}
    for key,(new,old) in COMPARISONS.items():
        transport[key]={}
        for pair in cfg['pairs']:
            count=dict(views=0,not_estimable=0,fit_improved=0,held_improved=0,fit_only=0)
            for row in rows:
                if row['pair']!=pair: continue
                for f in row['folds']:
                    count['views']+=1
                    a,b=[f['training'][arm]['training']['harm_MSE'] for arm in (new,old)]
                    c,d=[f['metrics'][arm]['all']['harm_MSE'] for arm in (new,old)]
                    if any(v is None for v in (a,b,c,d)): count['not_estimable']+=1; continue
                    count['fit_improved']+=int(a<b); count['held_improved']+=int(c<d); count['fit_only']+=int(a<b and c>=d)
            transport[key][pair]=count
    return dict(contrasts=cs,summary=summary,populations=pop,membership=membership,fit_transport=transport,gates=gates_for(summary))


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['identity']==identity
    rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    agg=aggregates(rows,cfg); run.immutable_json(run.PUBLIC/'aggregate_metrics.json',agg); run.immutable_json(run.PUBLIC/'gates.json',agg['gates'])
    rs=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    fits=[dict(group=r['input']['group']['group'],pair=r['input']['pair'],held=r['input']['held'],arm=r['input']['arm'],fit=r['fit']) for r in rs]
    run.immutable_json(run.PUBLIC/'training_metrics.json',fits)
    qpath=run.PRIVATE/'create_queue.json'; q=json.loads(qpath.read_text())
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        new_parameters_per_arm=24706,conditional_extra_frozen_membership_parameters=24641,
        equal_new_head_budget_not_equal_total_model_cost=True,threads=4,interop=1,workers=0,real_torch=True,remote_jobs_submitted=0,
        create_queue=dict(receipt=run.artifact(qpath),completed_utc=q['completed_utc'],returncode=q['response']['returncode'],remote_modified=q['remote_modified'])))
    lines=['# Membership-Factorized Cost Results','','## Material Passport',
        '288 fresh cost heads / 576,000 updates. Frozen membership, source forecasts and reference costs are cached_verified.',
        'Source development only; no new forecast, policy, independent calibration or confirmation.','',
        '| Comparison / pair / roles | Conditional easy-harm MSE gain % | 95% locality CI |','|---|---:|---:|']
    for comp,ps in agg['contrasts'].items():
        for pair,gs in ps.items():
            for role,ms in gs.items():
                v=ms['envelope_positive__harm_MSE_gain_percent']; point=v.get('point')
                lines.append(f"| {comp} / {pair} / {role} | {point if point is not None else 'not_estimable'} | {v.get('CI','not_estimable')} |")
    lines+=['','Three seeds averaged within locality, then 3,000 resamples of four localities. Dependent source roles/windows; exploratory CIs, no multiplicity adjustment.',
        'Conditional uses an extra frozen membership MLP; the two new cost heads have equal capacity and draw/update budgets, not equal total system cost.',
        'The constant-probability counterfactual keeps conditional experts fixed and uses fitting-only prevalence.',
        'Initial objectives and output priors differ by target; fixed-batch loss curves are not an across-arm accuracy comparison.','',
        '## Gates','','```json',json.dumps(agg['gates'],indent=2),'```','',
        'No metric/seconds, human-gold, physical-safety, true3D, foundation or submission-ready claim. Stage5C and SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=agg['gates'],primary={k:v['full']['envelope_positive__harm_MSE_gain_percent'] for k,v in agg['summary'].items()}),indent=2))


if __name__=='__main__': main()
