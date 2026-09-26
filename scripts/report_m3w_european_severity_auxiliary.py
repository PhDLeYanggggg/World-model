"""Fixed severity-auxiliary contrasts against all retained strong controls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_severity_auxiliary as run
from scripts.report_m3w_european_support_fractional import paired_contrasts,summarize_population
from scripts.report_m3w_european_frozen_harm_readout import summarize_contrasts
COMPARISONS={'severity_vs_original':('severity_aux','original_mean'),
    'severity_vs_control':('severity_aux','cost_only'),'severity_vs_ordinary':('severity_aux','membership_aux')}


def gates_for(summary):
    required=[summary[k]['full'] for k in COMPARISONS]
    primary=all(v['envelope_positive__harm_MSE_gain_percent']['positive']==6 for v in required)
    guards=all(v[k]['negative']==v[k]['not_estimable']==0 for v in required for k in
        ('envelope_positive__top10_gain_pp','envelope_positive__coverage_log_error_reduction','all__H_all_MSE_gain_percent'))
    return dict(primary_six_positive_vs_all_controls=primary,tail_coverage_all_harm_guards=guards,
        development_cost_signal=primary and guards,new_policy_evaluated=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)


def aggregates(rows,cfg):
    cs={}
    for key,(new,old) in COMPARISONS.items():
        proxy=[dict(r,folds=[dict(f,metrics=dict(fractional=f['metrics'][new],mean=f['metrics'][old])) for f in r['folds']]) for r in rows]
        cs[key]=paired_contrasts(proxy,cfg)
    summary=summarize_contrasts(cs)
    populations={p:{s:{a:summarize_population([f['metrics'][a][s] for r in rows if r['pair']==p for f in r['folds']])
        for a in ('original_mean','cost_only','membership_aux','severity_aux')} for s in ('all','envelope_positive')} for p in cfg['pairs']}
    transport={}
    for key,(new,old) in COMPARISONS.items():
        transport[key]={}
        for pair in cfg['pairs']:
            counts=dict(views=0,not_estimable=0,fit_improved=0,held_improved=0,fit_only=0)
            for r in rows:
                if r['pair']!=pair: continue
                for f in r['folds']:
                    counts['views']+=1
                    a,b=[f['training'][v]['training']['harm_MSE'] for v in (new,old)]
                    c,d=[f['metrics'][v]['all']['harm_MSE'] for v in (new,old)]
                    if any(v is None for v in (a,b,c,d)): counts['not_estimable']+=1; continue
                    counts['fit_improved']+=int(a<b); counts['held_improved']+=int(c<d); counts['fit_only']+=int(a<b and c>=d)
            transport[key][pair]=counts
    return dict(contrasts=cs,summary=summary,populations=populations,fit_transport=transport,gates=gates_for(summary))


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['identity']==identity
    rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    a=aggregates(rows,cfg); run.immutable_json(run.PUBLIC/'aggregate_metrics.json',a); run.immutable_json(run.PUBLIC/'gates.json',a['gates'])
    receipts=[json.loads((ROOT/r['path']).read_text()) for r in done['heads']]
    fits=[dict(group=r['input']['group']['group'],pair=r['input']['pair'],held=r['input']['held'],fit=r['fit']) for r in receipts]
    run.immutable_json(run.PUBLIC/'training_metrics.json',fits)
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        parameters=24901,threads=4,interop=1,workers=0,real_torch=True,remote_jobs_submitted=0,
        cached_control_heads=288,elapsed_excludes_loading_inference_and_verification=True))
    lines=['# Severity Auxiliary Results','','## Material Passport',
        '144 fresh Torch fits /288,000 updates. Source forecasts and matched controls are cached_verified.',
        'All six assignments, three seeds and four held localities; source development only.',
        'Harm-weighted auxiliary output is not ordinary membership probability and is never multiplied into costs.', '',
        '| Comparison / pair / assignment | Easy-harm MSE gain % | 95% locality CI |','|---|---:|---:|']
    for c,ps in a['contrasts'].items():
        for pair,gs in ps.items():
            for role,ms in gs.items():
                v=ms['envelope_positive__harm_MSE_gain_percent']
                lines.append(f"| {c} / {pair} / {role} | {v.get('point','not_estimable')} | {v.get('CI','not_estimable')} |")
    lines+=['','Three seeds averaged within locality, then3,000 resamples of four localities. Roles/windows dependent; exploratory and not multiplicity-adjusted.',
        'The original forecaster, denominator readout, sampling and deployment remain frozen. No independent roles opened.',
        'No metric/seconds, human-gold, true3D, foundation, physical-safety, independent-confirmation or deployment claim. Stage5C/SMC off.','',
        '```json',json.dumps(a['gates'],indent=2),'```']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=a['gates'],primary={k:v['full']['envelope_positive__harm_MSE_gain_percent'] for k,v in a['summary'].items()}),indent=2))


if __name__=='__main__': main()
