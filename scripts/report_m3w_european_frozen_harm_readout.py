"""Prespecified matched representation and original-magnitude comparisons."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_frozen_harm_readout as run
from scripts.report_m3w_european_support_fractional import paired_contrasts,summarize_population
from src.evaluation.m3w_easy_membership_diagnosis import summary as membership_summary

COMPARISONS={'fractional_vs_matched':('fractional_features','mean_features'),
    'fractional_vs_original':('fractional_features','original_mean'),
    'matched_vs_original':('mean_features','original_mean')}


def contrasts(rows,cfg):
    result={}
    for key,(new,old) in COMPARISONS.items():
        proxy=[dict(r,folds=[dict(f,metrics=dict(fractional=f['metrics'][new],mean=f['metrics'][old])) for f in r['folds']]) for r in rows]
        result[key]=paired_contrasts(proxy,cfg)
    return result


def summarize_contrasts(cs):
    out={}
    for comparison,pairs in cs.items():
        out[comparison]={}
        for pair,groups in pairs.items():
            result={}
            for key in next(iter(groups.values())):
                vs=[v[key] for v in groups.values()]; good=[v for v in vs if 'CI' in v]
                result[key]=dict(positive=sum(v['CI'][0]>0 for v in good),negative=sum(v['CI'][1]<0 for v in good),
                    overlap=sum(v['CI'][0]<=0<=v['CI'][1] for v in good),not_estimable=len(vs)-len(good),
                    point_range=[min(v['point'] for v in good),max(v['point'] for v in good)] if good else None)
            out[comparison][pair]=result
    return out


def gates_for(summary):
    required=[summary[c]['full'] for c in ('fractional_vs_matched','fractional_vs_original')]
    primary=[x['envelope_positive__harm_MSE_gain_percent']['positive']==6 for x in required]
    guards=all(x['envelope_positive__'+k]['negative']==x['envelope_positive__'+k]['not_estimable']==0
        for x in required for k in ('top10_gain_pp','coverage_log_error_reduction'))
    return dict(training_complete=True,reference_moments_preserved=True,matched_primary_six_positive=primary[0],
        original_primary_six_positive=primary[1],tail_coverage_guards=guards,development_advance_gate=all(primary) and guards,
        new_policy_evaluated=False,independent_confirmation=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)


def fit_transport(rows):
    result={}
    for comparison,(new,old) in COMPARISONS.items():
        result[comparison]={}
        for pair in sorted({r['pair'] for r in rows}):
            counts=dict(views=0,not_estimable=0,fit_improved=0,held_improved=0,both_improved=0,fit_only=0,held_only=0,neither=0)
            for row in rows:
                if row['pair']!=pair: continue
                for fold in row['folds']:
                    counts['views']+=1
                    a,b=[fold['training'][arm]['training'].get('harm_MSE') for arm in (new,old)]
                    c,d=[fold['metrics'][arm]['all'].get('harm_MSE') for arm in (new,old)]
                    if any(v is None for v in (a,b,c,d)):
                        counts['not_estimable']+=1; continue
                    fit,held=a<b,c<d
                    counts['fit_improved']+=int(fit); counts['held_improved']+=int(held)
                    counts['both_improved' if fit and held else 'fit_only' if fit else 'held_only' if held else 'neither']+=1
            result[comparison][pair]=counts
    return result


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    cs=contrasts(rows,cfg); summary=summarize_contrasts(cs); gates=gates_for(summary)
    populations={pair:{subset:{arm:summarize_population([f['metrics'][arm][subset] for r in rows if r['pair']==pair for f in r['folds']])
        for arm in ('original_mean','original_fractional',*cfg['arms'])} for subset in ('all','envelope_positive')} for pair in cfg['pairs']}
    membership={a:membership_summary([dict(r,folds=[dict(diagnosis=f['membership'][a]) for f in r['folds']]) for r in rows]) for a in cfg['arms']}
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',dict(contrasts=cs,summary=summary,populations=populations,membership=membership,
        fit_transport=fit_transport(rows)))
    run.immutable_json(run.PUBLIC/'gates.json',gates)
    fits=[json.loads((ROOT/ref['path']).read_text()) for ref in done['heads']]
    run.immutable_json(run.PUBLIC/'training_metrics.json',[dict(group=r['input']['group']['group'],pair=r['input']['pair'],
        held=r['input']['held'],arm=r['input']['arm'],fit=r['fit']) for r in fits])
    remote_path=run.PRIVATE/'create_queue.json'; remote=json.loads(remote_path.read_text())
    assert remote['response']['returncode']==0 and remote['jobs_submitted']==0 and not remote['remote_modified']
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        threads=4,interop=1,workers=0,real_torch=True,trainable_parameters_per_readout=130,new_forecaster_training=False,
        remote_jobs_submitted=0,create_queue_observation=dict(receipt=run.artifact(remote_path),
            started_utc=remote['started_utc'],completed_utc=remote['completed_utc'],returncode=0,
            response_nonempty=bool(remote['response']['stdout'].strip()),remote_modified=False,
            claim='read_only_account_queue_observation_not_M3W_remote_training')))
    lines=['# Frozen Harm Readout Results','','## Material Passport',
        '288 fresh readouts /576,000updates; frozen cached-verified encoders and original controls.',
        'Previously opened source development; no new forecast, policy or independent confirmation.','',
        '| Comparison / pair / source assignment | Conditional easy-harm MSE gain (%) | 95% locality CI |',
        '|---|---:|---:|']
    for comp,pairs in cs.items():
        for pair,groups in pairs.items():
            for group,m in groups.items():
                v=m['envelope_positive__harm_MSE_gain_percent']
                lines.append(f"| {comp} / {pair} / {group} | {v.get('point','not_estimable')} | {v.get('CI','not_estimable')} |")
    lines+=['','Three seeds averaged within locality, then3,000 resamples of four localities.',
        'Repeated roles/windows are not independent; intervals are exploratory, not multiplicity-adjusted.',
        '','## Gates','',*['- '+k+': '+str(v).lower() for k,v in gates.items()],
        '','Full component errors, tail capture, coverage and membership partitions are retained in aggregate_metrics.json.',
        'The nested easy-harm fraction is not a calibrated event probability. Reference-cost moments remain unchanged.',
        'No metric/seconds, human-gold, physical safety, true3D or foundation claim. Stage5C/SMC off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=gates,primary={k:v['full']['envelope_positive__harm_MSE_gain_percent'] for k,v in summary.items()}),indent=2))


if __name__=='__main__': main()
