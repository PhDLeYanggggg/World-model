"""All-role occurrence, tail retrieval and magnitude evidence."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_harm_tail_crossfit as run


def ci_by_assignment(rows,cfg):
    out={}
    for pair in cfg['pairs']:
        out[pair]={}
        for a,b in sorted({(r['producer'],r['controller']) for r in rows}):
            group=[r for r in rows if r['pair']==pair and (r['producer'],r['controller'])==(a,b)]
            assert len(group)==3 and sorted(r['seed'] for r in group)==cfg['seeds']
            localities=sorted({f['held'] for r in group for f in r['folds']}); assert len(localities)==4
            contrasts={}
            for subset in ('all','envelope_positive'):
                for score in ('moment','fraction'):
                    values=[]
                    for site in localities:
                        diffs=[]
                        for r in group:
                            f=next(f for f in r['folds'] if f['held']==site)['held_metrics'][subset]
                            if f.get('status')=='not_estimable': break
                            x,y=[f['scores'][k]['top10_harm_mass_share'] for k in (score,'envelope')]
                            if x is None or y is None: break
                            diffs.append(x-y)
                        if len(diffs)!=3: break
                        values.append(float(np.mean(diffs)))
                    key=subset+'__'+score+'_vs_envelope'
                    if len(values)!=4: contrasts[key]=dict(status='not_estimable',reason='missing_event_support_in_one_or_more_localities_or_seeds'); continue
                    v=np.array(values)*100; rng=np.random.default_rng(cfg['bootstrap_seed'])
                    draws=v[rng.integers(0,4,size=(cfg['bootstrap_resamples'],4))].mean(1)
                    contrasts[key]=dict(point_pp=float(v.mean()),CI_pp=np.quantile(draws,[.025,.975]).tolist(),
                        locality_points_pp=dict(zip(localities,v.tolist())),seeds=3,localities=4)
            out[pair][f'producer{a}_controller{b}']=contrasts
    return out


def summarize_population(values):
    values=[v for v in values if v.get('status')!='not_estimable']
    def stats(xs):
        xs=[x for x in xs if x is not None]
        return dict(n=len(xs),median=float(np.median(xs)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
    return dict(population_views=len(values),weak_support=sum(not r['stable_event_support'] for r in values),
        zero_event=sum(r['positive']==0 for r in values),prevalence=stats([r['positive_rate'] for r in values]),
        harm_coverage=stats([r['harm_coverage'] for r in values]),
        oracle_top1_mass_share=stats([r['oracle_top1_mass_share'] for r in values]),
        scores={s:{m:stats([r['scores'][s][m] for r in values])
            for m in ('AUROC','AUPRC','AP_over_prevalence','top10_harm_mass_share')}
            for s in ('moment','fraction','envelope')})


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['all_passed']
    rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    ci=ci_by_assignment(rows,cfg); summary={}
    for pair in cfg['pairs']:
        group=[r for r in rows if r['pair']==pair]
        summary[pair]={}
        for subset in ('all','envelope_positive'):
            summary[pair][subset]={
                'inner_held_B':summarize_population([f['held_metrics'][subset] for r in group for f in r['folds']]),
                'original_fit_B':summarize_population([v[subset] for r in group for v in r['original_metrics']['B'].values()]),
                'original_held_C':summarize_population([v[subset] for r in group for v in r['original_metrics']['C'].values()])}
    fits=[json.loads((ROOT/r['path']).read_text()) for r in done['heads']]
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json',dict(summary=summary,contrasts=ci))
    run.immutable_json(run.PUBLIC/'training_metrics.json',[dict(group=r['input']['group']['group'],pair=r['input']['pair'],
        held=r['input']['held'],fit=r['fit']) for r in fits])
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        threads=4,interop=1,workers=0,real_torch=True,new_forecaster_training=False,
        create_queries=[run.artifact(p) for p in sorted(run.PRIVATE.glob('create_queue*.json'))],jobs_submitted=0))
    status={p:dict(positive=sum(v['all__moment_vs_envelope'].get('CI_pp',[0,0])[0]>0 for v in c.values()),
        negative=sum(v['all__moment_vs_envelope'].get('CI_pp',[0,0])[1]<0 for v in c.values()),
        not_estimable=sum(v['all__moment_vs_envelope'].get('status')=='not_estimable' for v in c.values())) for p,c in ci.items()}
    run.immutable_json(run.PUBLIC/'gates.json',dict(training_complete=True,inner_locality_exclusion=True,
        training_only_inner_cuts_and_bins=True,held_predictions_frozen=True,ranking_contrasts=status,
        new_policy_evaluated=False,independent_calibration=False,independent_confirmation=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    lines=['# Harm Tail Crossfit Results','','## Material Passport',
        '144 fresh Torch heads / 288,000 updates; 36 source-role/seed/pair groups. No new trajectory or policy result.',
        'Inner B event cuts use three training localities; original B/C use their unchanged whole-B event cut.',
        'Do not interpret those populations as matched event definitions. No held-out or C model selection.','',
        '| Pair / subset / population | Views | Weak support | Median event rate | Moment AUROC | AP/prevalence | Moment top10 harm share | Envelope top10 share | Harm coverage | Oracle top1 share |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return 'not_estimable' if x is None else f'{x:.5f}'
    for pair,subsets in summary.items():
        for subset,populations in subsets.items():
            for pop,v in populations.items():
                s=v['scores']; vals=[v['prevalence']['median'],s['moment']['AUROC']['median'],s['moment']['AP_over_prevalence']['median'],
                    s['moment']['top10_harm_mass_share']['median'],s['envelope']['top10_harm_mass_share']['median'],
                    v['harm_coverage']['median'],v['oracle_top1_mass_share']['median']]
                lines.append(f"| {pair} / {subset} / {pop} | {v['population_views']} | {v['weak_support']} | "+' | '.join(map(fmt,vals))+' |')
    lines+=['','## Paired Tail-Retrieval Contrast','',
        'Difference in harm captured by the highest-scoring 10%: neural score minus envelope, percentage points.',
        'Three-seed locality averages, then 3,000 resamples of four localities. Not independent confirmation.','',
        '| Pair / assignment | Subset / score | Point (pp) | 95% CI (pp) |','|---|---|---:|---:|']
    for pair,groups in ci.items():
        for group,cs in groups.items():
            for key,v in cs.items():
                lines.append(f"| {pair} / {group} | {key} | {fmt(v.get('point_pp'))} | {v.get('CI_pp','not_estimable')} |")
    lines+=['','Expected harm scores are not event probabilities. AUROC/AP assess ranking, not calibration.',
        'Top10 shares are evaluation diagnostics, never a label-aware deployment rule. No policy or tolerance changes.',
        'Overlapping windows/roles and fold-varying event cuts prohibit pooling all rows as independent evidence.',
        'Image pixels, annotation steps, detector-derived labels; no metric/seconds, human-gold, true3D or foundation claim.',
        'Selection/calibration/confirmation remain closed this round. Stage5C and SMC remain off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,indent=2))


if __name__=='__main__': main()
