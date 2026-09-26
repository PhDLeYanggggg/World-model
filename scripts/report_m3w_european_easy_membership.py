"""Complete paired probability diagnostics; no best-role or threshold selection."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_easy_membership as run


def conditional_constant_sensitivity(rows,cfg):
    out={}
    for pair in cfg['pairs']:
        out[pair]={}
        for a,b in sorted({(r['producer'],r['controller']) for r in rows}):
            rs=[r for r in rows if r['pair']==pair and (r['producer'],r['controller'])==(a,b)]
            assert len(rs)==3
            sites=sorted({f['held'] for r in rs for f in r['folds']}); result={}
            for arm in cfg['arms']:
                for metric in ('Brier_skill_percent','log_loss_gain'):
                    values=[]
                    for site in sites:
                        vs=[]
                        for r in rs:
                            f=next(f for f in r['folds'] if f['held']==site)
                            fit=f['training'][arm]['model']['envelope_positive']; held=f['metrics'][arm]['envelope_positive']
                            if fit.get('status')=='not_estimable' or held.get('status')=='not_estimable': vs.append(None); continue
                            p,rate=fit['positive_rate'],held['positive_rate']; error=p*p-2*p*rate+rate
                            q=np.clip(p,1e-7,1-1e-7); loss=-(rate*np.log(q)+(1-rate)*np.log1p(-q))
                            vs.append(100*(1-held['Brier']/error) if metric=='Brier_skill_percent' and error>0 else
                                float(loss-held['log_loss']) if metric=='log_loss_gain' else None)
                        if any(v is None or not np.isfinite(v) for v in vs): break
                        values.append(float(np.mean(vs)))
                    if len(values)!=4: result[arm+'__'+metric]=dict(status='not_estimable'); continue
                    v=np.array(values); rng=np.random.default_rng(cfg['bootstrap_seed'])
                    ds=v[rng.integers(0,4,size=(cfg['bootstrap_resamples'],4))].mean(1)
                    result[arm+'__'+metric]=dict(point=float(v.mean()),CI=np.quantile(ds,[.025,.975]).tolist(),locality_points=dict(zip(sites,values)))
            out[pair][f'producer{a}_controller{b}']=result
    return out


def aggregates(rows,cfg):
    contrasts=run.metric.paired(rows,cfg); populations={}; transport={}
    for pair in cfg['pairs']:
        fs=[f for r in rows if r['pair']==pair for f in r['folds']]; populations[pair]={}; transport[pair]={}
        for subset in ('all','envelope_positive'):
            populations[pair][subset]={}
            for arm in (*cfg['arms'],'train_constant','reference_ratio'):
                ms=[f['metrics'][arm][subset] for f in fs]; known=[m for m in ms if m.get('status')!='not_estimable']
                summary=dict(views=len(ms),not_estimable=len(ms)-len(known),weak_support=sum(not m['stable_support'] for m in known))
                for key in ('Brier','log_loss','AUROC','AUPRC','ECE','positive_rate','mean_probability'):
                    vs=[m[key] for m in known if m[key] is not None]
                    summary[key]=dict(median=float(np.median(vs)) if vs else None,range=[min(vs),max(vs)] if vs else None,estimable=len(vs))
                populations[pair][subset][arm]=summary
        for arm in cfg['arms']:
            count=dict(views=len(fs),not_estimable=0,fit_improved=0,held_improved=0,fit_only=0,both_improved=0)
            for f in fs:
                a,b=[f['training'][arm][v]['envelope_positive'] for v in ('model','train_constant')]
                c,d=[f['metrics'][v]['envelope_positive'] for v in (arm,'train_constant')]
                if any(v.get('status')=='not_estimable' for v in (a,b,c,d)):
                    count['not_estimable']+=1; continue
                fit=a['Brier']<b['Brier']; held=c['Brier']<d['Brier']
                count['fit_improved']+=int(fit); count['held_improved']+=int(held)
                count['fit_only']+=int(fit and not held); count['both_improved']+=int(fit and held)
            transport[pair][arm]=count
    return dict(contrasts=contrasts,populations=populations,fit_transport=transport,gates=run.metric.gates(contrasts),
        conditional_constant_sensitivity=conditional_constant_sensitivity(rows,cfg))


def main():
    cfg,identity=run.registration(); done=run.checked_training(identity)
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert checks['identity']==identity
    rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    agg=aggregates(rows,cfg); run.immutable_json(run.PUBLIC/'aggregate_metrics.json',agg)
    run.immutable_json(run.PUBLIC/'gates.json',agg['gates'])
    rs=[json.loads((ROOT/r['path']).read_text()) for r in done['heads']]
    fits=[dict(group=r['input']['group']['group'],pair=r['input']['pair'],held=r['input']['held'],arm=r['input']['arm'],fit=r['fit']) for r in rs]
    run.immutable_json(run.PUBLIC/'training_metrics.json',fits)
    prior=ROOT/'data/stage_cvpr2027_experiments/european_frozen_harm_readout_v1/create_queue.json'
    remote=json.loads(prior.read_text()); assert remote['jobs_submitted']==0 and not remote['remote_modified']
    run.immutable_json(run.PUBLIC/'compute_receipt.json',dict(heads=len(fits),updates=sum(r['fit']['step'] for r in fits),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in fits),unknown_draws=sum(r['fit']['unknown_rows_sampled'] for r in fits),
        parameters={a:sorted({r['fit']['parameters'] for r in fits if r['arm']==a}) for a in cfg['arms']},
        summed_fit_seconds_by_arm={a:sum(r['fit']['seconds'] for r in fits if r['arm']==a) for a in cfg['arms']},
        real_torch=True,threads=4,interop=1,workers=0,remote_jobs_submitted=0,
        recent_CREATE_observation=dict(result_source='cached_verified_not_live_availability',receipt=run.artifact(prior),
            completed_utc=remote['completed_utc'],returncode=remote['response']['returncode'],M3W_remote_path='not_run_unverified')))
    lines=['# Direct Easy-Membership Results','','## Material Passport',
        '288 fresh Torch classifiers; cached_verified source/producer lineage. Source development only.',
        'No new forecast, cost model, policy evaluation or independent confirmation.','',
        '| Pair / roles / arm | Conditional Brier skill % [95% CI] | AUROC minus 0.5 [95% CI] | Log-loss gain [95% CI] |',
        '|---|---:|---:|---:|']
    def fmt(v): return 'not_estimable' if 'CI' not in v else f"{v['point']:.6f} [{v['CI'][0]:.6f}, {v['CI'][1]:.6f}]"
    for pair,gs in agg['contrasts'].items():
        for group,cs in gs.items():
            for arm in cfg['arms']:
                vs=[fmt(cs[arm+'__envelope_positive__'+k]) for k in ('Brier_skill_percent','AUROC_above_chance','log_loss_gain')]
                lines.append(f'| {pair} / {group} / {arm} | '+' | '.join(vs)+' |')
    lines+=['','Three seeds are averaged within locality; 3,000 resamples of four localities per assignment.',
        'Repeated source roles/windows are dependent. Exploratory CIs, no multiplicity adjustment.',
        'The reference-cost ratio is not a calibrated event probability; its proper-score diagnostics do not make it one.',
        'Brier skill compares with a train-prevalence constant, not a constant refitted on held labels.',
        'The inherited easy definition excludes exact-zero CV errors.','',
        'A separately declared pre-readout sensitivity also uses training prevalence conditional on positive disagreement.',
        'It is a stronger prevalence control, not a revised primary gate or held-label fit. Its complete contrasts are in aggregate_metrics.json.','',
        '## Gates','', '```json',json.dumps(agg['gates'],indent=2),'```','',
        'Image pixels and annotation steps, detector-derived labels; no metric/seconds, human-gold, physical safety, true3D or foundation claim.',
        'Independent calibration/confirmation remain closed. Stage5C/SMC off.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(agg['gates'],indent=2))


if __name__=='__main__': main()
