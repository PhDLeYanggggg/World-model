"""Fixed primary and safety readout for source-only checkpoint selection."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_source_checkpoint as run
from scripts import report_m3w_inner_separability as accounting


def signed_values(values):
    """Independent reconstruction of the three registered decision targets."""
    return np.column_stack((values[:, 0]-values[:, 1],
                            values[:, 1]-.02*values[:, 2],
                            values[:, 4]-.02*values[:, 3]))


def query_average_squared_error(prediction, target, scale, rms, recordings, frames):
    known=np.isfinite(target).all(1)
    _,group=np.unique(np.rec.fromarrays([recordings,frames]),return_inverse=True)
    count=np.bincount(group[known],minlength=int(group.max())+1)
    active=count>0
    error=((signed_values(prediction[known])-signed_values(target[known]))/scale/rms)**2
    per_query=np.column_stack([np.bincount(group[known],weights=error[:, k],minlength=len(count))[active]/count[active]
                               for k in range(3)])
    if not len(per_query):raise ValueError('No known query targets')
    return float(per_query.mean())


def independent_checks(readout, freeze):
    _,_,data,jobs,oid,_,_,_=run.parent.parent.old.load()
    causal={k:data[k] for k in run.parent.parent.old.parent.CAUSAL_KEYS}
    fits={Path(r['path']).parent.name:r for r in freeze['groups']}
    counts=dict(source_preprocessing=0,validation_scores=0,views=0,metric_values=0,
                quality_scores=0,query_count_checks=0,known_occurrences=0,unknown_occurrences=0)
    for c in run.base.floor_api.contexts(causal,jobs,oid):
        for site in run.parent.parent.sources(c):
            ref=fits[c['name']+'_fit_'+site];doc=json.loads((ROOT/ref['path']).read_text())
            state=run.api.core.read_checkpoint(ROOT/doc['checkpoint']['path'])
            _,ids,x,env,y,_,identity=run.parent.parent.training_arrays(c,data,site)
            assert state['identity']['upstream']==identity
            train,val,partition=run.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
            assert partition==state['partition']
            pr=run.api.core.preprocess(x[train],env[train],y[train],data['sites'][ids][train],
                data['recordings'][ids][train],data['frames'][ids][train],training_site=site)
            run.api.core.exact(pr,state['preprocess']);counts['source_preprocessing']+=1
            p,_=run.api.paired_predictions(state,x[val],env[val])
            for arm,expected in [('final',state['trace'][-1]['validation_signed_MSE']),('validation',state['best_score'])]:
                score=query_average_squared_error(p[arm],y[val],pr['scale'],pr['rms'][5:],
                    data['recordings'][ids][val],data['frames'][ids][val])
                # Validation uses float32 model outputs before rescaling; allow only rounding.
                np.testing.assert_allclose(score,expected,rtol=2e-6,atol=2e-7)
                counts['validation_scores']+=1
    metrics={(r['view'],r['policy']):r['metric'] for r in readout['rows']}
    quality={(r['view'],r['arm']):r for r in readout['quality']}
    decisions={r['view']:r for r in json.loads((run.PUBLIC/'decision_freeze.json').read_text())['rows']}
    unique=set();exchange=[]
    for c,at,ids,p,actions,pr,meta in run.views(data,jobs,oid):
        assert meta==decisions[meta['view']]
        cv,cf,(floor,ff),(neural,nf)=run.base.floor_api.costs(c,data,at)
        known=np.isfinite(cv);easy=known&(cv>0)&(cv<=c['job']['design']['easy_cut'])
        y=np.column_stack((np.maximum(floor-neural,0),np.maximum(neural-floor,0),floor,
                           np.where(easy,floor,0),np.where(easy,np.maximum(neural-floor,0),0)))
        y[~known]=np.nan
        _,group=np.unique(np.rec.fromarrays([data['recordings'][ids],data['frames'][ids]]),return_inverse=True)
        n=int(group.max())+1
        selected={k:np.bincount(group,weights=v.astype(int),minlength=n) for k,v in actions.items()}
        matched=np.minimum(selected['final'],selected['validation'])
        for arm in ('final','validation'):
            np.testing.assert_array_equal(selected[arm+'_matched'],matched)
            assert not (actions[arm+'_matched']&~actions[arm]).any()
            score=query_average_squared_error(p[arm],y,pr['scale'],pr['rms'][5:],data['recordings'][ids],data['frames'][ids])
            accounting.assert_value(quality[meta['view'],arm]['mean_signed_MSE'],score)
            counts['quality_scores']+=1
        counts['query_count_checks']+=n
        for policy,take in {**actions,'floor':np.zeros(len(ids),bool)}.items():
            counts['metric_values']+=accounting.audit_metric(metrics[meta['view'],policy],cv,floor,neural,cf,ff,nf,
                take,data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
        reference=float(floor[known].sum());left=actions['final_matched']&known;right=actions['validation_matched']&known
        benefit=np.maximum(floor-neural,0);harm=np.maximum(neural-floor,0)
        avoided=100*float(harm[left].sum()-harm[right].sum())/reference if reference>0 else None
        lost=100*float(benefit[left].sum()-benefit[right].sum())/reference if reference>0 else None
        improvement=(metrics[meta['view'],'validation_matched']['all_gain_floor']
                     -metrics[meta['view'],'final_matched']['all_gain_floor'])
        if reference>0:accounting.assert_value(improvement,avoided-lost)
        exchange.append(dict(view=meta['view'],site=meta['site'],source=meta['source'],seed=meta['seed'],
            selected_step=meta['best_step'],avoided_harm_pp=avoided,lost_benefit_pp=lost,
            net_gain_difference_pp=improvement,denominator='full_known_floor_diagnostic_not_selected_risk'))
        counts['views']+=1;counts['known_occurrences']+=int(known.sum());counts['unknown_occurrences']+=int((~known).sum())
        unique.update(ids.tolist())
        if counts['views']%36==0:run.beat(state='independent_accounting',**counts)
    assert counts['source_preprocessing']==72 and counts['validation_scores']==144
    assert counts['views']==216 and counts['quality_scores']==432
    counts['unique_row_ids']=len(unique)
    run.immutable(run.PUBLIC/'selection_exchange.json',dict(status='posthoc_descriptive_frozen_actions_unchanged',rows=exchange,
        summary={key:locality_interval([(r['site'],r[key]) for r in exchange])
                 for key in ('avoided_harm_pp','lost_benefit_pp','net_gain_difference_pp')},
        independent_confirmation=False,used_to_select_policy=False))
    return counts


def locality_interval(rows,draws=3000,seed=20260929):
    groups={}
    for site,value in rows:
        groups.setdefault(site,[]).append(value)
    missing=[s for s,v in groups.items() if any(x is None or not np.isfinite(x) for x in v)]
    if missing:return dict(mean=None,CI95=None,undefined_localities=missing,localities=len(groups))
    by={s:float(np.mean(v)) for s,v in sorted(groups.items())};a=np.array(list(by.values()))
    if not len(a):raise ValueError('No localities')
    rng=np.random.default_rng(seed);b=a[rng.integers(0,len(a),(draws,len(a)))].mean(1)
    return dict(mean=float(a.mean()),CI95=np.quantile(b,[.025,.975]).tolist(),by_locality=by,
                localities=len(a),draws=draws,nominal_development_only=True)


def summarize(data,cfg):
    q={};m={}
    for r in data['quality']:q.setdefault(r['view'],{})[r['arm']]=r
    for r in data['rows']:m.setdefault(r['view'],{})[r['policy']]=r
    assert len(q)==len(m)==216 and set(q)==set(m)
    primary=[];utility=[];seeds={}
    for view,row in q.items():
        assert set(row)=={'final','validation'}
        site=row['final']['site'];seed=row['final']['seed']
        diff=row['validation']['mean_signed_MSE']-row['final']['mean_signed_MSE']
        primary.append((site,diff))
        left=m[view]['final_matched']['metric'];right=m[view]['validation_matched']['metric']
        np.testing.assert_allclose(left['intervention_rate'],right['intervention_rate'],rtol=0,atol=0)
        gain=100*(left['error_sum']-right['error_sum'])/left['error_sum'] if left['error_sum']>0 else None
        utility.append((site,gain));seeds.setdefault(seed,[]).append((site,gain))
    out=dict(primary=locality_interval(primary,cfg['bootstrap_resamples'],cfg['bootstrap_seed']),
        matched_ADE_improvement=locality_interval(utility,cfg['bootstrap_resamples'],cfg['bootstrap_seed']),
        seed_matched_ADE={str(k):locality_interval(v,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])['mean'] for k,v in seeds.items()},policies={})
    for policy in ('final','validation','final_matched','validation_matched'):
        rows=[r for r in data['rows'] if r['policy']==policy];ms=[r['metric'] for r in rows]
        d={}
        for label,key in [('all','selected_positive_harm_ratio'),('easy','selected_easy_positive_harm_ratio')]:
            vals=[x[key] for x in ms]
            d[label+'_risk_violations']=sum(v is not None and v>.02+1e-10 for v in vals)
            d[label+'_risk_undefined']=sum(v is None for v in vals)
        easy=[r['easy_gain_floor'] for r in ms]
        d['worst_easy_degradation_percent']=max(-x for x in easy if x is not None) if any(x is not None for x in easy) else None
        d['undefined_easy_ADE_views']=sum(x is None for x in easy)
        d['unknown_interventions']=sum(r['unknown_interventions'] for r in ms)
        d['known_rows_repeated']=sum(r['known_rows'] for r in ms)
        d['unknown_rows_repeated']=sum(r['unknown_rows'] for r in ms)
        for key in ('all_gain_floor','easy_gain_floor','hard_gain_floor','FDE_gain_floor','intervention_rate'):
            d[key]=locality_interval([(r['site'],r['metric'][key]) for r in rows],cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
        out['policies'][policy]=d
    out.update(deployment_changed=False,independent_confirmation=False)
    return out


def main():
    start=time.monotonic();cfg,reg=run.registration();pub=run.PUBLIC
    run.api.torch.set_num_threads(cfg['cpu_threads']);run.api.torch.set_num_interop_threads(cfg['interop_threads'])
    assert reg==json.loads((pub/'registration.json').read_text())
    freeze=json.loads((pub/'training_freeze.json').read_text());assert len(freeze['groups'])==72
    choices=[];checks=0
    for ref in freeze['groups']:
        assert run.base.artifact(ROOT/ref['path'])==ref
        doc=json.loads((ROOT/ref['path']).read_text());cp=doc['checkpoint'];assert run.base.artifact(ROOT/cp['path'])==cp
        state=run.api.core.read_checkpoint(ROOT/cp['path'])
        assert state['step']==2000 and state['identity']==doc['identity']
        assert state['best_step']==min(state['trace'],key=lambda r:r['validation_signed_MSE'])['step']
        assert state['best_score']==min(r['validation_signed_MSE'] for r in state['trace'])
        assert state['partition']==doc['partition'] and state['preprocess']['training_site']==state['identity']['source']
        assert state['partition']['train_rows']>0 and state['partition']['validation_rows']>0
        if state['partition']['method']=='whole_recording_hash_split':
            assert set(state['partition']['train_recordings']).isdisjoint(state['partition']['validation_recordings'])
        choices.append(state['best_step']);checks+=8
    assert json.loads((pub/'fit_replay.json').read_text())['exact_except_elapsed']
    assert json.loads((pub/'evaluation_replay.json').read_text())['exact']
    data=json.loads((pub/'readout.json').read_text());summary=summarize(data,cfg)
    summary['independent_accounting']=independent_checks(data,freeze)
    summary['selected_step_counts']={str(k):v for k,v in sorted(Counter(choices).items())}
    summary['checkpoint_checks']=checks
    run.immutable(pub/'summary.json',summary)
    table=['| Policy | All gain vs floor (%) | Easy gain vs floor (%) | Hard gain vs floor (%) | All-risk violations /216 | Easy-risk violations /216 | Undefined all/easy risk | Worst easy degradation (%) | Unknown interventions |',
           '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for name,z in summary['policies'].items():
        values=[z[k]['mean'] for k in ('all_gain_floor','easy_gain_floor','hard_gain_floor')]
        fmt=lambda v:'undefined' if v is None else f'{v:.5f}'
        table.append('| '+ ' | '.join([name,*map(fmt,values),str(z['all_risk_violations']),str(z['easy_risk_violations']),
            f"{z['all_risk_undefined']}/{z['easy_risk_undefined']}",fmt(z['worst_easy_degradation_percent']),str(z['unknown_interventions'])])+' |')
    report=['# Source-Internal Validation Checkpoint Control','',
        '## Material Passport','',
        'fresh_run: 72 true Torch optimization runs, 144,000 updates including the 100-update pilot, '
        '216 frozen causal directional decisions and full readout replay. cached_verified: upstream '
        'data and forecasters. Independent calibration/confirmation are not_run. No deployment change.','',
        f"Primary validation-selected minus final signed-score MSE: {summary['primary']['mean']}; nominal 95% locality CI {summary['primary']['CI95']}. Lower is better.",
        f"Matched ADE improvement over final: {summary['matched_ADE_improvement']['mean']}%; nominal 95% locality CI {summary['matched_ADE_improvement']['CI95']}.",
        f"Selected checkpoint steps: {summary['selected_step_counts']}. A step 0 selection is a training-prior decoder, not a learned improvement.",
        f"Seed matched-ADE gains: {summary['seed_matched_ADE']}.",'',*table,'',
        'All summaries average within locality before across the 12 localities. The 216 role/seed views '
        'are dependent and development-exposed. The 3,000-draw bootstrap is nominal; it does not account '
        'for the complete adaptive research history. Unknown selected outcomes remain unknown. '
        'Undefined risk denominators do not pass a safety gate. Positive harm uses the unchanged 2% '
        'selected-reference budget, not net average ADE gain. Count matching need not match chosen sets.','',
        '## Reproduction','',
        'The first complete 2,000-update fit replays exactly excluding elapsed time; this does not '
        'retrain all 72 fits. All 216 action hashes are recreated and matched before each outcome '
        'readout. The complete readout also replays exactly. Checkpoints, normalization, sampler RNG, '
        'optimizer state, validation curves and both final/selected models remain local and excluded '
        'from Git. One disk-reserve interruption after the first fit was recovered from that checkpoint '
        'only after free space passed the original reserve again; no source or update was dropped.',
        f"Independent numerical checks: {summary['independent_accounting']}.",'',
        '## Limits','',
        'Recording-held source validation is a model-selection mechanism, not independent risk '
        'calibration. Changing the optimization/validation split means old full-source heads are not '
        'a matched causal comparison; only final versus selected within these runs is the primary '
        'contrast. No threshold search on transfer localities, independent-role opening or policy '
        'promotion. Image-local detector-silver, obs8/pred12 stride12 raw-frame only. No metric, '
        'seconds, human-gold, true3D, foundation, Stage5C or SMC claim.','']
    (pub/'results.md').write_text('\n'.join(report))
    tests=['tests/test_m3w_source_checkpoint.py','tests/test_m3w_source_checkpoint_report.py',
           'tests/test_m3w_inner_separability.py','tests/test_m3w_boundary_diagnostic.py']
    test=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    (pub/'scoped_tests.txt').write_text(test.stdout+test.stderr)
    run.immutable(pub/'verification.json',dict(status='verified_development_control_not_deployment',
        checkpoints=72,checkpoint_checks=checks,first_fit_replay=True,full_readout_replay=True,
        independent_accounting=summary['independent_accounting'],
        source_bindings={**reg['bindings'],str(Path(__file__).relative_to(ROOT)):run.digest(Path(__file__)),
            'tests/test_m3w_source_checkpoint_report.py':run.digest(ROOT/'tests/test_m3w_source_checkpoint_report.py'),
            'scripts/report_m3w_inner_separability.py':run.digest(ROOT/'scripts/report_m3w_inner_separability.py')},
        artifacts={p.name:run.digest(p) for p in pub.iterdir() if p.is_file() and p.name!='verification.json'},
        full_legacy_suite='not_run',independent_confirmation=False,deployment_changed=False,seconds=time.monotonic()-start))
    print(json.dumps(dict(primary=summary['primary'],matched_ADE=summary['matched_ADE_improvement'],selected_steps=summary['selected_step_counts'])))


if __name__=='__main__':main()
