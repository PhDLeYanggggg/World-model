"""Rebuild fold exclusion, sampling, bins and held-score metrics independently."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_harm_tail_crossfit as run
from scripts.report_m3w_european_harm_tail_crossfit import ci_by_assignment,summarize_population
from src.world_model.m3w_native_forecast import draw_batch
import numpy as np
import torch


def rank_metrics(score,harm):
    score,harm=np.asarray(score),np.asarray(harm); event=harm>0
    values,idx=np.unique(score,return_inverse=True)
    pos=np.bincount(idx,weights=event,minlength=len(values)); neg=np.bincount(idx,weights=~event,minlength=len(values))
    P,N=pos.sum(),neg.sum()
    auc=float(np.dot(pos,np.cumsum(neg)-neg/2)/(P*N)) if P and N else None
    cp=np.cumsum(pos[::-1]); count=np.cumsum((pos+neg)[::-1])
    ap=float(np.dot(pos[::-1]/P,cp/count)) if P else None
    return auc,ap


def top_share(score,harm,fraction):
    if harm.sum()<=0: return None
    budget=len(harm)*fraction; threshold=np.sort(score)[::-1][min(int(np.floor(budget)),len(harm)-1)]
    above=score>threshold; tied=score==threshold
    take=(budget-above.sum())/tied.sum()
    return float((harm[above].sum()+take*harm[tied].sum())/harm.sum())


def replay(cfg,identity):
    path=run.PUBLIC/'replay_receipt.json'; binding=run.artifact(Path(__file__))
    if path.exists():
        r=json.loads(path.read_text()); assert r['identity']==identity and r['source_binding']==binding and r['all_passed']
        assert r['training']==run.artifact(run.PRIVATE/'training_complete.json'); return r
    records=[]; metric_checks=0; coordinate_checks=0; original_checks=0
    for g,data,pairs in run.parent.contexts(identity['parent']['parent']['parent']):
        name=g['group']; seed=int(name.split('_seed')[1].split('_')[0]); bi=pairs['B']['ids']; sites=data['sites'][bi]
        for pair in cfg['pairs']:
            bx,env,by,*_=run.parent.pair_inputs(g,data,pairs,pair)
            row=json.loads((run.PUBLIC/'groups'/(name+'_'+pair+'.json')).read_text())
            ci=pairs['C']['ids']; directory=run.parent.PRIVATE/'heads'/(name+'_'+pair+'_mean')
            for key in ('original_model','original_C_scores','original_C_labels'):
                assert run.artifact(ROOT/row[key]['path'])==row[key]
            original,state=run.parent.restore(directory)
            original_b=run.parent.method.predict(original,bx,env,state['preprocess'])
            edges=run.training_edges(original_b,env,state['preprocess'],cfg)
            assert edges==row['original_B_edges']
            with np.load(ROOT/row['original_C_scores']['path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ci); original_c=z['scores'].copy()
            with np.load(ROOT/row['original_C_labels']['path'],allow_pickle=False) as z:
                cy=run.parent.method.event_targets(z['cv'],z['reference'],z['candidate'],g['easy_cut'])
            for role,pred,target,e,localities in (('B',original_b,by,env,sites),
                    ('C',original_c,cy,pairs['C'][pair][1],data['sites'][ci])):
                for site in sorted(set(localities)):
                    ss=localities==site
                    for subset,mask in (('all',None),('envelope_positive',e[ss]>0)):
                        actual=run.diagnostic.summarize(pred[ss],target[ss],e[ss],localities[ss],edges,subset=mask)
                        assert actual==row['original_metrics'][role][site][subset]
                        original_checks+=1
            errors=[]
            for pred in pairs['B'][pair][2:]:
                a=run.parent.base.cross.independent.coordinate_errors(pred.astype(float)+data['origin'][bi,None],
                    data['target_eval'][bi],data['valid'][bi])[0]
                errors.append(a); coordinate_checks+=1
            np.testing.assert_allclose(by[:,:2],np.column_stack((errors[0],np.maximum(errors[1]-errors[0],0))),rtol=1e-10,atol=1e-8,equal_nan=True)
            for held in sorted(set(sites)):
                tr,te,pr,cut,y,masks=run.fold_inputs(bx,env,by,data['baseline_ade'][bi,1],sites,held)
                assert held not in pr['training_sites'] and held not in g['producer_roster']
                positive=data['baseline_ade'][bi[tr],1]; positive=positive[np.isfinite(positive)&(positive>0)]
                assert cut==float(np.quantile(positive,.25))
                tag=name+'_'+pair+'_'+held; directory=run.PRIVATE/'heads'/tag
                model,state=run.parent.restore(directory); assert state['step']==2000
                for k in ('mean','std','weights','known'): np.testing.assert_array_equal(state['preprocess'][k],pr[k])
                assert state['preprocess']['cost_scale']==pr['cost_scale']
                rng=torch.Generator().manual_seed(seed+7919); groups=[np.flatnonzero(pr['known']&(sites[tr]==s)) for s in sorted(set(sites[tr]))]
                fixed_rng=torch.Generator().set_state(rng.get_state()); np.testing.assert_array_equal(draw_batch(groups,256,fixed_rng),state['fixed_ids'])
                draws=np.zeros(tr.sum(),np.int64)
                for _ in range(2000): np.add.at(draws,draw_batch(groups,256,rng),1)
                np.testing.assert_array_equal(draws,state['draws']); assert torch.equal(rng.get_state(),state['sampler_rng'])
                assert draws[~pr['known']].sum()==0
                with np.load(directory/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],bi[te]); score=z['scores'].copy()
                count=min(4096,te.sum())
                np.testing.assert_array_equal(run.parent.method.predict(model,bx[te][:count],env[te][:count],pr),score[:count])
                pred=run.parent.method.predict(model,bx[tr],env[tr],pr); fit=json.loads((directory/'fit_diagnosis.json').read_text())
                assert run.training_edges(pred,env[tr],pr,cfg)==fit['edges']
                assert run.diagnostic.summarize(pred,y,env[tr],sites[tr],fit['edges'])==fit['training']
                target=run.diagnostic.event_targets(by[te],data['baseline_ade'][bi[te],1],cut)
                heldrow=next(f for f in row['folds'] if f['held']==held)
                for subset,mask in (('all',np.ones(te.sum(),bool)),('envelope_positive',env[te]>0)):
                    measured=run.diagnostic.summarize(score,target,env[te],sites[te],fit['edges'],subset=mask)
                    assert measured==heldrow['held_metrics'][subset]
                    valid=np.isfinite(target).all(1)&mask; h=target[valid,3]
                    if not valid.any(): continue
                    for key,s in run.diagnostic.score_columns(score,env[te]).items():
                        independent=rank_metrics(s[valid],h); expected=measured['scores'][key]
                        for x,v in zip(independent,(expected['AUROC'],expected['AUPRC'])):
                            if x is None: assert v is None
                            else: np.testing.assert_allclose(x,v,rtol=1e-12,atol=1e-12)
                            metric_checks+=1
                        actual=top_share(s[valid],h,.1)
                        if actual is None: assert expected['top10_harm_mass_share'] is None
                        else: np.testing.assert_allclose(actual,expected['top10_harm_mass_share'],rtol=1e-12,atol=1e-12)
                        metric_checks+=1
                records.append(dict(group=name,pair=pair,held=held,checkpoint=run.artifact(directory/'checkpoint.pt'),
                    prefix_rows=int(count),fold_exclusion=True,train_bins_replayed=True,sampler_replayed=True))
                run.beat('verified_head',group=name,pair=pair,held=held,completed=len(records))
    assert len(records)==144
    r=dict(identity=identity,source_binding=binding,training=run.artifact(run.PRIVATE/'training_complete.json'),
        checkpoints=records,independent_coordinate_reductions=coordinate_checks,
        independent_rank_tail_checks=metric_checks,all_passed=True,changed_model=False,policy_changed=False)
    r['original_population_checks']=original_checks
    run.immutable_json(path,r); return r


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replay-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    cfg,identity=run.registration(); run.checked_training(identity); r=replay(cfg,identity)
    if args.replay_only: print(json.dumps(dict(checkpoints=len(r['checkpoints']),checks=r['independent_rank_tail_checks']))); return
    checks=json.loads((run.PUBLIC/'completion_checks.json').read_text()); rows=[]
    for ref in checks['groups']:
        assert run.artifact(ROOT/ref['path'])==ref; rows.append(json.loads((ROOT/ref['path']).read_text()))
    assert run.artifact(ROOT/checks['source_binding']['path'])==checks['source_binding']
    aggregate=json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    assert ci_by_assignment(rows,cfg)==aggregate['contrasts']
    for pair in cfg['pairs']:
        group=[r for r in rows if r['pair']==pair]
        for subset in ('all','envelope_positive'):
            expected={
                'inner_held_B':summarize_population([f['held_metrics'][subset] for r in group for f in r['folds']]),
                'original_fit_B':summarize_population([v[subset] for r in group for v in r['original_metrics']['B'].values()]),
                'original_held_C':summarize_population([v[subset] for r in group for v in r['original_metrics']['C'].values()])}
            assert expected==aggregate['summary'][pair][subset]
    previous=json.loads((run.previous.PUBLIC/'verification.json').read_text())
    tests=sorted(set(previous['test_files'])|{'tests/test_m3w_harm_tail_diagnostics.py','tests/test_m3w_harm_tail_crossfit.py','tests/test_m3w_harm_tail_reporting.py'})
    xml=run.PRIVATE/'tests.xml'
    p=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','--junitxml='+str(xml)],cwd=ROOT,capture_output=True,text=True)
    (run.PRIVATE/'tests.log').write_text(p.stdout+'\n'+p.stderr); print(p.stdout[-2500:],flush=True)
    assert p.returncode==0
    suites=list(ET.parse(xml).getroot()); count=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0))==0 for s in suites)
    required=['results.md','conclusions.md','failure_analysis.md','model_card.md','data_card.md','operation_zh.md','project_gap.md','harm_ranking.svg','training_loss.svg']
    assert all((run.PUBLIC/f).is_file() for f in required)
    artifacts={str(p.relative_to(run.PUBLIC)):run.digest(p) for p in run.PUBLIC.rglob('*') if p.is_file() and p.name!='verification.json'}
    assert all((run.PUBLIC/f).stat().st_size<1024**2 for f in artifacts)
    bindings=[*run.FILES,*tests,'scripts/evaluate_m3w_european_harm_tail_crossfit.py',
        'scripts/report_m3w_european_harm_tail_crossfit.py','scripts/plot_m3w_european_harm_tail_crossfit.py',str(Path(__file__).relative_to(ROOT))]
    run.immutable_json(run.PUBLIC/'verification.json',dict(all_passed=True,artifacts=artifacts,
        source_bindings={p:run.digest(ROOT/p) for p in bindings},tests=count,test_files=tests,full_legacy_suite='not_run',
        checkpoint_replays=len(r['checkpoints']),rank_tail_checks=r['independent_rank_tail_checks'],
        coordinate_reductions=r['independent_coordinate_reductions'],independent_confirmation=False,policy_changed=False))
    print(json.dumps(dict(all_passed=True,tests=count,test_files=len(tests),checkpoints=len(r['checkpoints']))))


if __name__=='__main__': main()
