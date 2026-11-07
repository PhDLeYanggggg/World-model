"""Three-seed missingness-aligned continuation with cached matched controls."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_easy_aligned_forecast as prior
from src.world_model.m3w_missingness_aligned_forecast import objective

np,torch,read,once,artifact,checked,sha,stable=(prior.np,prior.torch,prior.read,prior.once,
    prior.artifact,prior.checked,prior.sha,prior.stable)
PUBLIC=prior.bank.old.HOME/'moving_missingness_aligned_forecast_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_missingness_aligned_forecast_v1'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),ROOT/'src/world_model/m3w_missingness_aligned_forecast.py',PUBLIC/'protocol.md',
    ROOT/'tests/test_m3w_missingness_aligned_forecast.py')


def registration():
    names=('registration_v2.json','fit_population.json','training_freeze.json','forecast_freeze.json','analysis.json','replay_execution.json')
    p={k:prior.PUBLIC/k for k in names};parents=read(p['training_freeze.json']);receipt=read(p['replay_execution.json'])
    if (not parents['all_matched_draw_streams'] or parents['actual_updates']!=1800
            or receipt['analysis']['sha256']!=sha(p['analysis.json'])):
        raise ValueError('Verified, matched control forecasts required')
    jobs=read(p['registration_v2.json'])['jobs']
    once(REG,stable(dict(files=[artifact(x) for x in FILES],parents=[artifact(x) for x in
        (*p.values(),Path(prior.__file__),*prior.FILES[1:],prior.bank.REG,
         prior.bank.old.PARENTS[2],prior.bank.old.PARENTS[3],prior.bank.old.PARENTS[4])],
        jobs=jobs,objective='easy_aligned_plus_unknown_CV_radius',coefficient=1.,steps=300,
        seeds=prior.SEEDS,original_rules_unchanged=True,external_labels_opened=False)))
    return jobs,parents['fits']


def train(jobs,parents,beat):
    d=prior.fitting_only(jobs[0]['identity'],beat);results=[]
    for job in jobs:
        seed=job['seed'];i=job['identity'];warm=torch.load(checked(job['checkpoint']),map_location='cpu',weights_only=False)
        if warm['identity']!=i or warm['step']!=4000 or prior.source.array_hash(warm['factors'])!=i['factors_sha256']:
            raise ValueError('Changed source-only warm model or fitting weights')
        np.testing.assert_array_equal(warm['train_ids'],d['ids']);factor=warm['factors'][d['ids']]
        model=prior.source.SourceForecaster(i['architecture'],i['baseline_index']);model.load_state_dict(warm['model'])
        opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(warm['optimizer'])
        rng=torch.Generator();rng.set_state(warm['sampler_rng']);torch.set_rng_state(warm['torch_rng'])
        step=0;draws=np.zeros(len(factor),np.int64);trace=[]
        path=PRIVATE/f'fit_{seed}.pt';ident=dict(seed=seed,registration_sha256=sha(REG),parent=job['checkpoint'])
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['identity']!=ident:raise ValueError('Cannot resume different objective or source')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng.set_state(state['sampler'])
            torch.set_rng_state(state['torch_rng']);step,draws,trace=(state[k] for k in ('step','draws','trace'))
        model.train()
        while step<300:
            ix=prior.draw_batch(d['groups'],64,rng);inputs=prior.pack_geometry(d['geometry'][ix])
            for group in opt.param_groups:group['lr']=.0001
            opt.zero_grad(set_to_none=True)
            loss,p=objective(model(inputs),torch.from_numpy(d['target'][ix]),torch.from_numpy(d['valid'][ix]),
                torch.tensor(factor[ix],dtype=torch.float32),inputs['baseline'],d['easy_cut'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
            step+=1;np.add.at(draws,ix,1)
            if step==1 or step%50==0:
                trace.append(dict(step=step,loss=float(loss.detach()),**{k:float(v) for k,v in p.items()}))
                tmp=path.with_suffix('.tmp');torch.save(dict(identity=ident,model=model.state_dict(),optimizer=opt.state_dict(),
                    step=step,draws=draws,trace=trace,sampler=rng.get_state(),torch_rng=torch.get_rng_state()),tmp);tmp.replace(path)
                beat(state='training',seed=seed,**trace[-1])
        if step!=300:raise ValueError('Keep the matched 300-step budget')
        for t in parents:
            if t['identity']['seed']!=seed:continue
            old=torch.load(checked(t['checkpoint']),map_location='cpu',weights_only=False)
            if old['step']!=300 or old['identity']!=t['identity']:
                raise ValueError('Matched control training identity changed')
            prior.source.exact((draws,rng.get_state(),torch.get_rng_state()),(old['draws'],old['sampler'],old['torch_rng']))
        results.append(dict(seed=seed,identity=ident,checkpoint=artifact(path),steps=step,trace=trace,
            draws=int(draws.sum()),distinct_drawn_rows=int((draws>0).sum()),control_sampling_and_rng_exact=True))
    once(PUBLIC/'training_freeze.json',dict(fits=results,new_updates=900,cached_control_fits=6,before_A_labels=True))
    return dict(models=3,updates=900)


def infer(jobs,beat):
    a=prior.bank.old.prior.arrays(('global_ids','geometry','sites'));ids=a['global_ids']
    if len(ids)!=125540 or set(a['sites'])&set(jobs[0]['identity']['fit_groups']):
        raise ValueError('The full held source roster must stay outside fitting')
    refs=[]
    for fit in read(PUBLIC/'training_freeze.json')['fits']:
        seed=fit['seed']; i=next(x['identity'] for x in jobs if x['seed']==seed)
        state=torch.load(checked(fit['checkpoint']),map_location='cpu',weights_only=False)
        if state['step']!=300 or state['identity']!=fit['identity']:raise ValueError('Unfrozen model')
        model=prior.source.SourceForecaster(i['architecture'],i['baseline_index']);model.load_state_dict(state['model'])
        result=[]
        def forward(g):return prior.source.predict(model,{'geometry':g},np.arange(len(g)),128)
        for k,offset in enumerate(range(0,len(ids),4096)):
            take=slice(offset,offset+4096)
            x,_=prior.spool_batch(PRIVATE/'chunks'/str(seed),k,fit['checkpoint'],ids[take],a['geometry'][take],forward)
            result.append(x)
            if k%8==0:beat(state='inference',seed=seed,rows=min(offset+4096,len(ids)))
        x=np.concatenate(result); path=PRIVATE/f'{seed}.npz'
        if not path.exists():
            tmp=path.with_suffix('.pending.npz');np.savez(tmp,ids=ids,prediction=x);tmp.replace(path)
        else:
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['prediction'],x)
        refs.append(dict(seed=seed,forecast=artifact(path),checkpoint=fit['checkpoint'],rows=len(ids)))
    if len(refs)!=3:raise ValueError('Require every actual seed')
    once(PUBLIC/'forecast_freeze.json',dict(forecasts=refs,all_before_A_labels=True))
    return dict(models=3,rows=3*len(ids))


def score(beat):
    bank=prior.bank;old=bank.old;parent=bank.parent
    fresh=read(PUBLIC/'forecast_freeze.json'); controls=read(prior.PUBLIC/'forecast_freeze.json')
    if not fresh['all_before_A_labels'] or len(fresh['forecasts'])!=3 or len(controls['forecasts'])!=6:
        raise ValueError('Complete frozen paths and matched controls required')
    a=old.prior.arrays(('global_ids','history','origin','queries','sites','recordings'))
    y=old.prior.arrays(('target_eval','valid'));rows=[];records=[]
    cuts=read(old.PARENTS[4])['cuts'];cuts={s:cuts[str(r['job']['producer_fold'])] for r in read(old.PARENTS[3])['forecasts'] for s in r['job']['donor_sites']}
    for seed in prior.SEEDS:
        pred={}
        for ref in (*fresh['forecasts'],*controls['forecasts']):
            if ref['seed']!=seed:continue
            with np.load(checked(ref['forecast']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],a['global_ids']);pred[ref.get('arm','missingness_aligned')]=z['prediction'].astype(float)
        prior_paths=parent.read_banks(read(parent.REG)['forecasts'],seed,a['global_ids'],a['sites'])
        for s in np.unique(a['sites']):
            ix=np.flatnonzero(a['sites']==s);h=a['history'][ix];o=a['origin'][ix,None];cut=cuts[s]
            cv=np.arange(1,13)[None,:,None]*(h[:,-1]-h[:,-2])[:,None];truth=y['target_eval'][ix]-o;mask=y['valid'][ix]
            _,inv,c=np.unique(a['queries'][ix],return_inverse=True,return_counts=True);w=1./c[inv]
            p=pred['missingness_aligned'][ix]
            stats={k:old.prior.base.statistics(pred[k][ix],p,cv,truth,mask,w/w.sum(),cut['easy'],cut['hard'])[-1] for k in prior.ARMS}
            radius={k:parent.readout.comparison_masses(cv,z[ix],cv,truth,mask,w) for k,z in pred.items()}
            norm={k:100*v['unknown_radius']/v['reference'] for k,v in radius.items()}
            lower={k:100*(v['known_delta']-v['unknown_radius'])/v['reference'] for k,v in radius.items()}
            rows.append(dict(seed=seed,source=str(s),comparisons=stats,unknown_radius_to_CV_pp=norm,completion_lower_vs_CV_pp=lower))
            for rec in np.unique(a['recordings'][ix]):
                at=a['recordings'][ix]==rec;take=ix[at]
                paths=[prior_paths[k][take] for k in ('eqmotion','pct')]+[parent.baseline_numpy(h[at],k)-o[at]
                    for k in range(len(parent.BASELINES)) if k!=1]
                oracle=bank.api.solve(cv[at],np.stack(paths+[p[at]]),truth[at],mask[at],w[at],str(rec),cut['easy'],cut['hard'])
                original=parent.readout.policy(p[at],cv[at],cv[at],truth[at],mask[at],np.full(len(take),str(rec)),
                    easy_cut=cut['easy'],hard_cut=cut['hard'])['recordings'][str(rec)]
                records.append(dict(seed=seed,source=str(s),recording=str(rec),candidate_bank_ceiling=oracle,raw_path_original_risk=original))
        beat(state='scored_seed',seed=seed,records=len(records))
    if len(rows)!=42 or len(records)!=102:raise ValueError('No recording or source omitted')
    names=np.unique(a['sites']);draw=np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    def interval(fn):
        v=[[fn(r) for r in rows if r['source']==s] for s in names]
        if any(x is None for l in v for x in l):return None
        x=np.asarray(v).mean(1);return dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    contrasts={}
    for k in prior.ARMS:
        def arm_change(r,field):
            t=r['comparisons'][k]; u,v=(t[j][field] for j in ('control','treatment'))
            return None if u is None or v is None else v-u
        contrasts[k]=dict(known_pp=interval(lambda r:r['comparisons'][k]['paired_known_pp']),
            lower_pp=interval(lambda r:r['comparisons'][k]['paired_completion_lower_pp']),
            easy_harm_increase_pp=interval(lambda r:arm_change(r,'known_easy_positive_harm_percent')),
            hard_gain_pp=interval(lambda r:arm_change(r,'known_hard_gain_percent')),
            unknown_radius_to_CV_change_pp=interval(lambda r:r['unknown_radius_to_CV_pp']['missingness_aligned']-r['unknown_radius_to_CV_pp'][k]))
    result=dict(contrasts=contrasts,source_views=rows,recording_views=records,
        completion_lower_vs_CV=interval(lambda r:r['completion_lower_vs_CV_pp']['missingness_aligned']),
        new_bank_feasible_views=sum(r['candidate_bank_ceiling']['feasible'] for r in records),
        original_raw_risk_passes=sum(r['raw_path_original_risk']['supported'] for r in records),
        original_rules_unchanged=True,independent_confirmation=False,external_outcomes_opened=False,
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'analysis.json',stable(result));return dict(analysis=artifact(PUBLIC/'analysis.json'),contrasts=contrasts,
        feasible=result['new_bank_feasible_views'],raw_passes=result['original_raw_risk_passes'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','train','infer','score','replay'))
    phase=p.parse_args().phase;jobs,parents=registration()
    if phase=='register':return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit fixed training definition before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3 or time.monotonic()-start>43200:
                raise OSError('Keep the existing checkpoints and resume within resource limits')
            v=dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            prior.source.atomic(PRIVATE/'heartbeat.json',v);print(v,flush=True)
        beat(state='started')
        if phase=='train':result=train(jobs,parents,beat)
        elif phase=='infer':result=infer(jobs,beat)
        else:result=score(beat)
        once(PUBLIC/f'{phase}_execution.json',stable(dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**result)))


if __name__=='__main__':main()
