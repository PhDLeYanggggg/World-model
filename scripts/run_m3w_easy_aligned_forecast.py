"""Fitting-only warm PCT continuation with a matched objective ablation."""
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
from scripts import run_m3w_moving_source_forecast as source
from scripts import run_m3w_recording_bank_ceiling as bank
from src.data_unification import m3w_moving_forecast_data as prep
from src.data_unification.m3w_causal_forecast_stream import spool_batch
from src.world_model.m3w_easy_aligned_forecast import objective
from src.world_model.m3w_native_forecast import draw_batch,pack_geometry

np,torch,read,sha,once,artifact,checked,stable=(source.np,source.torch,bank.read,source.sha,source.once,
    bank.artifact,bank.parent.checked,source.stable)
PUBLIC=bank.old.HOME/'moving_easy_aligned_forecast_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_easy_aligned_forecast_v1'
REG=PUBLIC/'registration_v2.json';STEPS=300;SEEDS=(17,29,43);ARMS=('ade','easy_aligned')
FILES=(Path(__file__),ROOT/'src/world_model/m3w_easy_aligned_forecast.py',PUBLIC/'protocol.md',
       ROOT/'tests/test_m3w_easy_aligned_forecast.py',PUBLIC/'zero_support_repair.md')


def registration():
    refs=read(bank.old.PARENTS[-2])['prediction_receipts']; checks=read(bank.old.PARENTS[-1])['trials']
    jobs=[dict(t,identity=next(r['identity'] for r in refs if r['sha256']==t['prediction_sha256']))
          for t in checks if t['fold']==2 and t['family']=='pct']
    if tuple(j['seed'] for j in jobs)!=SEEDS or len({tuple(j['identity']['fit_groups']) for j in jobs})!=1:
        raise ValueError('Require the same fixed training roster and three actual seeds')
    parents=(bank.old.PARENTS[-2],bank.old.PARENTS[-1],bank.PUBLIC/'analysis.json',bank.REG,
        Path(prep.__file__),ROOT/'src/world_model/m3w_native_forecast.py',ROOT/'configs/m3w_moving_source_forecast_v1.json',
        bank.old.HOME/'moving_development_admission_v1/admission.json',
        bank.old.HOME/'movingdronecrowd_intake_v1/analysis.json',PUBLIC/'registration.json')
    once(REG,stable(dict(files=[artifact(p) for p in FILES],parents=[artifact(p) for p in parents],
        jobs=jobs,arms=ARMS,steps=STEPS,lr=.0001,batch_size=64,lambda_easy=2.,original_2_percent_unchanged=True,
        external_outcomes=False,stage5c_executed=False,smc_enabled=False)))
    return jobs


def fitting_only(identity,beat):
    cfg=read(source.CONFIG); admission=prep.load_admission(ROOT,ROOT/cfg['admission'])
    structure=read(bank.old.HOME/'movingdronecrowd_intake_v1/analysis.json'); lookup={v['clip']:v for v in structure['recordings']}
    declared=set(identity['fit_groups']);fit_groups=set(identity['normalizers']); cursor=0; chunks=[]
    empty=set(read(bank.old.PARENTS[-2])['structurally_empty_groups'])&declared
    if fit_groups!=declared-empty:raise ValueError('Only the originally declared empty group can lack fitting rows')
    for node,v in sorted(admission['recordings'].items()):
        clip=lookup[v['clip']];n=clip['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        start=cursor;cursor+=n
        if v['exclusion_group'] in empty and n!=0:raise ValueError('Original empty group acquired rows')
        if v['exclusion_group'] not in fit_groups or n==0:continue
        path=ROOT/v['source_annotation_path'];prep.require_fit_record(admission,node,'fit',path)
        r=prep.mot_record(path.read_bytes(),v['clip'],range(1,clip['image_frames']+1));packet=[]
        for q in range(7,r.valid.shape[1]-12):
            past=prep.past_inputs(r,q,prep.SPEC)
            g,ix,_=prep.encode_scene(past['positions'],past['history_mask'],past['agent_ids'])
            if not len(ix):continue
            y=prep.target_labels(r,past,prep.SPEC);mask=y['future_mask'][ix]
            relative=y['future_positions'][ix]-past['positions'][ix,-1,None]
            packet.append((g,np.where(mask[...,None],relative,0).astype(np.float32),mask))
        g,y,m=(np.concatenate([p[i] for p in packet]) for i in range(3))
        if len(g)!=n:raise ValueError('Original fit row IDs changed')
        chunks.append((g,y,m,np.arange(start,cursor),np.full(n,v['exclusion_group']),node))
        beat(state='packed_original_fit_record',record=node,fit_records=len(chunks))
    data={k:np.concatenate([p[i] for p in chunks]) for i,k in enumerate(('geometry','target','valid','ids','sites'))}
    data['recordings']=[p[-1] for p in chunks]
    if source.array_hash(data['ids'])!=identity['train_ids_sha256'] or set(data['sites'])!=fit_groups:
        raise ValueError('Warm training must exactly reproduce original global fit IDs')
    data['groups']=[np.flatnonzero(data['sites']==s) for s in sorted(fit_groups)]
    from src.evaluation.m3w_native_metrics import native_errors
    cv,_=native_errors(data['geometry'][:,332:356].reshape(-1,12,2),data['target'],data['valid'],np.ones(len(data['ids'])))
    data['easy_cut']=float(np.quantile(cv[np.isfinite(cv)&(cv>0)],.25))
    once(PUBLIC/'fit_population.json',dict(rows=len(cv),groups=sorted(fit_groups),declared_groups=sorted(declared),
        zero_support_groups=sorted(empty),recordings=data['recordings'],
        global_fit_ids_sha256=source.array_hash(data['ids']),geometry_sha256=source.array_hash(data['geometry']),
        target_sha256=source.array_hash(data['target']),valid_sha256=source.array_hash(data['valid']),
        fitting_easy_cut=data['easy_cut'],A_target_opened=False))
    return data


def train(jobs,beat,prefix):
    data=fitting_only(jobs[0]['identity'],beat); results=[]
    for j in jobs:
        warm=torch.load(checked(j['checkpoint']),map_location='cpu',weights_only=False);i=j['identity']
        if warm['step']!=4000 or warm['identity']!=i or source.array_hash(warm['factors'])!=i['factors_sha256']:
            raise ValueError('Use the complete original optimizer and fixed fitting normalizers')
        np.testing.assert_array_equal(warm['train_ids'],data['ids']); factor=warm['factors'][data['ids']]
        matched=[]
        for arm in ARMS:
            key=f'{j["seed"]}_{arm}';path=PRIVATE/'fits'/f'{key}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            ident=dict(registration_sha256=sha(REG),parent=j['checkpoint'],arm=arm,seed=j['seed'])
            model=source.SourceForecaster(i['architecture'],i['baseline_index']);model.load_state_dict(warm['model'])
            opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(warm['optimizer'])
            rng=torch.Generator();rng.set_state(warm['sampler_rng']);torch.set_rng_state(warm['torch_rng'])
            draws=np.zeros(len(factor),np.int64);step=0;trace=[]
            if path.exists():
                state=torch.load(path,map_location='cpu',weights_only=False)
                if state['identity']!=ident:raise ValueError('Changed warm continuation identity')
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng.set_state(state['sampler'])
                torch.set_rng_state(state['torch_rng']);step,draws,trace=(state[k] for k in ('step','draws','trace'))
            limit=20 if prefix else STEPS
            if step>limit:raise ValueError('Cannot restart a smaller completed budget')
            model.train()
            while step<limit:
                ix=draw_batch(data['groups'],64,rng);inputs=pack_geometry(data['geometry'][ix]);opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=.0001
                loss,ade,h=objective(model(inputs),torch.from_numpy(data['target'][ix]),torch.from_numpy(data['valid'][ix]),
                    torch.tensor(factor[ix],dtype=torch.float32),inputs['baseline'],data['easy_cut'],aligned=arm=='easy_aligned')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
                step+=1;np.add.at(draws,ix,1)
                if step==1 or step%50==0 or step==limit:
                    trace.append(dict(step=step,loss=float(loss.detach()),ade=float(ade),easy_harm_penalty=float(h)))
                    beat(state='training',key=key,**trace[-1]);tmp=path.with_suffix('.tmp')
                    torch.save(dict(identity=ident,model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                        sampler=rng.get_state(),torch_rng=torch.get_rng_state(),draws=draws,trace=trace),tmp);tmp.replace(path)
            if prefix:return dict(prefix_checkpoint=artifact(path),step=step,key=key)
            matched.append((draws.copy(),rng.get_state().clone(),torch.get_rng_state().clone()))
            results.append(dict(identity=ident,checkpoint=artifact(path),step=step,trace=trace,
                draws=int(draws.sum()),distinct_drawn_rows=int((draws>0).sum())))
        source.exact(matched[0],matched[1])
    once(PUBLIC/'training_freeze.json',dict(fits=results,actual_updates=6*STEPS,all_matched_draw_streams=True,
        before_A_inference_and_labels=True))
    return dict(models=6,updates=6*STEPS)


def infer(jobs,beat):
    a=bank.old.prior.arrays(('global_ids','geometry','sites'));ids=a['global_ids']
    if len(ids)!=125540 or set(a['sites'])&set(jobs[0]['identity']['fit_groups']):
        raise ValueError('Keep whole source-disjoint A bank')
    freeze=read(PUBLIC/'training_freeze.json');results=[]
    for fit in freeze['fits']:
        seed,arm=(fit['identity'][k] for k in ('seed','arm'));key=f'{seed}_{arm}'
        j=next(t for t in jobs if t['seed']==seed);i=j['identity']
        model=source.SourceForecaster(i['architecture'],i['baseline_index'])
        state=torch.load(checked(fit['checkpoint']),map_location='cpu',weights_only=False)
        if state['step']!=STEPS or state['identity']!=fit['identity']:raise ValueError('Complete frozen fit required')
        model.load_state_dict(state['model']);pieces=[]; cached=0
        def forward(g):return source.predict(model,{'geometry':g},np.arange(len(g)),128)
        for k,start in enumerate(range(0,len(ids),4096)):
            ix=slice(start,start+4096)
            p,reused=spool_batch(PRIVATE/'chunks'/key,k,fit['checkpoint'],ids[ix],a['geometry'][ix],forward)
            pieces.append(p);cached+=int(reused)*len(p)
            if k%8==0:beat(state='inference',key=key,completed_rows=min(start+4096,len(ids)))
        forecast=np.concatenate(pieces);dest=PRIVATE/f'{key}.npz'
        if not dest.exists():
            tmp=dest.with_suffix('.pending.npz');np.savez(tmp,ids=ids,prediction=forecast);tmp.replace(dest)
        else:
            with np.load(dest,allow_pickle=False) as p:
                np.testing.assert_array_equal(p['ids'],ids);np.testing.assert_array_equal(p['prediction'],forecast)
        results.append(dict(key=key,seed=seed,arm=arm,forecast=artifact(dest),rows=len(forecast),checkpoint=fit['checkpoint']))
        beat(state='forecast_frozen',key=key,complete=len(results),reused_rows=cached)
    if len(results)!=6:raise ValueError('No omitted fitted arm')
    once(PUBLIC/'forecast_freeze.json',dict(forecasts=results,all_before_A_labels=True,rows=6*len(ids)))
    return dict(models=6,rows=6*len(ids))


def score(beat):
    parent=bank.parent;f=read(PUBLIC/'forecast_freeze.json')
    if len(f['forecasts'])!=6 or not f['all_before_A_labels']:raise ValueError('Freeze both entire arms first')
    a=bank.old.prior.arrays(('global_ids','history','origin','queries','sites','recordings'))
    y=bank.old.prior.arrays(('target_eval','valid'));rows=[];ceiling=[]
    cuts=read(bank.old.PARENTS[4])['cuts'];cuts={s:cuts[str(r['job']['producer_fold'])] for r in read(bank.old.PARENTS[3])['forecasts'] for s in r['job']['donor_sites']}
    for seed in SEEDS:
        pred={}
        for r in f['forecasts']:
            if r['seed']==seed:
                with np.load(checked(r['forecast']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],a['global_ids']);pred[r['arm']]=z['prediction'].astype(float)
        old=parent.read_banks(read(parent.REG)['forecasts'],seed,a['global_ids'],a['sites'])
        for s in np.unique(a['sites']):
            ix=np.flatnonzero(a['sites']==s);h=a['history'][ix];o=a['origin'][ix,None];cut=cuts[s]
            cv=np.arange(1,13)[None,:,None]*(h[:,-1]-h[:,-2])[:,None];truth=y['target_eval'][ix]-o;mask=y['valid'][ix]
            _,inv,c=np.unique(a['queries'][ix],return_inverse=True,return_counts=True);w=1./c[inv]
            primary=bank.old.prior.base.statistics(*(pred[k][ix] for k in ARMS),cv,truth,mask,w/w.sum(),cut['easy'],cut['hard'])[-1]
            secondary=bank.old.prior.base.statistics(old['pct'][ix],pred['easy_aligned'][ix],cv,truth,mask,w/w.sum(),cut['easy'],cut['hard'])[-1]
            rows.append(dict(seed=seed,source=str(s),matched=primary,vs_original_pct=secondary))
            for rec in np.unique(a['recordings'][ix]):
                at=a['recordings'][ix]==rec;take=ix[at]
                paths=[old[k][take] for k in ('eqmotion','pct')]+[parent.baseline_numpy(h[at],i)-o[at]
                    for i in range(len(parent.BASELINES)) if i!=1]
                feasible={k:bank.api.solve(cv[at],np.stack(paths+[pred[k][take]]),truth[at],mask[at],w[at],
                    str(rec),cut['easy'],cut['hard']) for k in ARMS}
                ceiling.append(dict(seed=seed,source=str(s),recording=str(rec),banks=feasible))
        beat(state='full_seed_scored',seed=seed,source_views=len(rows),recording_views=len(ceiling))
    if len(rows)!=42 or len(ceiling)!=102:raise ValueError('Retain the full held development roster')
    names=np.unique(a['sites']);draw=np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    def interval(fn):
        v=[[fn(r['matched']) for r in rows if r['source']==s] for s in names]
        if any(x is None for line in v for x in line):return None
        x=np.asarray(v).mean(1);return dict(point=float(x.mean()),interval95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    easy='known_easy_positive_harm_percent'; hard='known_hard_gain_percent'
    def contrast(key):
        return interval(lambda r: r['treatment'][key]-r['control'][key]
            if r['treatment'][key] is not None and r['control'][key] is not None else None)
    result=dict(sources=rows,recording_ceiling=ceiling,
        paired=dict(known_pp=interval(lambda r:r['paired_known_pp']),lower_pp=interval(lambda r:r['paired_completion_lower_pp']),
            easy_harm_increase_pp=contrast(easy),hard_gain_pp=contrast(hard)),
        new_bank_feasible_views={k:sum(r['banks'][k]['feasible'] for r in ceiling) for k in ARMS},
        scope='exposed_A_development_three_real_seeds_matched_continuation',
        external_labels_opened=False,independent_confirmation=False,deployment_changed=False,submission_ready=False,
        stage5c_executed=False,smc_enabled=False)
    once(PUBLIC/'analysis.json',stable(result));return dict(analysis=artifact(PUBLIC/'analysis.json'),paired=result['paired'],
        bank=result['new_bank_feasible_views'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','prefix','train','infer','score','replay'))
    phase=p.parse_args().phase;jobs=registration()
    if phase=='register':return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit the prospective model ablation before fitting')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
        def beat(**v):
            if shutil.disk_usage(ROOT).free<2*1024**3:raise OSError('Keep 2 GiB free; checkpoints remain resumable')
            if time.monotonic()-start>43200:raise TimeoutError('Resume from the last coherent checkpoint')
            v=dict(pid=os.getpid(),phase=phase,seconds=time.monotonic()-start,**v)
            source.atomic(PRIVATE/'heartbeat.json',v);print(v,flush=True)
        beat(state='started')
        if phase in ('prefix','train'): result=train(jobs,beat,phase=='prefix')
        elif phase=='infer':result=infer(jobs,beat)
        else:result=score(beat)
        once(PUBLIC/f'{phase}_execution.json',stable(dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**result)))


if __name__=='__main__':main()
