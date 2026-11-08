"""Test causal anchor denoising as a full-roster forecast, not a selector."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_ols_crossfit_calibration as old
from src.world_model import m3w_ols_anchor_denoising as model

base,run,prev,np,torch = old.base,old.run,old.prev,old.np,old.torch
PUBLIC = old.PUBLIC.parent/'native_ols_anchor_denoising_v1'
PRIVATE = old.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_anchor_denoising.py')
CFG = old.old.CFG
ARMS = (*model.ARMS,'matched_anchor','matched_slope')
CONTRASTS = tuple((a,b) for a in ARMS if a not in ('cv','ols4') for b in ('cv','ols4'))+(
    ('ols4','cv'),('learned_anchor','learned_slope'),('learned_anchor','fixed_anchor'),('matched_anchor','matched_slope'))


def register():
    cut,jobs = old.register()
    prior = PUBLIC.parent/'native_ols_bounded_gain_v1'
    prev.receipt(prior,'replay')
    paths = (*FILES,old.REG,Path(old.__file__),prior/'replay_execution.json',base.fit_source.PUBLIC/'fit_population.json')
    run.once(REG,run.stable(dict(bindings=[run.artifact(f) for f in paths],arms=ARMS,seed=prev.SEEDS,
        settings=CFG,contrasts=CONTRASTS,primary='learned_anchor_vs_ols4',primary_mechanism='matched_anchor_vs_matched_slope',
        secondary_reference='cv',whole_roster=True,fit_role='leave_one_of_nine',draws=3000,
        external_access=False,development_targets=False,original_cv_rule_unchanged=True)))
    return cut,jobs


def data(jobs,beat):
    d = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    if len(d['ids']) != 69746 or len(set(d['sites'])) != 9 or set(d['sites']) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Expected fixed fitting cohort excluding all development sources')
    return d


def fit(x,ref,delta,target,valid,scale,train,groups,ident,beat,resume):
    seed = ident['seed']; torch.manual_seed(seed)
    net = model.network(x.shape[-1]); opt = torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    draws,trace,step = np.zeros(len(train),np.int64),[],0
    path = PRIVATE/f"{ident['held']}_{seed}_{ident['arm']}.pt"
    if path.exists():
        if not resume:
            raise FileExistsError('Existing fit requires --resume')
        st = torch.load(path,map_location='cpu',weights_only=False)
        if st['identity'] != ident or not 0 < st['step'] <= CFG['steps']:
            raise ValueError('Fold checkpoint has incompatible identity')
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); rng.set_state(st['rng'])
        draws,trace,step = (st[k] for k in ('draws','trace','step'))
    while step < CFG['steps']:
        ix = base.native.draw_batch(groups,CFG['batch'],rng); at = train[ix]
        cost = model.loss(net,x[at],ref[at],delta[at],target[at],valid[at],scale[at])
        if not torch.isfinite(cost):
            raise FloatingPointError('Nonfinite forecast loss')
        opt.zero_grad(set_to_none=True); cost.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step(); step+=1; np.add.at(draws,ix,1)
        if step % CFG['checkpoint'] == 0:
            trace.append(dict(step=step,loss=float(cost.detach())))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),
                            draws=draws,trace=trace,step=step),tmp)
            tmp.replace(path)
            beat(state='fit_checkpoint',held=ident['held'],seed=seed,arm=ident['arm'],**trace[-1])
    if draws.sum() != CFG['steps']*CFG['batch']:
        raise ValueError('Training budget changed')
    return net.eval(),run.artifact(path),(draws,rng.get_state())


def train(jobs,beat,resume):
    d = data(jobs,beat)
    x,p,a,b,scale,_ = model.design(d['geometry'][:,:16].reshape(-1,8,2))
    np.testing.assert_allclose(p['cv'],d['geometry'][:,332:356].reshape(-1,12,2),atol=.0001)
    x,ref,target,valid,scale = (torch.from_numpy(z) for z in (x,p['ols4'].astype(np.float32),d['target'],d['valid'],scale.astype(np.float32)))
    deltas = [torch.from_numpy(z.astype(np.float32)) for z in (a,b)]
    factors = np.full((len(prev.SEEDS),len(model.FITS),len(x)),np.nan,dtype=np.float32)
    sites = sorted(set(d['sites'])); fits = []
    for k,seed in enumerate(prev.SEEDS):
        for site in sites:
            train = np.flatnonzero(d['sites'] != site); held = np.flatnonzero(d['sites'] == site)
            groups = [np.flatnonzero(d['sites'][train] == s) for s in sites if s != site]
            usage = []
            for j,arm in enumerate(model.FITS):
                ident = dict(seed=seed,arm=arm,held=str(site),registration=run.sha(REG),
                             train_ids=base.fit_source.source.array_hash(d['ids'][train]))
                net,ck,state = fit(x,ref,deltas[j],target,valid,scale,train,groups,ident,beat,resume)
                usage.append(state)
                with torch.inference_mode():
                    factors[k,j,held] = net(x[held]).sigmoid().flatten().numpy()
                fits.append(dict(identity=ident,checkpoint=ck,held_rows=len(held)))
            base.fit_source.source.exact(*usage)
    if len(fits) != 54 or not np.isfinite(factors).all() or (factors < 0.).any() or (factors > 1.).any():
        raise ValueError('Incomplete held-source forecast freeze')
    path = PRIVATE/'coefficients.npz'; tmp = path.with_suffix('.pending.npz')
    np.savez(tmp,ids=d['ids'],factors=factors); tmp.replace(path)
    out = PRIVATE/'freeze.json'
    run.once(out,run.stable(dict(fits=fits,coefficients=run.artifact(path),registration=run.sha(REG),
                                 source_disjoint=True,matched_draws=True,past_only_predictions=True)))
    return out


def score(jobs,frozen,cut,beat):
    if frozen['registration'] != run.sha(REG) or not all(frozen[k] for k in ('source_disjoint','matched_draws','past_only_predictions')):
        raise ValueError('Past-only held-source freeze is required before scoring')
    d = data(jobs,beat); groups = old.fitting_layout(jobs[0]['identity'],d)
    _,base_paths,a,b,_,available = model.design(d['geometry'][:,:16].reshape(-1,8,2))
    truth,valid = d['target'],d['valid']; known = valid.any(1); endpoint = valid[:,-1]
    with np.load(run.checked(frozen['coefficients']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); f = z['factors']
    src = np.unique(d['sites'],return_inverse=True)[1]; w = np.empty(len(src))
    for q in groups:
        w[q] = 1./len(q)
    t = np.where(valid[...,None],truth,0.)

    def errors(p):
        e = np.linalg.norm(p-t,axis=2)
        return np.where(valid,e,0.).sum(1)/valid.sum(1).clip(1),e[:,-1]

    cv,_ = errors(base_paths['cv'])
    denom = np.bincount(src,weights=w*cv)
    if (denom <= 0.).any() or len(denom) != 9:
        raise ValueError('Incomplete CV reference support')
    weight = 100*w/denom[src]
    easy = known & (cv > 0.) & (cv <= cut['easy'])
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))

    def mean(v,mask):
        den = np.bincount(src,weights=w*mask,minlength=9)
        if (den == 0.).any():
            raise ValueError('The fixed slice lacks a fitting source denominator')
        return np.bincount(src,weights=w*np.where(mask,v,0.),minlength=9)/den

    contrasts = {f'{x}_vs_{y}':[] for x,y in CONTRASTS}; absolute = {a:[] for a in ARMS}
    for k,seed in enumerate(prev.SEEDS):
        anchor,slope = f[k,0,:,None,None],f[k,1,:,None,None]
        ref = base_paths['ols4']; joint = available & (np.linalg.norm(a,axis=2).max(1) > 1e-6)
        paths = dict(**base_paths,learned_anchor=ref+anchor*a,learned_slope=ref+slope*b,
                     matched_anchor=ref+anchor*a*joint[:,None,None],matched_slope=ref+anchor*b*joint[:,None,None])
        x = {a:errors(p) for a,p in paths.items()}
        for name,(ade,fde) in x.items():
            absolute[name].append(dict(ade=mean(ade,known),fde=mean(fde,endpoint),easy_ade=mean(ade,easy),
                max_source_p95=max(float(np.quantile(ade[(src==s)&known],.95)) for s in range(9))))
        for name,reference in CONTRASTS:
            g = x[reference][0]-x[name][0]
            debit = np.linalg.norm(paths[reference]-paths[name],axis=2).max(1)
            contrasts[f'{name}_vs_{reference}'].append(dict(known=np.bincount(src,weights=weight*np.where(known,g,0.)),
                lower=np.bincount(src,weights=weight*np.where(known,g,-debit)),
                harm=np.bincount(src,weights=weight*np.where(known,np.maximum(-g,0.),debit))))
        beat(state='whole_fitting_roster_scored',seed=seed,rows=len(src))
    interval = lambda cells:old.metric.intervals(np.stack(cells,1),draws)
    return dict(contrasts={a:{k:interval([r[k] for r in rows]) for k in ('known','lower','harm')} for a,rows in contrasts.items()},
        absolute={a:dict(**{k:interval([r[k] for r in rows]) for k in ('ade','fde','easy_ade')},
                         worst_source_p95=max(r['max_source_p95'] for r in rows)) for a,rows in absolute.items()},
        rows=len(src),queries=len(groups),fits=len(frozen['fits']),known_rows=int(known.sum()),fde_rows=int(endpoint.sum()),
        matched_direction_rows=int(joint.sum()),seeds=prev.SEEDS,source_groups=9,whole_roster=True,
        development_targets=False,external_access=False,independent_confirmation=False,
        original_cv_rule_unchanged=True,deployment=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','train','score','replay'))
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    cut,jobs = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit before training or scoring')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG['torch_threads']); torch.set_num_interop_threads(1)

        def beat(**v):
            event = dict(pid=os.getpid(),phase=a.phase,seconds=time.monotonic()-start,**v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json',event)
            print(event,flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Stopped with atomic checkpoints preserved')

        mark = PUBLIC/f'{a.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC,a.phase)
            return
        if a.phase == 'train':
            out = train(jobs,beat,a.resume)
        else:
            if a.phase == 'replay':
                prev.receipt(PUBLIC,'score')
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(score(jobs,prev.receipt(PUBLIC,'train'),cut,beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
