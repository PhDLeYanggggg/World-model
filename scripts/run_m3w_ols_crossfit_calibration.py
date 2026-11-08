"""Calibrate an OLS4 intervention score on held fitting sources, never external data."""
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
from scripts import run_m3w_ols_support_selection as old
from src.evaluation import m3w_ols_crossfit_calibration as metric

model,base,run,prev,np,torch = old.model,old.base,old.run,old.prev,old.np,old.torch
PUBLIC = old.PUBLIC.parent/'native_ols_crossfit_calibration_v1'
PRIVATE = old.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(metric.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_crossfit_calibration.py')
FIT = ('direct','factorized_neighbor')
ARMS = (*FIT,'factorized_constant')
BASE,OLS = old.BASE,old.OLS
PAIRS = {f'{a}_calibrated_vs_{b}':(b,f'{a}_calibrated',BASE) for a in ARMS for b in (OLS,BASE)}
PAIRS.update({f'{a}_calibrated_vs_raw':(f'{a}_raw',f'{a}_calibrated',BASE) for a in ARMS})
MIN_SOURCES = 7


def register():
    cut,jobs = old.register()
    prev.receipt(old.PUBLIC,'replay')
    paths = (*FILES,old.REG,Path(model.__file__),Path(old.__file__),old.PUBLIC/'train_execution.json',
             old.PUBLIC/'infer_execution.json',old.PUBLIC/'replay_execution.json')
    run.once(REG,run.stable(dict(bindings=[run.artifact(p) for p in paths],arms=ARMS,settings=old.CFG,
        seeds=prev.SEEDS,thresholds=metric.THRESHOLDS,min_positive_sources=MIN_SOURCES,fit_role='leave_one_of_nine',
        source_bootstrap=3000,original_cv_rule_unchanged=True,external_access=False,
        dev_only_after_oof_selection=True,skip_dev_if_all_abstain=True)))
    return cut,jobs


def fitting_layout(identity,d):
    fit = base.fit_source
    adm = fit.prep.load_admission(ROOT,ROOT/fit.read(fit.source.CONFIG)['admission'])
    home = fit.bank.old.HOME
    lookup = {c['clip']:c for c in fit.read(home/'movingdronecrowd_intake_v1/analysis.json')['recordings']}
    groups,ids = [],[]
    offset,cursor = 0,0
    for node,v in sorted(adm['recordings'].items()):
        clip = lookup[v['clip']]
        n = clip['window_support']['obs8_pred12_native_stride1']['history_complete_agent_instances']
        start = offset; offset += n
        if v['exclusion_group'] not in identity['normalizers'] or not n:
            continue
        path = ROOT/v['source_annotation_path']; fit.prep.require_fit_record(adm,node,'fit',path)
        rec = fit.prep.mot_record(path.read_bytes(),v['clip'],range(1,clip['image_frames']+1))
        initial = cursor
        for q in range(7,rec.valid.shape[1]-12):
            past = fit.prep.past_inputs(rec,q,fit.prep.SPEC)
            count = int(past['history_mask'].all(1).sum())
            if count:
                groups.append(np.arange(cursor,cursor+count))
                cursor += count
        if cursor-initial != n or not (d['sites'][initial:cursor] == v['exclusion_group']).all():
            raise ValueError('Fitting query roster differs from authorized source rows')
        ids.extend(range(start,start+n))
    np.testing.assert_array_equal(ids,d['ids'])
    np.testing.assert_array_equal(np.concatenate(groups),np.arange(len(ids)))
    return groups


def pack(d):
    g = d['geometry']; h = g[:,:16].reshape(-1,8,2)
    n = g[:,38:166].reshape(-1,8,8,2); mask = g[:,230:294].reshape(-1,8,8).all(2)
    return model.design(h,n,mask)


def fit_fold(x,near,truth,missing,groups,train,ident,path,beat,resume):
    seed,arm = ident['seed'],ident['arm']
    torch.manual_seed(seed)
    net = model.GainSupport(x.shape[-1]); opt = torch.optim.AdamW(net.parameters(),lr=old.CFG['lr'],weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    step,trace,draws = 0,[],np.zeros(len(train),np.int64)
    if path.exists():
        if not resume:
            raise FileExistsError('Existing fold requires --resume')
        st = torch.load(path,map_location='cpu',weights_only=False)
        if st['identity'] != ident or not 0 < st['step'] <= old.CFG['steps']:
            raise ValueError('Fold checkpoint identity changed')
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); rng.set_state(st['rng'])
        step,trace,draws = (st[k] for k in ('step','trace','draws'))
    while step < old.CFG['steps']:
        ix = base.native.draw_batch(groups,old.CFG['batch'],rng)
        at = train[ix]
        out,logit = net(x[at],near[at],ego_support=False)
        loss,_,_ = model.objective(out,logit,truth[at],missing[at],factorized=arm != 'direct')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite held-source training')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step(); step+=1; np.add.at(draws,ix,1)
        if step % old.CFG['checkpoint'] == 0:
            trace.append(dict(step=step,loss=float(loss.detach())))
            temp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),
                            draws=draws,trace=trace,step=step),temp)
            temp.replace(path)
            beat(state='fold_checkpoint',held=ident['held'],seed=seed,arm=arm,**trace[-1])
    if draws.sum() != old.CFG['steps']*old.CFG['batch']:
        raise ValueError('Fold fitting budget changed')
    return net.eval(),draws,rng.get_state()


def predictions(net,x,n,debit,arm,constant=None):
    with torch.inference_mode():
        out = [net(xx,nn,ego_support=False) for xx,nn in zip(x.split(1024),n.split(1024),strict=True)]
        cost = torch.cat([z[0] for z in out]).numpy()
        p = torch.cat([z[1] for z in out]).sigmoid().numpy()
    result = {arm:model.risk_score(cost,p,debit,factorized=arm != 'direct')}
    if arm != 'direct':
        result['factorized_constant'] = model.risk_score(cost,np.full(len(p),constant),debit,factorized=True)
    return result


def oof(jobs,beat,resume):
    d = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    sites = sorted(set(d['sites']))
    if len(sites) != 9 or set(sites) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Fitting sources must exclude development')
    x,near,_,ols,bank,scale,debit = pack(d)
    truth = torch.from_numpy(old.parent.model.targets(ols,bank,d['target'],d['valid'],scale)); del bank
    x,near = torch.from_numpy(x),torch.from_numpy(near)
    missing = torch.from_numpy(~d['valid'].any(1))
    score = np.full((len(prev.SEEDS),len(ARMS),len(x),debit.shape[1]),np.nan,dtype=np.float32)
    fits = []
    for k,seed in enumerate(prev.SEEDS):
        for i,site in enumerate(sites):
            held = np.flatnonzero(d['sites'] == site)
            train = np.flatnonzero(d['sites'] != site)
            if np.intersect1d(train,held).size:
                raise ValueError('Held source is in training')
            local = [np.flatnonzero(d['sites'][train] == s) for s in sites if s != site]
            prevalence = float(np.mean([float(missing[train[q]].float().mean()) for q in local]))
            draws = []
            for arm in FIT:
                path = PRIVATE/f'fold_{i}_{seed}_{arm}.pt'
                ident = dict(held=str(site),seed=seed,arm=arm,registration=run.sha(REG),
                             train_ids=base.fit_source.source.array_hash(d['ids'][train]))
                net,usage,rng = fit_fold(x,near,truth,missing,local,train,ident,path,beat,resume)
                draws.append((usage,rng))
                result = predictions(net,x[held],near[held],debit[held],arm,prevalence)
                for a,z in result.items():
                    score[k,ARMS.index(a),held] = z
                fits.append(dict(identity=ident,checkpoint=run.artifact(path),rows=len(train),held_rows=len(held)))
            base.fit_source.source.exact(*draws)
    if not np.isfinite(score).all() or len(fits) != 54:
        raise ValueError('Require all held-source scores without omissions')
    out = PRIVATE/'oof_scores.npz'
    tmp = out.with_suffix('.pending.npz')
    np.savez(tmp,ids=d['ids'],score=score)
    tmp.replace(out)
    receipt = PRIVATE/'oof.json'
    run.once(receipt,run.stable(dict(scores=run.artifact(out),registration=run.sha(REG),fits=fits,
                                    rows=len(x),sources=sites,development_access=False)))
    return receipt


def select(jobs,receipt,beat):
    d = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    if receipt['registration'] != run.sha(REG) or receipt['development_access']:
        raise ValueError('Require the registered fitting-only out-of-fold scores')
    groups = fitting_layout(jobs[0]['identity'],d)
    _,_,cv,ols,bank,scale,debit = pack(d)
    cv_mass,g,h,known = metric.targets(cv,ols,bank,d['target'],d['valid'])
    w = np.empty(len(cv)); src = np.unique(d['sites'],return_inverse=True)[1]
    for q in groups:
        w[q] = 1./len(q)
    mass = np.bincount(src,weights=w*cv_mass)
    if (mass <= 0.).any() or len(mass) != 9:
        raise ValueError('All fitting sources must have positive CV reference mass')
    unit = 100*w/mass[src]
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))
    with np.load(run.checked(receipt['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); score = z['score']
    legal = debit > 1e-6
    policies = {}
    for i,a in enumerate(ARMS):
        rows = []
        for t in metric.THRESHOLDS:
            gs,hs,counts = [],[],[]
            for k in range(len(prev.SEEDS)):
                choice = model.choose(score[k,i]-t,legal,groups); ix = np.flatnonzero(choice >= 0); c = choice[ix]
                gs.append(np.bincount(src[ix],weights=unit[ix]*g[ix,c],minlength=9))
                hs.append(np.bincount(src[ix],weights=unit[ix]*h[ix,c],minlength=9))
                counts.append((len(ix),int((~known[ix]).sum())))
            n,m = np.sum(counts,axis=0)
            rows.append(dict(threshold=t,switches=int(n),wholly_unknown=int(m),
                lower_gain_pp=metric.intervals(np.stack(gs,1),draws),positive_harm_pp=metric.intervals(np.stack(hs,1),draws)))
        nominee = []
        counts = [min(2,len(q)-1,int(legal[q].any(1).sum())) for q in groups]
        for k in range(len(prev.SEEDS)):
            c = model.choose(score[k,i],legal,groups,counts=counts); ix = np.flatnonzero(c >= 0); c = c[ix]
            y = (g[ix,c]-h[ix,c])/(12.*scale[ix])
            nominee.append((score[k,i,ix,c],y,w[ix]))
        b = metric.score_bins(*(np.concatenate(z) for z in zip(*nominee,strict=True)))
        policies[a] = dict(ladder=rows,selected_threshold=metric.choose_threshold(rows,min_sources=MIN_SOURCES),
                           nominated_score_bins=b)
        beat(state='fitting_source_threshold_locked',arm=a,selected=policies[a]['selected_threshold'])
    return dict(policies=policies,role='fitting_only_held_source_diagnostics',sources=receipt['sources'],
        rows=len(w),queries=len(groups),seeds=prev.SEEDS,fits=len(receipt['fits']),bootstrap_draws=3000,
        no_development_targets=True,external_access=False,independent_confirmation=False,official_cv_rule_unchanged=True,
        stage5c_executed=False,smc_enabled=False)


def apply(selected,beat):
    thresholds = {a:selected['policies'][a]['selected_threshold'] for a in ARMS}
    if all(t is None for t in thresholds.values()):
        return dict(thresholds=thresholds,development_scored=False,reason='no_fitting_only_policy_qualifies',
                    no_development_targets=True,independent_confirmation=False)
    d = run.population()
    x,n,cv,ols,bank,_,debit = model.design(d['history'],*model.neighbor_windows(d['history'],d['groups']))
    x,n = torch.from_numpy(x),torch.from_numpy(n)
    legal = debit > 1e-6
    receipt = prev.receipt(old.PUBLIC,'train')
    cells = []
    for s in prev.SEEDS:
        paths = {BASE:cv,OLS:ols}
        for f in receipt['fits']:
            ident = f['identity']; arm = ident['arm']
            if ident['seed'] != s or arm not in FIT:
                continue
            st = torch.load(run.checked(f['checkpoint']),map_location='cpu',weights_only=False)
            if st['identity'] != ident or st['step'] != old.CFG['steps'] or ident['registration'] != run.sha(old.REG):
                raise ValueError('Full fitted checkpoint mismatch')
            net = model.GainSupport(x.shape[-1]); net.load_state_dict(st['model']); net.eval()
            for a,z in predictions(net,x,n,debit,arm,receipt['missing_prevalence_equal_source']).items():
                for name,t in (('raw',0.),('calibrated',thresholds[a])):
                    c = model.choose(z-t,legal,d['groups']) if t is not None else np.full(len(x),-1,dtype=np.int16)
                    p = ols.copy(); take = c >= 0; p[take] = bank[take,c[take]]
                    paths[f'{a}_{name}'] = p
        path = PRIVATE/f'dev_paths_{s}.npz'
        temp = path.with_suffix('.pending.npz')
        np.savez(temp,ids=d['global_ids'],**paths); temp.replace(path)
        cells.append((s,run.artifact(path)))
        beat(state='full_model_development_paths_frozen',seed=s)
    run.once(PRIVATE/'dev_freeze.json',run.stable(dict(cells=cells,thresholds=thresholds,selection=run.artifact(PUBLIC/'analysis.json'))))
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d,lab['frames'])
    roster = {r:str(d['sites'][ix[0]]) for r,(ix,_) in records.items()}
    y = lab['target_eval']-d['origin'][:,None]
    cut = run.read(old.parent.REG)['cut']
    rows = []
    for s,f in cells:
        with np.load(run.checked(f),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            paths = {k:z[k] for k in z.files if k != 'ids'}
        for rec,(ix,_) in records.items():
            r = run.parent.readout.score_recording({k:p[ix] for k,p in paths.items()},PAIRS,y[ix],lab['valid'][ix],
                                                    d['queries'][ix],rec,easy_cut=cut['easy'],hard_cut=cut['hard'])
            rows.append(dict(r,source=roster[rec],seed=s))
    grouped = prev.old.prior.grouped_readout(rows,roster,prev.SEEDS)
    return dict(thresholds=thresholds,grouped=grouped,development_scored=True,independent_confirmation=False,
                original_cv_rule_unchanged=True,external_access=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','oof','select','apply','replay'))
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    _,jobs = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the calibration protocol before execution')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(old.CFG['torch_threads']); torch.set_num_interop_threads(1)

        def beat(**v):
            item = dict(pid=os.getpid(),phase=a.phase,seconds=time.monotonic()-start,**v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json',item)
            print(item,flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resource guard stopped with checkpoints preserved')

        mark = PUBLIC/f'{a.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC,a.phase)
            return
        if a.phase == 'oof':
            out = oof(jobs,beat,a.resume)
        elif a.phase == 'select':
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(select(jobs,prev.receipt(PUBLIC,'oof'),beat)))
        elif a.phase == 'apply':
            out = PUBLIC/'application.json'
            run.once(out,run.stable(apply(prev.receipt(PUBLIC,'select'),beat)))
        else:
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(select(jobs,prev.receipt(PUBLIC,'oof'),beat)))
            run.once(PUBLIC/'application.json',run.stable(apply(prev.receipt(PUBLIC,'select'),beat)))
            prev.receipt(PUBLIC,'apply')
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
