"""Held fitting-source test of geometric gain bounds and causal feature families."""
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
from src.world_model import m3w_ols_bounded_gain as model
from sklearn.metrics import average_precision_score

base,run,prev,np,torch = old.base,old.run,old.prev,old.np,old.torch
PUBLIC = old.PUBLIC.parent/'native_ols_bounded_gain_v1'
PRIVATE = old.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_bounded_gain.py')
ARMS = ('direct',*model.ARMS)
CFG = old.old.CFG


def register():
    _,jobs = old.register()
    prior = prev.receipt(old.PUBLIC,'oof')
    prev.receipt(old.PUBLIC,'replay'); prev.receipt(old.PUBLIC,'apply')
    if prior['development_access'] or len(prior['fits']) != 54:
        raise ValueError('Require original held fitting-source control')
    bindings = (*FILES,old.REG,Path(old.__file__),old.PUBLIC/'oof_execution.json',old.PUBLIC/'replay_execution.json')
    run.once(REG,run.stable(dict(bindings=[run.artifact(f) for f in bindings],arms=ARMS,settings=CFG,seeds=prev.SEEDS,
        primary='bounded_full_vs_direct_matched_query_counts',feature_controls=model.ARMS[1:],bootstrap=3000,
        development_targets=False,external_access=False,official_cv_rule_unchanged=True,threshold_search=False)))
    return jobs,prior


def fitting(jobs,beat):
    d = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    if len(d['ids']) != 69746 or len(set(d['sites'])) != 9 or set(d['sites']) & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Expected source-separated fitting cohort')
    return d


def fit(x,near,target,legal,train,groups,ident,beat,resume):
    seed,arm = ident['seed'],ident['arm']
    torch.manual_seed(seed)
    net = model.network(x.shape[-1]+near.shape[-1])
    opt = torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    step,draws,trace = 0,np.zeros(len(train),np.int64),[]
    path = PRIVATE/f"fold_{ident['held']}_{seed}_{arm}.pt"
    if path.exists():
        if not resume:
            raise FileExistsError('Existing fold requires --resume')
        st = torch.load(path,map_location='cpu',weights_only=False)
        if st['identity'] != ident or not 0 < st['step'] <= CFG['steps']:
            raise ValueError('Mismatched resumed fold')
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); rng.set_state(st['rng'])
        step,draws,trace = (st[k] for k in ('step','draws','trace'))
    while step < CFG['steps']:
        ids = base.native.draw_batch(groups,CFG['batch'],rng); at = train[ids]
        loss = model.objective(model.logits(net,x[at],near[at],arm),target[at],legal[at])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite bounded fit')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step(); step+=1; np.add.at(draws,ids,1)
        if step % CFG['checkpoint'] == 0:
            trace.append(dict(step=step,loss=float(loss.detach())))
            temp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,step=step,model=net.state_dict(),optimizer=opt.state_dict(),
                            rng=rng.get_state(),draws=draws,trace=trace),temp)
            temp.replace(path)
            beat(state='fit_checkpoint',held=ident['held'],arm=arm,seed=seed,**trace[-1])
    if draws.sum() != CFG['steps']*CFG['batch']:
        raise ValueError('Training draw budget changed')
    return net.eval(),run.artifact(path),(draws,rng.get_state())


def train(jobs,prior,beat,resume):
    d = fitting(jobs,beat)
    x,near,cv,ols,bank,scale,debit = old.pack(d)
    _,g,h,_ = old.metric.targets(cv,ols,bank,d['target'],d['valid']); del bank
    unit = 12.*scale[:,None]
    target,legal = model.fractional_target(torch.from_numpy((g/unit).astype(np.float32)),
        torch.from_numpy((h/unit).astype(np.float32)),torch.from_numpy(debit))
    x,near = torch.from_numpy(x),torch.from_numpy(near)
    scores = np.full((len(prev.SEEDS),len(model.ARMS),len(x),debit.shape[1]),np.nan,np.float32)
    sites = sorted(set(d['sites'])); fits = []
    for k,seed in enumerate(prev.SEEDS):
        for site in sites:
            train = np.flatnonzero(d['sites'] != site); held = np.flatnonzero(d['sites'] == site)
            groups = [np.flatnonzero(d['sites'][train] == s) for s in sites if s != site]
            digest = base.fit_source.source.array_hash(d['ids'][train])
            control = [f for f in prior['fits'] if f['identity']['held'] == site and f['identity']['seed'] == seed
                       and f['identity']['arm'] == 'direct']
            if len(control) != 1 or control[0]['identity']['train_ids'] != digest:
                raise ValueError('Held and fitting row IDs must match direct control')
            p = torch.load(run.checked(control[0]['checkpoint']),map_location='cpu',weights_only=False)
            for i,a in enumerate(model.ARMS):
                ident = dict(arm=a,held=str(site),seed=seed,train_ids=digest,registration=run.sha(REG))
                net,ck,state = fit(x,near,target,legal,train,groups,ident,beat,resume)
                base.fit_source.source.exact(state,(p['draws'],p['rng']))
                with torch.inference_mode():
                    scores[k,i,held] = torch.cat([model.score(model.logits(net,x[q],near[q],a),torch.from_numpy(debit[q]))
                                                 for q in np.array_split(held,1+len(held)//1024)]).numpy()
                fits.append(dict(identity=ident,checkpoint=ck,held_rows=len(held)))
    if not np.isfinite(scores).all() or len(fits) != 81:
        raise ValueError('Incomplete out-of-fold score freeze')
    path = PRIVATE/'scores.npz'; tmp = path.with_suffix('.pending.npz')
    np.savez(tmp,ids=d['ids'],scores=scores); tmp.replace(path)
    receipt = PRIVATE/'freeze.json'
    run.once(receipt,run.stable(dict(scores=run.artifact(path),fits=fits,registration=run.sha(REG),past_only_inference=True)))
    return receipt


def evaluate(jobs,prior,frozen,beat):
    if not frozen['past_only_inference'] or frozen['registration'] != run.sha(REG):
        raise ValueError('Expected immutable held source scores')
    d = fitting(jobs,beat); groups = old.fitting_layout(jobs[0]['identity'],d)
    _,_,cv,ols,bank,scale,debit = old.pack(d)
    cv_mass,g,h,known = old.metric.targets(cv,ols,bank,d['target'],d['valid'])
    src = np.unique(d['sites'],return_inverse=True)[1]; w = np.empty(len(cv))
    for q in groups:
        w[q] = 1./len(q)
    mass = np.bincount(src,weights=w*cv_mass)
    if len(mass) != 9 or (mass <= 0.).any():
        raise ValueError('Incomplete CV mass')
    unit = 100*w/mass[src]; legal = debit > 1e-6
    rng = np.random.default_rng(73107); draws = rng.integers(9,size=(3000,9))
    with np.load(run.checked(prior['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); direct = z['score'][:,old.ARMS.index('direct')]
    with np.load(run.checked(frozen['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); scores = np.concatenate((direct[:,None],z['scores']),axis=1)
    collect = {a:[] for a in ARMS}
    for k,seed in enumerate(prev.SEEDS):
        dc = old.model.choose(direct[k],legal,groups)
        counts = [(dc[q] >= 0).sum() for q in groups]
        at_d = np.flatnonzero(dc >= 0); act_d = dc[at_d]
        same_label = g[at_d,act_d] > 0.
        dp = ols.copy(); dp[at_d] = bank[at_d,act_d]
        d_gain = np.zeros(len(cv)); d_gain[at_d] = g[at_d,act_d]
        fc = old.model.choose(scores[k,ARMS.index('bounded_full')],legal,groups,counts=counts)
        f_at = np.flatnonzero(fc >= 0)
        fp = ols.copy(); fp[f_at] = bank[f_at,fc[f_at]]
        f_gain = np.zeros(len(cv)); f_gain[f_at] = g[f_at,fc[f_at]]
        for i,a in enumerate(ARMS):
            c = old.model.choose(scores[k,i],legal,groups,counts=counts); at = np.flatnonzero(c >= 0)
            if a == 'direct':
                np.testing.assert_array_equal(c,dc)
            ca = c[at]; path = ols.copy(); path[at] = bank[at,ca]
            effect = -np.linalg.norm(path-dp,axis=2).max(1)
            effect[known] = -d_gain[known]
            take = known[at]
            effect[at[take]] += g[at[take],ca[take]]
            feature_effect = -np.linalg.norm(fp-path,axis=2).max(1)
            feature_effect[known] = f_gain[known]
            feature_effect[at[take]] -= g[at[take],ca[take]]
            to_mass = lambda rows,v:np.bincount(src[rows],weights=unit[rows]*v,minlength=9)
            uc = old.model.choose(scores[k,i],legal,groups); ui = np.flatnonzero(uc >= 0)
            collect[a].append(dict(gain=to_mass(at,g[at,ca]),harm=to_mass(at,h[at,ca]),
                paired=np.bincount(src,weights=unit*effect,minlength=9),
                full_vs_arm=np.bincount(src,weights=unit*feature_effect,minlength=9),
                free_gain=to_mass(ui,g[ui,uc[ui]]),free_switches=len(ui),switches=len(at),unknown=int((~take).sum()),
                ap=float(average_precision_score(same_label,scores[k,i,at_d,act_d],sample_weight=w[at_d])),
                positive_rate=float(w[at_d]@same_label/w[at_d].sum())))
        beat(state='matched_fitting_scores_read',seed=seed,count=int(len(at_d)))
    result = {}
    for a,rows in collect.items():
        result[a] = {k:old.metric.intervals(np.stack([r[k] for r in rows],1),draws)
                     for k in ('gain','harm','paired','full_vs_arm','free_gain')}
        result[a].update({k:int(sum(r[k] for r in rows)) for k in ('switches','unknown','free_switches')})
        result[a]['same_direct_nominees'] = [{k:r[k] for k in ('ap','positive_rate')} for r in rows]
    old_result = prev.receipt(old.PUBLIC,'select')['policies']['direct']['ladder'][0]
    np.testing.assert_allclose(result['direct']['gain']['point'],old_result['lower_gain_pp']['point'],atol=1e-10)
    if result['direct']['switches'] != old_result['switches']:
        raise ValueError('Existing direct comparison count changed')
    return dict(arms=result,training_fits=len(frozen['fits']),rows=len(cv),queries=len(groups),sources=9,seeds=prev.SEEDS,
        intervention_counts_matched=True,development_targets=False,external_access=False,independent_confirmation=False,
        official_cv_rule_unchanged=True,deployment=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','train','score','replay'))
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    jobs,prior = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit before running')
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
                raise OSError('Resource guard stopped with checkpoints preserved')

        mark = PUBLIC/f'{a.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC,a.phase)
            return
        if a.phase == 'train':
            out = train(jobs,prior,beat,a.resume)
        else:
            if a.phase == 'replay':
                prev.receipt(PUBLIC,'score')
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(evaluate(jobs,prior,prev.receipt(PUBLIC,'train'),beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
