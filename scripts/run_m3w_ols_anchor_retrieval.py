"""Does a past-prefix signal identify useful fixed OLS4 anchor replacements?"""
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
from scripts import run_m3w_ols_anchor_denoising as old
from src.world_model import m3w_ols_anchor_retrieval as model
from sklearn.metrics import average_precision_score

run,prev,base,np,torch = old.run,old.prev,old.base,old.np,old.torch
PUBLIC = old.PUBLIC.parent/'native_ols_anchor_retrieval_v1'
PRIVATE = old.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_anchor_retrieval.py')
CFG = old.CFG


def register():
    _,jobs = old.register()
    if not prev.receipt(old.PUBLIC,'replay')['reporting_amendment_only']:
        raise ValueError('Require the previous source-supported forecast readout')
    paths = (*FILES,old.REG,Path(old.model.__file__),old.PUBLIC/'replay_execution.json',old.PUBLIC/'capacity.json')
    run.once(REG,run.stable(dict(files=[run.artifact(f) for f in paths],fit=model.FITS,arms=model.ARMS,
        fraction=model.FRACTION,margin=model.MARGIN,settings=CFG,seeds=prev.SEEDS,bootstrap=3000,
        primary='learned_probe_vs_ols4_matched_query_counts',role='leave_one_fitting_source_of_nine',
        at_most_one_per_multi_agent_query=True,selection_or_threshold_search=False,
        development_targets=False,external_access=False,official_cv_rule_unchanged=True)))
    return jobs


def fit(x,y,legal,train,groups,ident,beat,resume):
    seed,arm = ident['seed'],ident['arm']; torch.manual_seed(seed)
    net = model.network(x.shape[-1]); opt = torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    step,draws,trace = 0,np.zeros(len(train),np.int64),[]
    path = PRIVATE/f"{ident['held']}_{seed}_{arm}.pt"
    if path.exists():
        if not resume:
            raise FileExistsError('Existing checkpoint requires --resume')
        st = torch.load(path,map_location='cpu',weights_only=False)
        if st['identity'] != ident or not 0 < st['step'] <= CFG['steps']:
            raise ValueError('Resume fold identity differs')
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); rng.set_state(st['rng'])
        step,draws,trace = (st[k] for k in ('step','draws','trace'))
    while step < CFG['steps']:
        i = base.native.draw_batch(groups,CFG['batch'],rng); at = train[i]
        prediction = model.score(net,x[at],arm)
        ce = torch.nn.functional.cross_entropy(prediction,y[at],reduction='none')
        loss = (ce*legal[at]).sum()/legal[at].sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite retrieval training loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step(); step+=1; np.add.at(draws,i,1)
        if step % CFG['checkpoint'] == 0:
            trace.append(dict(step=step,loss=float(loss.detach())))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=ident,step=step,model=net.state_dict(),optimizer=opt.state_dict(),
                            rng=rng.get_state(),draws=draws,trace=trace),tmp)
            tmp.replace(path)
            beat(state='fit_checkpoint',held=ident['held'],seed=seed,arm=arm,**trace[-1])
    if draws.sum() != CFG['steps']*CFG['batch']:
        raise ValueError('Fitting draws do not match the fixed budget')
    return net.eval(),run.artifact(path),(draws,rng.get_state())


def train(jobs,beat,resume):
    d = old.data(jobs,beat)
    x,_,_,floor,change = model.past_features(d['geometry'][:,:16].reshape(-1,8,2))
    label,_,eligible,_ = model.supervision(floor,change,d['target'],d['valid'])
    x,y,legal = (torch.from_numpy(z) for z in (x,label,eligible))
    out = np.full((len(prev.SEEDS),len(model.FITS),len(y)),np.nan,np.float32)
    sites = sorted(set(d['sites'])); fits = []
    for k,seed in enumerate(prev.SEEDS):
        for site in sites:
            train = np.flatnonzero(d['sites'] != site); held = np.flatnonzero(d['sites'] == site)
            groups = [np.flatnonzero(d['sites'][train] == s) for s in sites if s != site]
            states = []
            for j,a in enumerate(model.FITS):
                ident = dict(seed=seed,arm=a,held=str(site),registration=run.sha(REG),
                             train_ids=base.fit_source.source.array_hash(d['ids'][train]))
                net,artifact,stats = fit(x,y,legal,train,groups,ident,beat,resume)
                states.append(stats)
                with torch.inference_mode():
                    p = model.score(net,x[held],a).softmax(1)
                    out[k,j,held] = (p[:,2]-p[:,0]).numpy()
                fits.append(dict(identity=ident,checkpoint=artifact,held_rows=len(held)))
            base.fit_source.source.exact(*states)
    if not np.isfinite(out).all() or len(fits) != 54:
        raise ValueError('Incomplete held-source predictions')
    path = PRIVATE/'scores.npz'; tmp = path.with_suffix('.pending.npz')
    np.savez(tmp,ids=d['ids'],scores=out); tmp.replace(path)
    receipt = PRIVATE/'freeze.json'
    run.once(receipt,run.stable(dict(registration=run.sha(REG),scores=run.artifact(path),fits=fits,
                                    past_only=True,matched_draws=True)))
    return receipt


def score(jobs,frozen,beat):
    if frozen['registration'] != run.sha(REG) or not frozen['past_only'] or not frozen['matched_draws']:
        raise ValueError('Require committed past-only score freeze')
    d = old.data(jobs,beat); queries = old.old.fitting_layout(jobs[0]['identity'],d)
    _,rule,cv,ols,change = model.past_features(d['geometry'][:,:16].reshape(-1,8,2))
    labels,gain,legal,known = model.supervision(ols,change,d['target'],d['valid'])
    with np.load(run.checked(frozen['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); scores = z['scores']
    y,v = d['target'],d['valid']; clean = np.where(v[...,None],y,0.)
    cv_ade = np.where(v,np.linalg.norm(cv-clean,axis=2),0.).sum(1)/v.sum(1).clip(1)
    w = np.empty(len(y)); src = np.unique(d['sites'],return_inverse=True)[1]
    for q in queries:
        w[q] = 1./len(q)
    cv_mass = np.bincount(src,weights=w*cv_ade)
    if len(cv_mass) != 9 or (cv_mass <= 0.).any():
        raise ValueError('Positive CV reference needed in every fitting group')
    scale = 100*w/cv_mass[src]
    radius = np.linalg.norm(change-ols,axis=2).max(1)
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))
    records = {a:[] for a in model.ARMS}
    for k,seed in enumerate(prev.SEEDS):
        rand = np.random.default_rng(seed+20261108).random(len(y))
        p = dict(zip(model.FITS,scores[k],strict=True)); p.update(prefix_rule=rule,random=rand,oracle=gain)
        chosen = {a:model.choose(value,legal,queries) for a,value in p.items()}
        ref = chosen['learned_probe']
        for a,c in chosen.items():
            np.testing.assert_array_equal([c[q].sum() for q in queries],[ref[q].sum() for q in queries])
            optional = model.choose(p[a],legal,queries,threshold=0.)
            diff = np.where(known,gain*(ref.astype(float)-c),-radius*(ref^c))
            mass = lambda val:np.bincount(src,weights=val*scale,minlength=9)
            path = np.where(c[:,None,None],change,ols)
            ade = np.where(v,np.linalg.norm(path-clean,axis=2),0.).sum(1)/v.sum(1).clip(1)
            lower_cv = np.where(known,cv_ade-ade,-np.linalg.norm(cv-path,axis=2).max(1))
            records[a].append(dict(lower=mass(gain*c),harm=mass(np.maximum(-gain,0.)*c),vs_cv=mass(lower_cv),
                full_vs_arm=mass(diff),optional_lower=mass(gain*optional),switches=int(c.sum()),
                positive=int((c&(gain>0.)).sum()),unknown=int((c&~known).sum()),optional_switches=int(optional.sum()),
                ap=float(average_precision_score(labels[legal]==2,p[a][legal],sample_weight=w[legal]))))
        beat(state='matched_fitting_source_roster_scored',seed=seed,switches=int(ref.sum()))
    result = {a:dict(**{m:old.old.metric.intervals(np.stack([r[m] for r in rows],1),draws)
                    for m in ('lower','harm','vs_cv','full_vs_arm','optional_lower')},
        **{m:int(sum(r[m] for r in rows)) for m in ('switches','positive','unknown','optional_switches')},
        ap_by_seed=[r['ap'] for r in rows]) for a,rows in records.items()}
    return dict(arms=result,oracle_uses_labels_only_in_scoring=True,wholly_unknown_rows=int((~known).sum()),
        fitting_positive_label_rate=float(w[legal]@(labels[legal]==2)/w[legal].sum()),source_groups=9,rows=len(y),
        queries=len(queries),fits=len(frozen['fits']),seeds=prev.SEEDS,development_targets=False,
        external_access=False,independent_confirmation=False,official_cv_rule_unchanged=True,deployment=False,
        stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','train','score','replay'))
    p.add_argument('--resume',action='store_true')
    a = p.parse_args(); jobs = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the experiment before execution')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG['torch_threads']); torch.set_num_interop_threads(1)

        def beat(**v):
            e = dict(pid=os.getpid(),phase=a.phase,seconds=time.monotonic()-start,**v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json',e)
            print(e,flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resource guard stopped with atomic checkpoints preserved')

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
            run.once(out,run.stable(score(jobs,prev.receipt(PUBLIC,'train'),beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
