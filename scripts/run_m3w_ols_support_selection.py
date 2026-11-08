"""Source-separated OLS4 selection with optional abstention and support learning."""
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
from scripts import run_m3w_ols_replacement as parent
from src.world_model import m3w_ols_support_selection as model
from sklearn.metrics import average_precision_score,roc_auc_score

base,run,prev,np,torch = parent.base,parent.run,parent.prev,parent.np,parent.torch
PUBLIC = parent.PUBLIC.parent/'native_ols_support_selection_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_support_selection.py')
BASE,OLS = parent.BASE,parent.OLS
CONTROL = 'factorized_constant'
UNMATCHED = (*model.ARMS,CONTROL)
PRIMARY = 'factorized_neighbor'
ARMS = (*UNMATCHED,*(f'matched_{a}' for a in UNMATCHED if a != PRIMARY),'parent_ols_hindcast')
CFG = dict(steps=600,batch=128,checkpoint=100,lr=.001,torch_threads=2,workers=0)
PAIRS = {f'{a}_vs_{b}':(b,a,BASE) for a,b in (
    *[(a,b) for a in ARMS for b in (OLS,BASE)],
    *[(PRIMARY,a) for a in ARMS if a != PRIMARY])}


def register():
    cut,jobs = parent.register()
    amplitude = PUBLIC.parent/'native_ols_replacement_amplitude_v1'
    prev.receipt(amplitude,'replay'); prev.receipt(parent.PUBLIC,'replay')
    paths = (*FILES,parent.REG,parent.PUBLIC/'infer_execution.json',amplitude/'replay_execution.json',
             base.fit_source.PUBLIC/'fit_population.json',Path(parent.model.__file__))
    run.once(REG,run.stable(dict(files=[run.artifact(p) for p in paths],arms=ARMS,amplitudes=model.AMPLITUDES,
        seeds=prev.SEEDS,settings=CFG,threshold=0.,harm_weight=1.,bootstrap=3000,fit_sources=9,
        primary='factorized_neighbor_vs_history_ols4_velocity',mechanism='factorized_neighbor_vs_matched_factorized_constant',
        no_development_selection=True,external_access=False,official_cv_rule_unchanged=True)))
    return cut,jobs


def train(jobs,beat,resume):
    d = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    sites = set(d['sites'])
    if len(d['ids']) != 69746 or len(sites) != 9 or sites & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('Require the original disjoint training sources')
    g = d['geometry']
    h = g[:,:16].reshape(-1,8,2)
    n = g[:,38:166].reshape(-1,8,8,2)
    mask = g[:,230:294].reshape(-1,8,8).all(2)
    x,near,_,ols,bank,scale,_ = model.design(h,n,mask)
    target = torch.from_numpy(parent.model.targets(ols,bank,d['target'],d['valid'],scale))
    del bank
    missing = torch.from_numpy(~d['valid'].any(1))
    x,near = torch.from_numpy(x),torch.from_numpy(near)
    weights = [float(missing[at].float().mean()) for at in d['groups']]
    prevalence = float(np.mean(weights))
    fits = []
    for seed in prev.SEEDS:
        initial,states = None,[]
        for arm in model.ARMS:
            torch.manual_seed(seed)
            net = model.GainSupport(x.shape[-1])
            if initial is None:
                initial = {k:v.clone() for k,v in net.state_dict().items()}
            else:
                base.fit_source.source.exact(initial,net.state_dict())
            opt = torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001)
            gen = torch.Generator().manual_seed(seed+7919)
            path = PRIVATE/f'{seed}_{arm}.pt'
            ident = dict(seed=seed,arm=arm,registration=run.sha(REG),fit_ids=jobs[0]['identity']['train_ids_sha256'])
            step,trace,draws = 0,[],np.zeros(len(x),dtype=np.int64)
            if path.exists():
                if not resume:
                    raise FileExistsError('Existing checkpoints require --resume')
                st = torch.load(path,map_location='cpu',weights_only=False)
                if st['identity'] != ident or not 0 < st['step'] <= CFG['steps']:
                    raise ValueError('Resume checkpoint has different identity')
                net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); gen.set_state(st['rng'])
                step,trace,draws = (st[k] for k in ('step','trace','draws'))
            while step < CFG['steps']:
                ids = base.native.draw_batch(d['groups'],CFG['batch'],gen)
                cost,logit = net(x[ids],near[ids],ego_support=arm == 'factorized_ego')
                loss,conditional,bce = model.objective(cost,logit,target[ids],missing[ids],factorized=arm != 'direct')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite support objective')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
                opt.step(); step+=1; np.add.at(draws,ids,1)
                if step % CFG['checkpoint'] == 0:
                    trace.append(dict(step=step,loss=float(loss.detach()),cost=float(conditional.detach()),
                                      support_bce=float(bce.detach()),missing=int(missing[ids].sum())))
                    tmp = path.with_suffix('.pending')
                    torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),rng=gen.get_state(),
                                    step=step,trace=trace,draws=draws),tmp)
                    tmp.replace(path)
                    beat(state='fit_checkpoint',seed=seed,arm=arm,**trace[-1])
            if draws.sum() != CFG['steps']*CFG['batch']:
                raise ValueError('Fit budget differs')
            states.append((draws,gen.get_state()))
            fits.append(dict(identity=ident,checkpoint=run.artifact(path),step=step,trace=trace,
                             parameters=sum(p.numel() for p in net.parameters())))
        for s in states[1:]:
            base.fit_source.source.exact(states[0],s)
    out = PRIVATE/'train.json'
    run.once(out,run.stable(dict(fits=fits,rows=len(d['ids']),sources=len(sites),same_draws_and_init=True,
                                 missing_prevalence_equal_source=prevalence)))
    return out


def infer(d,receipt,beat):
    x,near,_,_,_,_,debit = model.design(d['history'],*model.neighbor_windows(d['history'],d['groups']))
    legal = debit > 1e-6
    if not receipt['same_draws_and_init'] or len(receipt['fits']) != 9:
        raise ValueError('All three matched training arms are required')
    old = {c['seed']:c for c in prev.receipt(parent.PUBLIC,'infer')['cells']}
    cells = []
    for seed in prev.SEEDS:
        values,probs = {},{}
        for f in receipt['fits']:
            ident = f['identity']; arm = ident['arm']
            if ident['seed'] != seed:
                continue
            st = torch.load(run.checked(f['checkpoint']),map_location='cpu',weights_only=False)
            if st['identity'] != ident or st['step'] != CFG['steps'] or ident['registration'] != run.sha(REG):
                raise ValueError('Unexpected frozen fitting state')
            net = model.GainSupport(x.shape[-1]); net.load_state_dict(st['model']); net.eval()
            with torch.inference_mode():
                out = [net(torch.from_numpy(x[i:i+1024]),torch.from_numpy(near[i:i+1024]),
                           ego_support=arm == 'factorized_ego') for i in range(0,len(x),1024)]
                cost = torch.cat([z[0] for z in out]).numpy()
                probs[arm] = torch.cat([z[1] for z in out]).sigmoid().numpy()
            values[arm] = model.risk_score(cost,probs[arm],debit,factorized=arm != 'direct')
            if arm == PRIMARY:
                probs[CONTROL] = np.full(len(x),receipt['missing_prevalence_equal_source'],np.float64)
                values[CONTROL] = model.risk_score(cost,probs[CONTROL],debit,factorized=True)
        choices = {a:model.choose(v,legal,d['groups']) for a,v in values.items()}
        counts = [(choices[PRIMARY][q] >= 0).sum() for q in d['groups']]
        choices.update({f'matched_{a}':model.choose(v,legal,d['groups'],counts=counts)
                        for a,v in values.items() if a != PRIMARY})
        with np.load(run.checked(old[seed]['artifact']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            c = z['choices'][parent.model.ARMS.index('ols_hindcast')]
        choices['parent_ols_hindcast'] = np.where(c >= 0,3*c+2,-1)
        if set(choices) != set(ARMS) or set(probs) != set(UNMATCHED):
            raise ValueError('Incomplete choice or score freeze')
        path = PRIVATE/f'choices_{seed}.npz'
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp,ids=d['global_ids'],choices=np.stack([choices[a] for a in ARMS]),
                 probs=np.stack([probs[a] for a in UNMATCHED]))
        tmp.replace(path)
        cells.append(dict(seed=seed,artifact=run.artifact(path),switches={a:int((choices[a] >= 0).sum()) for a in ARMS}))
        beat(state='choices_frozen',**cells[-1])
    out = PRIVATE/'freeze.json'
    run.once(out,run.stable(dict(cells=cells,registration=run.sha(REG),past_only=True)))
    return out


def score(d,freeze,cut,beat):
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    blocks = prev.old.capacity.cal.prior.h.previous.prior.blocks(d,lab['frames'])
    roster = {r:str(d['sites'][ix[0]]) for r,(ix,_) in blocks.items()}
    cv,ols,menu,_ = parent.model.rollout(d['history'],12)
    delta = (menu-ols[:,None])[:,:,None]*np.asarray(model.AMPLITUDES)[None,None,:,None,None]
    bank = ols[:,None]+delta.reshape(len(ols),-1,12,2)
    y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    missing = ~valid.any(1)
    rows,support = [],{a:[] for a in UNMATCHED}
    switches = {a:dict(known=0,unknown=0,harmful=0,helpful=0,missing_debit=0.,positive_harm=0.,selected_ols_error=0.) for a in ARMS}
    for f in freeze['cells']:
        with np.load(run.checked(f['artifact']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices = z['choices']
            for a,p in zip(UNMATCHED,z['probs'],strict=True):
                support[a].append(dict(seed=f['seed'],auroc=float(roc_auc_score(missing,p)),
                                       auprc=float(average_precision_score(missing,p)),brier=float(np.mean((p-missing)**2))))
        for rec,(ix,groups) in blocks.items():
            paths = {BASE:cv[ix],OLS:ols[ix]}
            ref,_ = parent.errors(paths[OLS],y[ix],valid[ix]); unknown = missing[ix]
            for a,c in zip(ARMS,choices[:,ix],strict=True):
                use = c >= 0
                path = paths[OLS].copy(); path[use] = bank[ix[use],c[use]]
                np.testing.assert_array_equal((path != paths[OLS]).any((1,2)),use)
                if a.startswith('matched_'):
                    p = choices[ARMS.index(PRIMARY),ix]
                    if any((c[q] >= 0).sum() != (p[q] >= 0).sum() for q in groups):
                        raise ValueError('Matched controls changed the number in a query')
                take = use & ~unknown
                e,_ = parent.errors(path,y[ix],valid[ix]); diff = e[take]-ref[take]
                z = switches[a]; z['known'] += int(take.sum()); z['unknown'] += int((use&unknown).sum())
                z['harmful'] += int((diff > 1e-6).sum()); z['helpful'] += int((diff < -1e-6).sum())
                z['missing_debit'] += float(np.linalg.norm((path-paths[OLS])[use&unknown],axis=2).max(1).sum())
                z['positive_harm'] += float(np.maximum(diff,0.).sum()); z['selected_ols_error'] += float(ref[take].sum())
                paths[a] = path
            row = run.parent.readout.score_recording(paths,PAIRS,y[ix],valid[ix],d['queries'][ix],rec,
                                                      easy_cut=cut['easy'],hard_cut=cut['hard'])
            rows.append(dict(row,seed=f['seed'],source=roster[rec]))
        beat(state='dev_scored',seed=f['seed'],views=len(rows))
    for a,s in switches.items():
        if s['known']+s['unknown'] != sum(f['switches'][a] for f in freeze['cells']):
            raise ValueError('Frozen OLS4 intervention totals changed')
    grouped = prev.old.prior.grouped_readout(rows,roster,prev.SEEDS)
    legacy = prev.receipt(parent.PUBLIC,'replay')['grouped']['effects']['ols_hindcast_vs_'+OLS]
    got = grouped['effects']['parent_ols_hindcast_vs_'+OLS]
    for name in legacy:
        np.testing.assert_allclose(got[name]['point'],legacy[name]['point'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(got[name]['interval95'],legacy[name]['interval95'],atol=1e-10,rtol=1e-10)
    return dict(grouped=grouped,ols_switches=switches,missing_probability_diagnostic=support,
        missing_prevalence=float(missing.mean()),scope='exposed_development_only',official_cv_rule_unchanged=True,
        independent_confirmation=False,external_access=False,deployment=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','train','infer','score','replay'))
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    cut,jobs = register()
    if a.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit experiment before running')
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
        elif a.phase == 'infer':
            out = infer(run.population(),prev.receipt(PUBLIC,'train'),beat)
        else:
            frozen = prev.receipt(PUBLIC,'infer')
            if not frozen['past_only'] or frozen['registration'] != run.sha(REG):
                raise ValueError('Freeze all past-only forecasts before scoring')
            if a.phase == 'replay':
                prev.receipt(PUBLIC,'score')
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(score(run.population(),frozen,cut,beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
