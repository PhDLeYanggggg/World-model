"""Train and replay causal OLS4-relative replacement on development sources."""
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
from scripts import run_m3w_dual_reference_forecast as parent
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_ols_replacement as model

base, run, prev, np, torch = parent.base, parent.run, parent.prev, parent.np, parent.torch
PUBLIC = parent.PUBLIC.parent/'native_ols_replacement_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(),Path(model.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_ols_replacement.py')
BASE, OLS = parent.BASE,parent.OLS
LEARNED = model.ARMS[:-1]
CFG = dict(steps=600,batch=128,lr=.001,checkpoint=100,torch_threads=2,workers=0)
PAIRS = {f'{a}_vs_{b}':(b,a,BASE) for a,b in (
    *[(a,b) for a in model.ARMS for b in (BASE,OLS)],
    ('ols_hindcast','ols_motion'),('ols_hindcast','cv_hindcast'),('ols_hindcast','hindcast_rule'))}


def register():
    cut,jobs = parent.register()
    for d in ('dronecrowd_calibration_readout_v1','dronecrowd_forecast_diagnostics_v1'):
        if not (PUBLIC.parent/d/'replay_execution.json').is_file():
            raise ValueError('Require the closed prior calibration readout, not a prospective test')
    bindings = (*FILES,parent.REG,base.fit_source.PUBLIC/'fit_population.json',Path(errors.__code__.co_filename))
    run.once(REG,run.stable(dict(files=[run.artifact(f) for f in bindings],arms=model.ARMS,menu=model.NAMES,
        seeds=prev.SEEDS,settings=CFG,bootstrap=3000,fit_sources=9,dev_sources=14,cut=cut,
        primary='ols_hindcast_vs_history_ols4_velocity',
        comparisons=PAIRS,external_access=False,official_cv_rule_unchanged=True,model_selection=False)))
    return cut,jobs


def train(jobs, beat, resume):
    data = base.fit_source.fitting_only(jobs[0]['identity'],beat)
    sites = set(data['sites'])
    if len(data['ids']) != 69746 or len(sites) != 9 or sites & set(run.old.prior.arrays(('sites',))['sites']):
        raise ValueError('The fitting roster must exclude all exposed development sources')
    x,cv,ols,bank,scale = model.features(data['geometry'][:,:16].reshape(-1,8,2))
    np.testing.assert_allclose(cv,data['geometry'][:,332:356].reshape(-1,12,2),atol=.0001)
    x = torch.from_numpy(x)
    labels = {k:torch.from_numpy(model.targets(ref,bank,data['target'],data['valid'],scale))
              for k,ref in (('cv',cv),('ols',ols))}
    fits = []
    for seed in prev.SEEDS:
        initial,states = None,[]
        for arm in LEARNED:
            torch.manual_seed(seed)
            net = model.network(x.shape[-1])
            if initial is None:
                initial = {k:v.clone() for k,v in net.state_dict().items()}
            else:
                base.fit_source.source.exact(initial,net.state_dict())
            opt = torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001)
            gen = torch.Generator().manual_seed(seed+7919)
            path = PRIVATE/f'{seed}_{arm}.pt'
            ident = dict(seed=seed,arm=arm,registration=run.sha(REG),fit_ids=jobs[0]['identity']['train_ids_sha256'])
            step, trace, draws = 0,[],np.zeros(len(x),dtype=np.int64)
            if path.exists():
                if not resume:
                    raise FileExistsError('Existing checkpoint requires --resume')
                old = torch.load(path,map_location='cpu',weights_only=False)
                if old['identity'] != ident or not 0 < old['step'] <= CFG['steps']:
                    raise ValueError('Continuation identity changed')
                net.load_state_dict(old['model']); opt.load_state_dict(old['optimizer']); gen.set_state(old['rng'])
                step,trace,draws = (old[k] for k in ('step','trace','draws'))
            target = labels['cv' if arm == 'cv_hindcast' else 'ols']
            while step < CFG['steps']:
                ids = base.native.draw_batch(data['groups'],CFG['batch'],gen)
                z = x[ids]
                if arm == 'ols_motion':
                    z[...,-model.HINDCAST_WIDTH:] = 0.
                out = model.prediction(net,z)
                loss = torch.nn.functional.smooth_l1_loss(out,target[ids],beta=.05)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite learning objective')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
                opt.step(); step+=1; np.add.at(draws,ids,1)
                if step % CFG['checkpoint'] == 0:
                    trace.append(dict(step=step,loss=float(loss.detach())))
                    tmp = path.with_suffix('.pending')
                    torch.save(dict(identity=ident,model=net.state_dict(),optimizer=opt.state_dict(),
                        rng=gen.get_state(),step=step,trace=trace,draws=draws),tmp)
                    tmp.replace(path)
                    beat(state='fit_checkpoint',seed=seed,arm=arm,**trace[-1])
            if draws.sum() != CFG['steps']*CFG['batch']:
                raise ValueError('Training budget changed')
            states.append((draws,gen.get_state()))
            fits.append(dict(identity=ident,checkpoint=run.artifact(path),step=step,trace=trace,
                             parameters=sum(p.numel() for p in net.parameters())))
        for s in states[1:]:
            base.fit_source.source.exact(states[0],s)
    dest = PRIVATE/'train.json'
    run.once(dest,run.stable(dict(fits=fits,rows=len(x),sources=len(sites),same_initialization_and_draws=True)))
    return dest


def infer(d, receipt, beat):
    x,cv,ols,bank,_ = model.features(d['history'])
    for k,ref in ((1,cv),(4,ols),(5,bank[:,1])):
        np.testing.assert_allclose(ref,base.fit_source.prep.baseline_numpy(d['history'],k)-d['origin'][:,None],atol=1e-8)
    legal = np.linalg.norm(bank-ols[:,None],axis=3).max(2) > 1e-6
    rule = model.choose(x[...,-4]-x[...,-2],legal,d['groups'])
    if not receipt['same_initialization_and_draws'] or len(receipt['fits']) != 9:
        raise ValueError('Require all matched fitted arms')
    cells = []
    for seed in prev.SEEDS:
        choices = {'hindcast_rule':rule}
        for f in receipt['fits']:
            ident = f['identity']; arm = ident['arm']
            if ident['seed'] != seed:
                continue
            state = torch.load(run.checked(f['checkpoint']),map_location='cpu',weights_only=False)
            if state['identity'] != ident or state['step'] != CFG['steps'] or ident['registration'] != run.sha(REG):
                raise ValueError('Unexpected model checkpoint')
            net = model.network(x.shape[-1]); net.load_state_dict(state['model']); net.eval()
            with torch.inference_mode():
                pred = []
                for a in range(0,len(x),1024):
                    z = torch.from_numpy(x[a:a+1024].copy())
                    if arm == 'ols_motion':
                        z[...,-model.HINDCAST_WIDTH:] = 0.
                    pred.append(model.prediction(net,z).numpy())
                pred = np.concatenate(pred)
            choices[arm] = model.choose(pred[...,0]-pred[...,1],legal,d['groups'])
        if set(choices) != set(model.ARMS):
            raise ValueError('Incomplete inference freeze')
        matrix = np.stack([choices[k] for k in model.ARMS])
        counts = np.stack([(matrix[:,q] >= 0).sum(1) for q in d['groups']])
        if not (counts == counts[:,:1]).all():
            raise ValueError('Different switch counts per query')
        path = PRIVATE/f'choices_{seed}.npz'
        tmp = path.with_suffix('.pending.npz')
        np.savez(tmp,ids=d['global_ids'],choices=matrix)
        tmp.replace(path)
        cells.append(dict(seed=seed,artifact=run.artifact(path),switches=int(counts[:,0].sum())))
        beat(state='choices_frozen',seed=seed,switches=cells[-1]['switches'])
    out = PRIVATE/'freeze.json'
    run.once(out,run.stable(dict(cells=cells,registration=run.sha(REG),past_only=True)))
    return out


def score(d,freeze,cut,beat):
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    blocks = prev.old.capacity.cal.prior.h.previous.prior.blocks(d,lab['frames'])
    roster = {r:str(d['sites'][ix[0]]) for r,(ix,_) in blocks.items()}
    cv,ols,bank,_ = model.rollout(d['history'],12)
    y, valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    n = {a:dict(known_switches=0,unknown_switches=0,harmful_known_switches=0,helpful_known_switches=0,
                ols_disagreement_debit=0.,ols_positive_harm=0.,ols_reference_mass=0.) for a in model.ARMS}
    rows = []
    for cell in freeze['cells']:
        with np.load(run.checked(cell['artifact']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            choices = z['choices']
        for rec,(ix,queries) in blocks.items():
            p = {BASE:cv[ix], OLS:ols[ix]}
            floor_error,_ = errors(p[OLS],y[ix],valid[ix]); known = valid[ix].any(1)
            for j,a in enumerate(model.ARMS):
                c = choices[j,ix]; on = c >= 0
                path = p[OLS].copy(); path[on] = bank[ix[on],c[on]]
                changed = np.linalg.norm(path-p[OLS],axis=2).max(1) > 1e-6
                np.testing.assert_array_equal(changed,on)
                v = (path-p[OLS])[on & ~known]
                e,_ = errors(path,y[ix],valid[ix]); diff = e[on & known]-floor_error[on & known]
                stat = n[a]
                stat['known_switches'] += len(diff); stat['unknown_switches'] += len(v)
                stat['harmful_known_switches'] += int((diff > 1e-6).sum())
                stat['helpful_known_switches'] += int((diff < -1e-6).sum())
                stat['ols_disagreement_debit'] += float(np.linalg.norm(v,axis=2).max(1).sum())
                stat['ols_positive_harm'] += float(np.maximum(diff,0.).sum())
                stat['ols_reference_mass'] += float(floor_error[on & known].sum())
                p[a] = path
            for q in queries:
                if len({int((z[ix[q]] >= 0).sum()) for z in choices}) != 1:
                    raise ValueError('Frozen query counts differ')
            row = run.parent.readout.score_recording(p,PAIRS,y[ix],valid[ix],d['queries'][ix],rec,
                                                      easy_cut=cut['easy'],hard_cut=cut['hard'])
            rows.append(dict(row,source=roster[rec],seed=cell['seed']))
        beat(state='dev_scored',seed=cell['seed'],views=len(rows))
    for s in n.values():
        if s['known_switches']+s['unknown_switches'] != sum(c['switches'] for c in freeze['cells']):
            raise ValueError('Matched budget does not replay')
    grouped = prev.old.prior.grouped_readout(rows,roster,prev.SEEDS)
    return dict(grouped=grouped,ols_switch_diagnostics=n,diagnostic_equal_counts=True,
        score_origin='exposed_development_only',external_access=False,independent_confirmation=False,
        official_cv_rule_unchanged=True,deployment=False,stage5c_executed=False,smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','train','infer','score','replay'))
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    cut,jobs = register()
    if args.phase == 'register':
        return
    for f in (*FILES,REG):
        if subprocess.check_output(['git','show',f'HEAD:{f.relative_to(ROOT)}'],cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit protocol and code before the experiment')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(CFG['torch_threads']); torch.set_num_interop_threads(1)

        def beat(**v):
            item = dict(pid=os.getpid(),phase=args.phase,seconds=time.monotonic()-start,**v)
            base.fit_source.source.atomic(PRIVATE/'heartbeat.json',item)
            print(item,flush=True)
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resource guard stopped with checkpoints preserved')

        mark = PUBLIC/f'{args.phase}_execution.json'
        if mark.exists():
            prev.receipt(PUBLIC,args.phase)
            return
        if args.phase == 'train':
            out = train(jobs,beat,args.resume)
        elif args.phase == 'infer':
            out = infer(run.population(),prev.receipt(PUBLIC,'train'),beat)
        else:
            freeze = prev.receipt(PUBLIC,'infer')
            if not freeze['past_only'] or freeze['registration'] != run.sha(REG):
                raise ValueError('A full past-only freeze must precede development labels')
            if args.phase == 'replay':
                prev.receipt(PUBLIC,'score')
            out = PUBLIC/'analysis.json'
            run.once(out,run.stable(score(run.population(),freeze,cut,beat)))
        run.once(mark,dict(pid=os.getpid(),seconds=time.monotonic()-start,artifact=run.artifact(out)))


if __name__ == '__main__':
    main()
