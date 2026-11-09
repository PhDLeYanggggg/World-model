"""Fit a rare-event decomposition on the same frozen OpenDD query menus."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_counterfactual_support as base
from src.world_model import m3w_factored_counterfactual_support as model

run, np, torch, study = base.run, base.np, base.torch, base.study
control = base.control
OUT = base.OUT.parent/'opendd_factored_counterfactual_support_v1'
PRIVATE = base.PRIVATE.parent/OUT.name
ARMS = ('factor', 'factor_local', 'factor_matched_gain')
REFS = ('anchor','prefix_match','local','whole')


def register():
    key, source, forecasts, producers, upstream, ranks, old, _ = base.register()
    f,r = (base.OUT/k for k in ('freeze.json','readout.json'))
    original, metrics = map(run.read,(f,r))
    if (original['registration'] != key or metrics['freeze'] != run.sha(f) or not original['before_held_scoring']
            or metrics['confirmation_access'] or metrics['independent_confirmation'] or not metrics['original_2pct']):
        raise ValueError('Require frozen same-menu development controls')
    p = OUT/'registration.json'
    run.once(p,run.stable(dict(parents=[run.artifact(k) for k in (f,r)],
        code=[run.artifact(k) for k in (Path(__file__),Path(model.__file__),OUT/'protocol.md')],
        steps=base.STEPS,seeds=base.m.SEEDS,gain_margin=base.MARGIN,loss_cap=base.LOSS_CAP,
        original_2pct=True,confirmation_access=False,post_result_followup=True)))
    return run.sha(p), source, forecasts, producers, upstream, ranks, old, original, metrics


def batches(train):
    def batch(rng):
        out = []
        for i in rng.integers(len(train),size=128):
            x,y = train[i][rng.integers(len(train[i]))]
            j = rng.integers(len(x))
            out.append((x[j],y[j]))
        return tuple(torch.from_numpy(np.stack(z)) for z in zip(*out,strict=True))
    return batch


@torch.inference_mode()
def decide(net, menu, anchor):
    p = {k:anchor.copy() for k in ARMS}
    fired = {k:{} for k in ARMS}
    for b in menu:
        r = b['rec']
        z = net(torch.from_numpy(b['x']))
        local, whole = (u.numpy() for u in model.risks(z))
        g = z[:,0].numpy()
        for k, risk in (('factor',whole),('factor_local',local)):
            if r in fired[k]:
                continue
            eligible = (g > base.MARGIN) & (risk <= base.LOSS_CAP)
            if not eligible.any():
                continue
            i = int(np.where(eligible,g,-np.inf).argmax())
            a = slice(b['lo'],b['hi'])
            p[k][a] = b['path'][i]
            fired[k][r] = (b['query'],i)
            if k == 'factor':
                j = int(g.argmax())
                p['factor_matched_gain'][a] = b['path'][j]
                fired['factor_matched_gain'][r] = (b['query'],j)
    return p,fired


def fit(data, source, key, producers, upstream, ranks, old, original, replay, beat):
    sites = {s:z['d'] for s,z in data.items()}
    bank = {s:control.parent.prior.analytic(d) for s,d in sites.items()}
    source_cells = {(v['held'],v['seed']):v for v in old['cells']}
    cells, fits, updates = [], {}, 0
    for c in original['cells']:
        held, seed = c['held'], c['seed']
        donors = [s for s in sites if s != held]
        train = []
        for s in donors:
            p, anchor = base.donor_design(sites,data,source,bank,producers,upstream,ranks,held,s,seed)
            menu = base.blocks(sites[s],bank[s][0],p,anchor)
            if menu:
                cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in donors if t != s]),.25)
                train.append([(b['x'],y) for b,y in zip(menu,base.truth(sites[s],bank[s][0],anchor,menu,cut),strict=True)])
        ident = dict(registration=key, held=held, fit=donors, seed=seed)
        name = f'{held}_{seed}'
        net, fits[name], n = study.fit(ident,name,seed,lambda:model.Value(train[0][0][0].shape[1]),base.STEPS,
            batches(train),model.objective,replay,beat)
        updates += n
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            paths = {k:z[k] for k in REFS}
        with np.load(run.checked(source_cells[held,seed]['spline']),allow_pickle=False) as z:
            p = base.old.design(sites[held],bank[held],paths['anchor'],z['comoving_spline'])
        cv = bank[held][0]
        menu = base.blocks(sites[held],cv,p,paths['anchor'])
        pred,fired = decide(net,menu,paths['anchor'])
        paths.update(pred)
        for group in base.m.donor_spline.prior.query_groups(sites[held]['query']):
            def peak(z): return np.sort(control.radius.peak(z[group]-cv[group]))
            for k in ARMS:
                np.testing.assert_allclose(peak(paths[k]),peak(paths['anchor']),atol=1e-9,rtol=0.)
        cells.append(dict(held=held,seed=seed,cut=c['cut'],paths=base.m.save(PRIVATE/f'{name}_paths.npz',paths),fired=fired))
        beat(phase='frozen',held=held,seed=seed,updates=updates)
    return dict(cells=cells,fits=fits,configured_updates=len(fits)*base.STEPS,before_held_scoring=True),updates


def evaluate(data, frozen, ref):
    rows = []
    lookup = {(r['site'],r['seed']):r for r in ref['rows']}
    for c in frozen['cells']:
        s,seed = c['held'],c['seed']
        d = data[s]['d']
        cv,ols = control.parent.prior.analytic(d)[:2]
        y,mask,rec = d['y'],d['valid'],d['rec']
        mass = np.linalg.norm(cv-y,axis=-1)[mask].sum()
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            paths = {k:z[k] for k in (*REFS,*ARMS)}
        refrow = lookup[s,seed]
        scores = {k:control.parent.score(d,cv,ols,p,c['cut']) for k,p in paths.items()}
        if any(scores[k]!=refrow['scores'][k] for k in REFS):
            raise ValueError('Same-budget control changed')
        stats = {k:refrow['outcomes'][k] for k in REFS}
        cuts = dict(easy_cut=c['cut'][0],hard_cut=c['cut'][1])
        def policy(z): return base.stream.policy(z,cv,cv,y,mask,rec,**cuts)
        def support(v): return {r for r,x in v['recordings'].items() if x['completion_easy_risk_upper'] is not None}
        anchor = support(policy(paths['anchor']))
        for k in ARMS:
            p = policy(paths[k])
            easy = [x['known_easy_net_degradation'] for x in p['recordings'].values() if x['known_easy_net_degradation'] is not None]
            u = support(p)
            stats[k] = dict(loss=len(anchor-u),rescue=len(u-anchor),replaced=len(c['fired'][k]),
                            hard=p['subsets']['hard']['gain_percent'],worst_easy=100.*max(easy) if easy else None)
        effects = {f'factor_vs_{k}':float(100.*control.paired(p,paths['factor'],y,mask)[1].sum()/mass)
                   for k,p in paths.items() if k!='factor'}
        effects['local_vs_prefix_match'] = float(100.*control.paired(paths['prefix_match'],paths['local'],y,mask)[1].sum()/mass)
        rows.append(dict(site=s,seed=seed,scores=scores,outcomes=stats,effects=effects))
    draw = np.random.default_rng(817).integers(len(data),size=(3000,len(data)))
    ci = lambda fn:base.stream.geometry.bootstrap([[fn(r) for r in rows if r['site']==s] for s in data],draw)
    summary = dict(policies={k:{f:sum(r['scores'][k][f] for r in rows) for f in
        ('edits','risk_pass','dual_positive','undefined','views')} for k in (*REFS,*ARMS)},
        outcomes={k:dict(**{f:sum(r['outcomes'][k][f] for r in rows) for f in ('loss','rescue','replaced')},
            hard=ci(lambda r:r['outcomes'][k]['hard']),
            worst_easy=max(r['outcomes'][k]['worst_easy'] for r in rows if r['outcomes'][k]['worst_easy'] is not None))
            for k in (*REFS,*ARMS)},
        effects={k:ci(lambda r:r['effects'][k]) for k in rows[0]['effects']})
    return rows,summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**kw):
            z = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**kw)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',z)
            if kw.get('phase') != 'fit':
                print(z,flush=True)
        key,source,_,producers,upstream,ranks,old,original,ref = register()
        data = base.m.load(source)
        payload,u = fit(data,source,key,producers,upstream,ranks,old,original,a.replay,beat)
        f = OUT/'freeze.json'
        run.once(f,run.stable(dict(registration=key,**payload)))
        rows,summary = evaluate(data,run.read(f),ref)
        run.once(OUT/'readout.json',run.stable(dict(freeze=run.sha(f),rows=rows,summary=summary,
            original_2pct=True,confirmation_access=False,independent_confirmation=False,deployment_changed=False)))
        beat(phase='complete',updates=u,replay=a.replay)


if __name__ == '__main__':
    main()
