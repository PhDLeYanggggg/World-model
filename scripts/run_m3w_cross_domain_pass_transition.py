"""Learn on OpenDD, then transfer pass-transition scoring to Moving development."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_counterfactual_support as op
from scripts import train_m3w_cv_matched_exchange as moving
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_transfer_pass_transition as model

run, np, torch, study = op.run, op.np, op.torch, op.study
OUT = op.OUT.parent/'cross_domain_pass_transition_v1'
PRIVATE = op.PRIVATE.parent/OUT.name
STEPS = 600
NET_MARGIN, GAIN_MARGIN = .02, .005
ARMS = ('anchor','geometry','gain','transition','matched_gain')
MOMENTS = ('all_excess','easy_excess','easy_signed')


def register():
    key, source, _, producers, upstream, ranks, frozen, _ = op.register()
    mp, mr = (moving.OUT/k for k in ('freeze.json','readout.json'))
    mf,mv = map(run.read,(mp,mr))
    if (mv['freeze']!=run.sha(mp) or not mf['before_held_scoring'] or mv['a_labels_opened']
            or mv['independent_confirmation'] or not mv['original_2pct']):
        raise ValueError('Require frozen Moving fitting-source controls, not Moving A')
    path = OUT/'registration.json'
    run.once(path,run.stable(dict(parents=[run.artifact(f) for f in (op.OUT/'registration.json',op.old.OUT/'freeze.json',mp,mr)],
        files=[run.artifact(f) for f in (Path(__file__),Path(model.__file__),Path(moving.model.__file__),OUT/'protocol.md')],
        op_registration=key,steps=STEPS,seeds=op.m.SEEDS,threads=4,workers=0,original_2pct=True,
        gain_margin=GAIN_MARGIN,transition_margin=NET_MARGIN,a_labels_opened=False,confirmation_access=False)))
    return run.sha(path),source,producers,upstream,ranks,frozen,mf,mv


def packet(d, cv, ols, anchor):
    h, q, rec = d['h'],d['query'],d['rec']
    groups = moving.prior.query_groups(q)
    i,j,k = moving.model.shortlist(cv,ols,anchor,(anchor!=cv).any((1,2)),groups)
    if not len(i):
        return dict(old=i,new=j,query=k,record=rec[i],incoming=cv[i],x=None,scale=np.empty(0),geometry=np.empty(0))
    if not ((q[i]==q[j]) & (rec[i]==rec[j])).all():
        raise ValueError('Invalid exchange boundaries')
    peak = moving.model.peak(anchor[i]-cv[i])
    path = moving.model.incoming_path(cv[j],ols[j],peak)
    past,s = moving.prior_run.context.context(dict(history=h,query=q))
    scale = .5*(s[i]+s[j])
    x = moving.model.features(past,h,cv,ols,s,anchor[i],path,i,j)
    delta = lambda b,c,p: (np.linalg.norm(b-c,axis=-1)-np.linalg.norm(b-p,axis=-1)).mean(1)
    g = (delta(ols[j],cv[j],path)-delta(ols[i],cv[i],anchor[i]))/scale
    return dict(old=i,new=j,query=k,record=rec[i],incoming=path,x=x,scale=scale,geometry=g)


def coefficients(cv, ols, path, y, mask, w, cut):
    r = moments(cv,path,y,mask,cut)
    edited = (path!=cv).any((1,2))
    g = [op.control.paired(a,path,y,mask)[1].sum(1)*w for a in (cv,ols)]
    return np.column_stack((*g,*[r[k]*edited for k in MOMENTS],r['easy']&edited)), r['ref_easy']


def labels(d, cv, ols, anchor, p, cut, query_weighted):
    y,mask,rec,q = (d[k] for k in ('y','valid','rec','query'))
    w = moving.old.weight(q) if query_weighted else np.ones(len(q))
    # The original recording-risk denominator is not query-weighted.
    orig,easy = coefficients(cv,ols,anchor,y,mask,w,cut)
    _,inv = np.unique(rec,return_inverse=True)
    totals = np.stack([np.bincount(inv,weights=z) for z in orig.T],axis=1)
    cap = .02*np.bincount(inv,weights=easy)
    def qualifies(x, upper):
        return ((x[..., :2] > 0.).all(-1) & (x[...,2:4] <= 1e-12).all(-1)
                & (x[...,4] <= upper+1e-12) & (x[...,5] > 0.))
    baseline = qualifies(totals,cap)
    i,j = p['old'],p['new']
    removed,_ = coefficients(cv[i],ols[i],cv[i],y[i],mask[i],w[i],cut)
    added,_ = coefficients(cv[j],ols[j],p['incoming'],y[j],mask[j],w[j],cut)
    z = totals[inv[i]] + removed + added - orig[i] - orig[j]
    actual = qualifies(z,cap[inv[i]])
    delta = (op.control.paired(anchor[i],cv[i],y[i],mask[i])[1].sum(1)*w[i]
             + op.control.paired(cv[j],p['incoming'],y[j],mask[j])[1].sum(1)*w[j])
    gain = delta/(12.*p['scale'])
    return gain.clip(-8.,8.).astype(np.float32), model.transition(baseline[inv[i]],actual),baseline,actual,inv[i]


def batches(training):
    blocks = []
    for x,g,y,p in training:
        if len(g):
            groups = moving.prior.query_groups(p['query'])
            blocks.append((torch.from_numpy(x),torch.from_numpy(g),torch.from_numpy(y),groups))
    if not blocks:
        raise ValueError('Need matched training exchanges')
    def batch(rng):
        x,y,z = [],[],[]
        for t in rng.integers(len(blocks),size=128):
            a,b,c,g = blocks[t]
            i = int(rng.choice(g[rng.integers(len(g))]))
            x.append(a[i]); y.append(b[i]); z.append(c[i])
        return torch.stack(x),torch.stack(y),torch.stack(z)
    return batch


@torch.inference_mode()
def choose(net, p):
    picks = {k:[] for k in ARMS[1:]}
    if p['x'] is None:
        return {k:np.empty(0,dtype=int) for k in picks}
    scores = [net(torch.from_numpy(p['x'][i:i+4096])) for i in range(0,len(p['x']),4096)]
    z = torch.cat(scores)
    gain = z[:,0].numpy()
    prob = z[:,1:].softmax(1).numpy()
    value = prob[:,1]-prob[:,2]
    used = {k:set() for k in ('geometry','gain','transition')}
    for q in moving.prior.query_groups(p['query']):
        r = p['record'][q[0]]
        for k, score, eligible in (('geometry',p['geometry'],p['geometry']>0.),
                                   ('gain',gain,gain>GAIN_MARGIN),
                                   ('transition',value,(value>NET_MARGIN)&(gain>-GAIN_MARGIN))):
            if r in used[k] or not eligible[q].any():
                continue
            at = q[np.where(eligible[q],score[q],-np.inf).argmax()]
            picks[k].append(at)
            used[k].add(r)
            if k=='transition':
                picks['matched_gain'].append(q[gain[q].argmax()])
    return {k:np.asarray(v,dtype=int) for k,v in picks.items()}


def install(cv, anchor, p, at):
    return moving.model.apply(cv,anchor,p['old'],p['new'],p['incoming'],at)


def freeze_cell(net,d,cv,ols,anchor,name):
    p = packet(d,cv,ols,anchor)
    picks = choose(net,p)
    paths = {k:install(cv,anchor,p,z) for k,z in picks.items()}
    f = op.m.save(PRIVATE/f'{name}_paths.npz',dict(anchor=anchor,**paths,**{f'choice_{k}':z for k,z in picks.items()}))
    return f


def train(data, source, key, producers, upstream, ranks, frozen, replay, beat):
    sites = {s:z['d'] for s,z in data.items()}
    banks = {s:op.control.parent.prior.analytic(d) for s,d in sites.items()}
    cache = {}
    def donor(held,s,seed):
        ident = (held,s,seed)
        if ident not in cache:
            _,anchor = op.donor_design(sites,data,source,banks,producers,upstream,ranks,held,s,seed)
            cv,ols = banks[s][:2]
            p = packet(sites[s],cv,ols,anchor)
            cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in sites if t not in (held,s)]),.25)
            gain,tag,*_ = labels(sites[s],cv,ols,anchor,p,cut,False)
            cache[ident] = p['x'],gain,tag,p
        return cache[ident]

    cells, fits, all_nets, update = [], {}, {}, 0
    roster = list(sites)
    for held in (*roster,'all'):
        for seed in op.m.SEEDS:
            name = f'{held}_{seed}'
            use = [s for s in sites if s!=held]
            rows = [(held if held!='all' else roster[(roster.index(s)+1)%len(roster)],s) for s in use]
            data0 = [donor(h,s,seed) for h,s in rows]
            width = next(x.shape[1] for x,g,_,_ in data0 if len(g))
            ident = dict(registration=key,held=held,fit=use,seed=seed,donor_clock_exclusions=[list(z) for z in rows])
            net,f,n = study.fit(ident,name,seed,lambda:model.Value(width),STEPS,batches(data0),model.loss,replay,beat)
            fits[name] = dict(f,options=sum(len(z[1]) for z in data0),rescues=sum(int((z[2]==1).sum()) for z in data0),
                              destroyed=sum(int((z[2]==2).sum()) for z in data0))
            update += n
            if held=='all':
                all_nets[seed] = net
            else:
                c = next(z for z in frozen['cells'] if (z['held'],z['seed'])==(held,seed))
                with np.load(run.checked(c['paths']),allow_pickle=False) as z:
                    anchor = z['anchor']
                f = freeze_cell(net,sites[held],*banks[held][:2],anchor,name)
                cells.append(dict(site=held,seed=seed,cut=c['cut'],paths=f))
            beat(phase='opendd_frozen',held=held,seed=seed,updates=update)
    return cells,fits,all_nets,update


def moving_cells(beat):
    _,donor,_ = moving.base.register()
    data = moving.base.fit_population(beat)
    for c in run.read(moving.OUT/'freeze.json')['cells']:
        ident = c['identity']
        if len(set().union(*(set(ident[k]) for k in ('held','train','calibration')))) != 9:
            raise ValueError('Non-disjoint Moving roles')
        ix,h = moving.base.subset(data,ident['held'])
        cv,ols,raw = moving.prior.bank(h['history'],{k:v[ix] for k,v in moving.prior.read_donor(donor,ident['seed'],data).items()})
        with np.load(run.checked(c['choices']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            a = z['anchor']
        anchor = cv+.5*(moving.old.slate.path_for(cv,moving.base.expanded(cv,raw),a)-cv)
        yield dict(fold=ident['fold'],seed=ident['seed'],cut=c['cut']),dict(h=h['history'],rec=h['recording'],
            query=h['query'],site=h['sites'],y=h['target'],valid=h['valid']),cv,ols,anchor


def score_cell(c, d, cv, ols, query_weighted):
    y,valid,rec,q = (d[k] for k in ('y','valid','rec','query'))
    w = moving.old.weight(q) if query_weighted else np.ones(len(q))
    with np.load(run.checked(c['paths']),allow_pickle=False) as z:
        paths = {k:z[k] for k in ARMS}
        picks = {k:z[f'choice_{k}'] for k in ARMS[1:]}
    p = packet(d,cv,ols,paths['anchor'])
    _,tag,base,option,gid = labels(d,cv,ols,paths['anchor'],p,c['cut'][0],query_weighted)
    names = np.unique(rec)
    can = base.copy()
    can[gid[option]] = True
    best = paths['anchor'].copy()
    # Verify one candidate for every additional feasible recording.
    for i in np.flatnonzero(can & ~base):
        k = np.flatnonzero((gid==i)&option)[0]
        a,b = p['old'][k],p['new'][k]
        best[a],best[b] = cv[a],p['incoming'][k]
    paths['oracle'] = best
    risk = {k:op.stream.policy(z,cv,cv,y,valid,rec,easy_cut=c['cut'][0],hard_cut=c['cut'][1]) for k,z in paths.items()}
    rows = []
    for i,name in enumerate(names):
        at = rec==name
        s, = np.unique(d['site'][at])
        g = {k:[float((op.control.paired(u[at],a[at],y[at],valid[at])[1].sum(1)*w[at]).sum())
                for u in (cv,ols)] for k,a in paths.items()}
        states = {k:bool(risk[k]['recordings'][str(name)]['supported'] and min(g[k])>0.) for k in paths}
        if states['anchor']!=base[i] or states['oracle']!=can[i]:
            raise ValueError('Exact capacity failed original score replay')
        row = {k:dict(dual=states[k],risk=risk[k]['recordings'][str(name)]['supported'],
            undefined=risk[k]['recordings'][str(name)]['completion_easy_risk_upper'] is None,
            easy=risk[k]['recordings'][str(name)]['known_easy_net_degradation'],
            edits=risk[k]['recordings'][str(name)]['selected']) for k in paths}
        for k in ARMS[1:]:
            selected = picks[k][gid[picks[k]]==i]
            if len(selected)>1 or (len(selected) and int(tag[selected[0]]) != int(model.transition(states['anchor'],states[k]))):
                raise ValueError('Actual transition changed its fixed replacement label')
            row[k].update(replaced=len(selected),rescued=bool(states[k] and not states['anchor']),
                          destroyed=bool(states['anchor'] and not states[k]))
        err = np.linalg.norm(cv[at]-y[at],axis=-1)
        mass = float(w[at]@np.where(valid[at],err,0.).sum(1))
        effects = {k:float(w[at]@op.control.paired(paths[k][at],paths['transition'][at],y[at],valid[at])[1].sum(1))
                   for k in ('anchor','geometry','gain','matched_gain')}
        rows.append(dict(site=str(s),seed=c['seed'],rec=str(name),scores=row,reference_mass=mass,effects=effects,
                         options=int((gid==i).sum()),rescue_options=int(((gid==i)&(tag==1)).sum())))
    return rows


def summarize(rows):
    sites = sorted({r['site'] for r in rows})
    draw = np.random.default_rng(817).integers(len(sites),size=(3000,len(sites)))
    by = {(s,k):[r for r in rows if (r['site'],r['seed'])==(s,k)] for s in sites for k in op.m.SEEDS}
    def score(k, part):
        mass = sum(r['reference_mass'] for r in part)
        return 100.*sum(r['effects'][k] for r in part)/mass if mass>0. else None
    effects = {k:op.stream.geometry.bootstrap([[score(k,by[s,j]) for j in op.m.SEEDS] for s in sites],draw)
               for k in rows[0]['effects']}
    totals = {k:dict(**{f:sum(r['scores'][k].get(f,0) for r in rows) for f in
        ('dual','risk','undefined','edits','replaced','rescued','destroyed')},
        worst_easy_percent=100.*max(r['scores'][k]['easy'] for r in rows if r['scores'][k]['easy'] is not None))
        for k in rows[0]['scores']}
    return dict(views=len(rows),sources=len(sites),recordings=len({(r['site'],r['rec']) for r in rows}),
                options=sum(r['options'] for r in rows),rescue_options=sum(r['rescue_options'] for r in rows),
                policies=totals,transition_effects=effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    study.PRIVATE = PRIVATE
    start = time.monotonic()
    with (PRIVATE/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**v):
            state = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',state)
            if v.get('phase')!='fit':
                print(state,flush=True)
        key,source,prod,up,ranks,frozen,_,moving_ref = register()
        data = op.m.load(source)
        oc,fits,nets,n = train(data,source,key,prod,up,ranks,frozen,args.replay,beat)
        mc,inputs = [],[]
        for c,d,cv,ols,anchor in moving_cells(beat):
            f = freeze_cell(nets[c['seed']],d,cv,ols,anchor,f"moving_{c['fold']}_{c['seed']}")
            mc.append(dict(c,paths=f))
            inputs.append((d,cv,ols))
            beat(phase='transfer_frozen',fold=c['fold'],seed=c['seed'],updates=n)
        f = OUT/'freeze.json'
        run.once(f,run.stable(dict(registration=key,opendd=oc,moving=mc,fits=fits,before_scoring=True,
                                   configured_updates=len(fits)*STEPS)))
        o_rows = []
        for c in oc:
            d = dict(data[c['site']]['d'])
            d['site'] = np.full(len(d['h']),c['site'])
            o_rows.extend(score_cell(c,d,*op.control.parent.prior.analytic(d)[:2],False))
        m_rows = [z for c,(d,cv,ols) in zip(mc,inputs,strict=True) for z in score_cell(c,d,cv,ols,True)]
        o,m = summarize(o_rows),summarize(m_rows)
        expected = moving_ref['summary']['risk']['half_anchor']
        if (o['views']!=606 or m['views']!=72 or any(z['edits']!=1457 for z in o['policies'].values())
                or any(z['edits']!=expected['switches'] for z in m['policies'].values())
                or m['policies']['anchor']['risk']!=expected['supported_views']
                or m['policies']['anchor']['undefined']!=expected['undefined_easy_views']):
            raise ValueError('Matched budgets or frozen risk control changed')
        run.once(OUT/'readout.json',run.stable(dict(freeze=run.sha(f),opendd=o,moving=m,opendd_rows=o_rows,moving_rows=m_rows,
            a_labels_opened=False,original_2pct=True,independent_confirmation=False,confirmation_access=False)))
        beat(phase='complete',updates=n,replay=args.replay)


if __name__ == '__main__':
    main()
