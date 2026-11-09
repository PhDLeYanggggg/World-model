"""Train a past-only single-query guard on exact easy-reference transitions."""
import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import itertools
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_feasible_recipient_imitation as old
from src.world_model import m3w_counterfactual_support as model

run, np, torch, study = old.run, old.np, old.torch, old.study
stream, control, m = old.prior, old.control, old.m
OUT = old.OUT.parent/'opendd_counterfactual_support_v1'
PRIVATE = old.PRIVATE.parent/OUT.name
STEPS = 600
MARGIN, LOSS_CAP = .005, .20
MODES = ('local', 'whole')
ARMS = ('anchor', 'prefix_match', *MODES, 'gain', 'matched_gain')


def register():
    key, source, forecasts, producers, upstream, ranks = old.register()
    f, r = (old.OUT/k for k in ('freeze.json','readout.json'))
    frozen, result = map(run.read, (f,r))
    if (frozen['registration'] != key or result['freeze'] != run.sha(f) or not frozen['before_held_scoring']
            or result['confirmation_access'] or result['independent_confirmation'] or not result['original_2pct']):
        raise ValueError('Require bound recipient experiment')
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(parents=[run.artifact(k) for k in (f,r)],
        code=[run.artifact(k) for k in (Path(__file__), Path(model.__file__), OUT/'protocol.md')],
        seeds=m.SEEDS, steps=STEPS, gain_margin=MARGIN, reference_loss_cap=LOSS_CAP,
        threads=4, workers=0, original_2pct=True, confirmation_access=False, no_threshold_tuning=True)))
    return run.sha(p), source, forecasts, producers, upstream, ranks, frozen, result


def donor_design(sites, data, source, banks, producers, upstream, ranks, held, site, seed):
    ex = tuple(sorted((held,site)))
    name = '_'.join(ex)+f'_{seed}'
    rank = ranks['rank_'+name]
    count = upstream['count_heads']['count_'+name]
    for ref in (rank,count):
        stream.cal.require_roles(ref['identity'], sites, ex, seed)
    if count['identity']['registration'] != upstream['registration']:
        raise ValueError('Wrong donor count registration')
    d, bank = sites[site], banks[site]
    with np.load(source.PRIVATE/f'{site}_cohort.npz', allow_pickle=False) as z:
        rec, start = z['recording'], z['start']
    net = study.load(rank, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
    menu = study.menus(d, bank, net)
    x, _, _ = stream.base.features(d, bank, menu)
    net = study.load(count, lambda: stream.base.model.Value(x['motion'].shape[1]), stream.base.STEPS)
    with torch.inference_mode():
        score = stream.base.model.score(net(torch.from_numpy(x['motion'])).numpy(), risk=False)
    n = study.model.streaming_count(score, menu['legal'], rec, start)
    frozen = next(v for v in upstream['roster'][held][site] if v['seed'] == seed)
    if n.sum() != frozen['actual'] or (n == 2).sum() != frozen['two_slot_queries']:
        raise ValueError('Frozen donor count changed')
    use = np.arange(2) < n[:, None]
    rows, amp = menu['row'][use], old.v.AMPLITUDES[menu['arm'][use]]
    anchor = bank[0].copy()
    if len(rows):
        anchor[rows] = old.slot.features(d['h'][rows], amp)[3]
    ref = producers['producer_fits']['producer_'+name]
    ident = ref['identity']
    stream.cal.require_roles(dict(ident, omitted=ident['excluded']), sites, ex, seed)
    spline = m.forecast(study.load(ref, m.model.Spline, m.STEPS), data[site]['inputs']['comoving_spline'])
    return old.design(d, bank, anchor, spline), anchor


def blocks(d, cv, packet, anchor):
    query = d['query']
    prefix = Counter()
    out = []
    for q in np.unique(packet['query']):
        slots = np.flatnonzero(packet['query'] == q)
        at = packet['anchors'][slots]
        row = packet['rows'][at]
        record = str(d['rec'][row[0]])
        choices = [np.flatnonzero(packet['slots'] == j) for j in slots]
        pick = np.array([e for e in itertools.product(*choices)
                         if len(set(packet['rows'][list(e)])) == len(e)], dtype=int)
        lo, hi = np.searchsorted(query, q, 'left'), np.searchsorted(query, q, 'right')
        path = np.broadcast_to(cv[lo:hi], (len(pick), hi-lo, *cv.shape[1:])).copy()
        path[np.arange(len(pick))[:, None], packet['rows'][pick]-lo] = packet['paths'][pick]
        change = (path != anchor[None, lo:hi]).any((1,2,3))
        pick, path = pick[change], path[change]
        scale = 12.*np.linalg.norm(np.diff(d['h'][row], axis=1), axis=-1).mean(1).clip(.05).sum()
        peak = control.radius.peak(anchor[row]-cv[row])
        z = packet['x'][pick]-packet['x'][at]
        # The prefix counts original committed slots, not observed future errors.
        context = np.array([np.log1p(prefix[record]), np.log1p(hi-lo), len(slots), np.log1p(peak.sum()/scale)])
        if len(pick):
            width = z.shape[-1]
            feat = np.zeros((len(pick), 2, 2*width), np.float32)
            feat[:, :len(slots)] = np.concatenate((packet['x'][pick], z), axis=-1)
            x = np.c_[feat.reshape(len(pick), -1), np.broadcast_to(context, (len(pick),len(context)))].astype(np.float32)
            out.append(dict(query=int(q), rec=record, lo=lo, hi=hi, x=x, path=path, scale=scale))
        prefix[record] += len(slots)
    return out


def truth(d, cv, anchor, menus, cut):
    y, valid, rec = d['y'], d['valid'], d['rec']
    edited = (cv != anchor).any((1,2))
    easy = old.moments(cv, cv, y, valid, cut)['easy']
    total = Counter(map(str,rec[edited & easy]))
    output = []
    for b in menus:
        a = slice(b['lo'],b['hi'])
        n = ((b['path'] != cv[None,a]).any((2,3)) & easy[a]).sum(1)
        lost = model.loss_events(total[b['rec']], int((easy[a] & edited[a]).sum()), n)
        gain = control.paired(anchor[None,a],b['path'],y[None,a],valid[None,a])[1].sum((1,2))/b['scale']
        output.append(np.c_[gain.clip(-8.,8.),lost].astype(np.float32))
    return output


def batcher(training, mode):
    col = 1 + MODES.index(mode)
    def batch(rng):
        site = rng.integers(len(training), size=128)
        pairs = []
        for i in site:
            x,y = training[i][rng.integers(len(training[i]))]
            row = rng.integers(len(x))
            pairs.append((x[row],y[row,[0,col]]))
        return tuple(torch.from_numpy(np.stack(z)) for z in zip(*pairs, strict=True))
    return batch


@torch.inference_mode()
def choices(nets, menus, anchor):
    paths = {k:anchor.copy() for k in (*MODES,'gain','matched_gain')}
    fired = {k:{} for k in paths}
    for b in menus:
        r = b['rec']
        x = torch.from_numpy(b['x'])
        pred = {k:net(x).numpy() for k,net in nets.items()}
        for k in (*MODES,'gain'):
            if r in fired[k]:
                continue
            z = pred['whole' if k == 'gain' else k]
            on = z[:,0] > MARGIN
            if k != 'gain':
                on &= z[:,1] <= np.log(LOSS_CAP/(1.-LOSS_CAP))
            if not on.any():
                continue
            i = int(np.where(on,z[:,0],-np.inf).argmax())
            a = slice(b['lo'],b['hi'])
            paths[k][a] = b['path'][i]
            fired[k][r] = (b['query'],i)
            if k == 'whole':
                j = int(z[:,0].argmax())
                paths['matched_gain'][a] = b['path'][j]
                fired['matched_gain'][r] = (b['query'],j)
    return paths,fired


def fit(data, source, reg, forecasts, producers, upstream, ranks, frozen, replay, beat):
    sites = {s:z['d'] for s,z in data.items()}
    banks = {s:control.parent.prior.analytic(d) for s,d in sites.items()}
    cells, fits, stats, updates = [], {}, {}, 0

    @lru_cache(maxsize=24)
    def donor(held, site, seed):
        p, anchor = donor_design(sites,data,source,banks,producers,upstream,ranks,held,site,seed)
        d, cv = sites[site], banks[site][0]
        menu = blocks(d,cv,p,anchor)
        cut = np.quantile(np.concatenate([sites[s]['cut_samples'] for s in sites if s not in (held,site)]), .25)
        return [(b['x'],t) for b,t in zip(menu,truth(d,cv,anchor,menu,cut),strict=True)]

    for c in frozen['cells']:
        s, seed = c['held'],c['seed']
        use = [k for k in sites if k != s]
        train = [z for k in use if (z := donor(s,k,seed))]
        name = f'{s}_{seed}'
        z = np.concatenate([a[1] for d in train for a in d])
        stats[name] = dict(options=len(z), local_losses=int(z[:,1].sum()), whole_losses=int(z[:,2].sum()))
        nets = {}
        for k in MODES:
            ident = dict(registration=reg, held=s, fit=use, seed=seed, mode=k)
            width = train[0][0][0].shape[1]
            nets[k], fits[f'{name}_{k}'], n = study.fit(ident,f'{name}_{k}',seed,lambda:model.Value(width),STEPS,
                batcher(train,k), model.objective,replay,beat)
            updates += n
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            paths = {k:z[k] for k in ('anchor','prefix_match')}
        with np.load(run.checked(c['spline']),allow_pickle=False) as z:
            p = old.design(sites[s],banks[s],paths['anchor'],z['comoving_spline'])
        cv = banks[s][0]
        menu = blocks(sites[s],cv,p,paths['anchor'])
        plan,fired = choices(nets,menu,paths['anchor'])
        paths.update(plan)
        query = sites[s]['query']
        groups = m.donor_spline.prior.query_groups(query)
        for g in groups:
            base = np.sort(control.radius.peak(paths['anchor'][g]-cv[g]))
            for path in paths.values():
                np.testing.assert_allclose(np.sort(control.radius.peak(path[g]-cv[g])),base,atol=1e-9,rtol=0.)
        cells.append(dict(held=s,seed=seed,cut=c['cut'],paths=m.save(PRIVATE/f'{name}_paths.npz',paths),
                          fired=fired,spline=c['spline']))
        beat(phase='frozen',held=s,seed=seed,updates=updates)
    return dict(cells=cells,fits=fits,training=stats,before_held_scoring=True),updates


def evaluate(data, frozen, reference):
    rows = []
    contrasts = (('local','whole'),('gain','whole'),('matched_gain','whole'),('prefix_match','whole'),('anchor','whole'))
    ref = {(z['site'],z['seed']):z for z in reference['rows']}
    for c in frozen['cells']:
        s, seed = c['held'],c['seed']
        d = data[s]['d']
        cv,ols = control.parent.prior.analytic(d)[:2]
        y,valid,rec = d['y'],d['valid'],d['rec']
        mass = np.linalg.norm(cv-y,axis=-1)[valid].sum()
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            paths = {k:z[k] for k in ARMS}
        scores = {k:control.parent.score(d,cv,ols,p,c['cut']) for k,p in paths.items()}
        if any(scores[k] != ref[s,seed]['scores'][k] for k in ('anchor','prefix_match')):
            raise ValueError('Original controls did not replay')
        risk = {k:stream.policy(p,cv,cv,y,valid,rec,easy_cut=c['cut'][0],hard_cut=c['cut'][1]) for k,p in paths.items()}
        old_support = {r for r,x in risk['anchor']['recordings'].items() if x['completion_easy_risk_upper'] is not None}
        outcomes = {}
        for k,v in risk.items():
            support = {r for r,x in v['recordings'].items() if x['completion_easy_risk_upper'] is not None}
            easy = [x['known_easy_net_degradation'] for x in v['recordings'].values() if x['known_easy_net_degradation'] is not None]
            outcomes[k] = dict(loss=len(old_support-support), rescue=len(support-old_support),
                hard=v['subsets']['hard']['gain_percent'], worst_easy=100.*max(easy) if easy else None,
                replaced=len(c['fired'].get(k,{})))
        effect = {f'{b}_vs_{a}':float(100.*control.paired(paths[a],paths[b],y,valid)[1].sum()/mass) for a,b in contrasts}
        rows.append(dict(site=s,seed=seed,scores=scores,effects=effect,outcomes=outcomes))
    draw = np.random.default_rng(817).integers(len(data),size=(3000,len(data)))
    ci = lambda fn: stream.geometry.bootstrap([[fn(r) for r in rows if r['site']==s] for s in data],draw)
    summary = dict(policies={k:{f:sum(r['scores'][k][f] for r in rows) for f in
        ('edits','risk_pass','dual_positive','undefined','views')} for k in ARMS},
        effects={k:ci(lambda r:r['effects'][k]) for k in rows[0]['effects']},
        outcomes={k:dict(**{f:sum(r['outcomes'][k][f] for r in rows) for f in ('loss','rescue','replaced')},
                         hard=ci(lambda r:r['outcomes'][k]['hard']),
                         worst_easy=max(r['outcomes'][k]['worst_easy'] for r in rows
                                        if r['outcomes'][k]['worst_easy'] is not None)) for k in ARMS})
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
        reg,source,forecasts,producers,upstream,ranks,frozen,ref = register()
        data = m.load(source)
        payload,u = fit(data,source,reg,forecasts,producers,upstream,ranks,frozen,a.replay,beat)
        f = OUT/'freeze.json'
        run.once(f,run.stable(dict(registration=reg,**payload,configured_updates=STEPS*len(payload['fits']))))
        rows,summary = evaluate(data,run.read(f),ref)
        run.once(OUT/'readout.json',run.stable(dict(freeze=run.sha(f),rows=rows,summary=summary,
            original_2pct=True,confirmation_access=False,independent_confirmation=False,deployment_changed=False)))
        beat(phase='complete',updates=u,replay=a.replay)


if __name__ == '__main__':
    main()
