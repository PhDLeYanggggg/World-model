"""Separate offline count-matched reallocation from causal credit carry."""
import fcntl
import os
from pathlib import Path
import sys
import time
from collections import deque

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_paired_nomination as rank

screen, base, prior, model = rank.screen, rank.base, rank.prior, rank.model
np, torch, read, sha, once, stable, artifact, checked = (
    base.np, base.torch, base.read, base.sha, base.once, base.stable, base.artifact, base.checked)
OUT = rank.OUT.parent/'moving_cv_record_allocation_v1'
PRIVATE = rank.PRIVATE.parent/OUT.name
CAP, TTL, PENALTY = base.CAP, 4, 2.
ARMS = ('local', 'offline_gain', 'offline_risk', 'online_local', 'online_carry', 'online_calibrated')


def blocks(q, recording):
    ids = np.unique(recording)
    rows = {r:prior.query_groups(q[recording==r]) for r in ids}
    return {r:[np.flatnonzero(recording==r)[g] for g in groups] for r,groups in rows.items()}


def top(group, path, score, cap):
    eligible = group[path[group]>=0]
    return eligible[np.argsort(-score[eligible],kind='stable')[:cap]]


def offline(path, score, rec_blocks, quota):
    """Know every query's scores and budget; this is not a streaming policy."""
    out = np.full_like(path,-1)
    for groups in rec_blocks.values():
        available = np.concatenate([top(g,path,score,CAP) for g in groups])
        k = sum(int(quota[g[0]]) for g in groups)
        if len(available) < k:
            raise ValueError('A recording lacks its matched budget')
        take = top(available,path,score,k)
        out[take] = path[take]
    return out


def online(path, score, rec_blocks, quota, threshold, *, carry):
    """Spend only present and unexpired past credits. Reset at each recording."""
    out = np.full_like(path,-1)
    retained_spends = 0
    for groups in rec_blocks.values():
        credits = deque()
        spent, earned = 0,0
        for i,g in enumerate(groups):
            while credits and (not carry or credits[0]+TTL < i):
                credits.popleft()
            credits.extend([i]*int(quota[g[0]]))
            earned += int(quota[g[0]])
            take = top(g,path,score,min(CAP,len(credits)))
            take = take[score[take]>=threshold]
            for _ in take:
                retained_spends += int(credits.popleft() < i)
            spent += len(take)
            if spent>earned:
                raise ValueError('An edit used future credit')
            while len(credits)>CAP:
                credits.popleft()
            out[take] = path[take]
    return out,retained_spends


def candidates(pred, score, d, gate):
    groups = blocks(d['query'],d['recording'])
    counts = np.empty(len(score),dtype=np.intp)
    for gs in groups.values():
        for g in gs:
            counts[g] = len(g)
    nomination = rank.nomination(score,pred,gate)
    old = screen.nominate(pred)
    eligible = model.admissible(old[1],old[2],old[3],old[0]>=0,*gate)
    quota = np.empty(len(score),dtype=np.int8)
    for gs in groups.values():
        for g in gs:
            quota[g] = min(CAP,int(eligible[g].sum()))
    return groups,counts,nomination,quota


def priorities(nomination, counts, scale, median):
    path,score,risk,_ = nomination
    gain = score*scale/counts
    priority = gain-PENALTY*np.maximum(risk,0.).sum(1)*scale/median
    return path,gain,priority


def threshold(priority, path):
    return float(np.median(priority[path>=0])) if (path>=0).any() else None


def calibrate(pred, score, d, cv, ols, bank, cut):
    s = prior.model.scale(d['history'])
    out = []
    for gate in base.GRID:
        gs,n,nom,q = candidates(pred,score,d,gate)
        median = float(np.median([len(g) for v in gs.values() for g in v]))
        p,_,value = priorities(nom,n,s,median)
        t = threshold(value,p)
        z = np.full_like(p,-1) if t is None else online(p,value,gs,q,t,carry=True)[0]
        rows = base.metrics(cv,ols,bank,d,{'strict':z},cut)
        effects = [[prior.prior.parent.readout.merge_comparisons([r['comparisons'][f'strict_vs_{k}']
                    for r in rows if r['source']==g])['completion_lower_pp'] for k in ('cv','ols4')]
                   for g in sorted(set(d['sites']))]
        out.append(dict(tolerance=gate[0],min_easy=gate[1],priority_cut=t,median_query_size=median,
            edits=int((z>=0).sum()),passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            gain_pp=np.mean(effects,axis=0).tolist()))
    best = model.choose_setting(out,len(np.unique(d['recording'])))
    pick = next((r for r in out if (r['tolerance'],r['min_easy'])==best),None)
    relaxed = next(r for r in out if (r['tolerance'],r['min_easy'])==screen.RELAXED)
    return out,pick,relaxed


def frozen_cell(old, rank_cell, d, donor):
    ident = old['identity']
    if ident != rank_cell['identity']:
        raise ValueError('Parent roles do not align')
    fold,seed = ident['fold'],ident['seed']
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if any(len(g)!=3 for g in roles) or len(set.union(*roles))!=9:
        raise ValueError('Source roles must be disjoint')
    _,cal = base.subset(d,ident['calibration'])
    hi,held = base.subset(d,ident['held'])
    width = prior.study.motion.features(cal['history'][:1],'prefix')[0].shape[1]
    risk = screen.load_head(old['fits'][0],width+25,8)
    pair_fit = rank_cell['fits'][1]
    expected = dict(ident,arm='paired',registration=sha(rank.OUT/'registration.json'),weight=rank.RANK_WEIGHT)
    if pair_fit['identity'] != expected or old['fits'][0]['identity'] != dict(ident,head='risk'):
        raise ValueError('Unexpected model lineage')
    net = screen.load_head(pair_fit,width+49,2)
    c = prior.study.motion.bank(cal['history'])
    cb = base.expanded(c[:,0],c[:,1:])
    cp = screen.predictions(risk,c[:,0],cb,cal['history'])
    cs = rank.forecast(net,c[:,0],c[:,1],cb,cal['history'])
    ladder,selected,relaxed = calibrate(cp,cs,cal,c[:,0],c[:,1],cb,old['cut'])
    cv,ols,raw = prior.bank(held['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    bank = base.expanded(cv,raw)
    h = held['history']
    scale = prior.model.scale(h)
    pred = screen.predictions(risk,cv,bank,h)
    score = rank.forecast(net,cv,ols,bank,h)
    g,n,nom,q = candidates(pred,score,held,screen.RELAXED)
    p,value,priority = priorities(nom,n,scale,relaxed['median_query_size'])
    with np.load(checked(rank_cell['actions']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],held['ids'])
        local = z['paired_match']
    for gs in g.values():
        for block in gs:
            want = top(block,p,nom[1],q[block[0]])
            np.testing.assert_array_equal(np.sort(want),block[local[block]>=0])
            np.testing.assert_array_equal(local[want],p[want])
    t = relaxed['priority_cut'] if relaxed['priority_cut'] is not None else np.inf
    o, _ = online(p,priority,g,q,t,carry=False)
    carry, late = online(p,priority,g,q,t,carry=True)
    calibrated = np.full_like(p,-1)
    if selected is not None:
        gs,nc,nm,qc = candidates(pred,score,held,(selected['tolerance'],selected['min_easy']))
        pc,_,rc = priorities(nm,nc,scale,selected['median_query_size'])
        calibrated = online(pc,rc,gs,qc,selected['priority_cut'],carry=True)[0]
    acts = dict(local=local,offline_gain=offline(p,value,g,q),offline_risk=offline(p,priority,g,q),
                online_local=o,online_carry=carry,online_calibrated=calibrated)
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choices_{fold}_{seed}.npz',dict(ids=held['ids'],**acts))
    return dict(identity=ident,cut=old['cut'],actions=path,selection=selected,ladder=ladder,carried_spends=late,
        calibration_recordings=len(np.unique(cal['recording'])))


def score(cells,d,donor,beat):
    scorer = prior.prior.parent.readout
    pairs = {f'{a}_vs_{tag}':(b,a) for a in ARMS for tag,b in (('cv',base.CV),('ols4',base.OLS))}
    pairs.update(offline_risk_vs_local=('local','offline_risk'),offline_gain_vs_local=('local','offline_gain'),
        offline_risk_vs_gain=('offline_gain','offline_risk'),online_carry_vs_local=('online_local','online_carry'))
    rows, moved = [], dict(offline_gain=0,offline_risk=0)
    changes = {k:dict(gained_pass=0,lost_pass=0,undefined_to_defined=0,defined_to_undefined=0)
               for k in ('offline_gain','offline_risk','online_carry')}
    for cell in cells:
        ident = cell['identity']
        hi,held = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(held['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        with np.load(checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],held['ids'])
            acts = {k:z[k] for k in ARMS}
        for rec, gs in blocks(held['query'],held['recording']).items():
            ix = held['recording']==rec
            q,y,v = (held[k][ix] for k in ('query','target','valid'))
            c = cv[ix]
            counts = {k:[int((a[g]>=0).sum()) for g in gs] for k,a in acts.items()}
            for name in moved:
                diff = np.array(counts[name])-counts['local']
                if diff.sum()!=0 or max(counts[name])>CAP:
                    raise ValueError('Offline recording budget not matched')
                moved[name] += int(np.maximum(diff,0).sum())
            if (np.cumsum(counts['online_carry'])>np.cumsum(counts['local'])).any():
                raise ValueError('Online carry spent future credit')
            paths = {base.CV:c,base.OLS:ols[ix],**{k:prior.apply(c,bank[ix],a[ix]) for k,a in acts.items()}}
            r = scorer.score_recording(paths,{},y,v,q,str(rec),easy_cut=cell['cut'][0],hard_cut=cell['cut'][1])
            for k,t in changes.items():
                before = r['metrics']['online_local' if k=='online_carry' else 'local']['risk']
                after = r['metrics'][k]['risk']
                t['gained_pass'] += int(after['supported'] and not before['supported'])
                t['lost_pass'] += int(before['supported'] and not after['supported'])
                x,y0 = [z['completion_easy_risk_upper'] is not None for z in (before,after)]
                t['undefined_to_defined'] += int(not x and y0)
                t['defined_to_undefined'] += int(x and not y0)
            _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
            comparison = {k:prior.study.motion.full_masses(paths[a],paths[b],c,y,v,1./n[inv])
                          for k,(a,b) in pairs.items()}
            source, = np.unique(held['sites'][ix])
            rows.append(dict(r,source=str(source),seed=seed,comparisons=comparison))
        beat(phase='held_scored',fold=ident['fold'],seed=seed)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if (summary['views'],summary['source_groups'],summary['risk']['local']['switches'])!=(72,9,2317):
        raise ValueError('Wrong fitting source cohort')
    return summary,moved,changes


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if 'phase' in v:
                print(info,flush=True)

        reg,donor,_ = base.register()
        parent,rp = base.OUT/'freeze.json',rank.OUT/'freeze.json'
        f,r = read(parent),read(rp)
        if (f['registration']!=reg or r['registration']!=sha(rank.OUT/'registration.json')
                or not all(z['before_held_scoring'] and len(z['cells'])==9 for z in (f,r))):
            raise ValueError('Missing parent freezes')
        registration = OUT/'registration.json'
        once(registration,stable(dict(parents=[artifact(parent),artifact(rp)],files=[artifact(p) for p in
            (Path(__file__),Path(rank.__file__),OUT/'protocol.md')],risk_price=PENALTY,expiry_queries=TTL,
            cap=CAP,threshold_quantile=.5,original_2pct=True,independent_confirmation=False)))
        d = base.fit_population(beat)
        freeze = OUT/'freeze.json'
        if not freeze.exists():
            cells = []
            for a,b in zip(f['cells'],r['cells'],strict=True):
                cell = frozen_cell(a,b,d,donor)
                cells.append(cell)
                beat(phase='choice_frozen',fold=cell['identity']['fold'],seed=cell['identity']['seed'])
            once(freeze,stable(dict(registration=sha(registration),cells=cells,before_held_scoring=True)))
        bank = read(freeze)
        if bank['registration']!=sha(registration) or not bank['before_held_scoring'] or len(bank['cells'])!=9:
            raise ValueError('All cells must be frozen before scoring')
        summary,moved,changes = score(bank['cells'],d,donor,beat)
        cells = bank['cells']
        once(OUT/'readout.json',stable(dict(freeze=sha(freeze),summary=summary,offline_cross_query_edits=moved,
            online_spent_past_credits=sum(c['carried_spends'] for c in cells),risk_transitions=changes,
            qualified=sum(c['selection'] is not None for c in cells),
            calibration_max_passes=[max(r['passes'] for r in c['ladder']) for c in cells],optimizer_updates=0,
            a_labels_opened=False,independent_confirmation=False,original_2pct=True)))
        beat(phase='complete')


if __name__=='__main__':
    main()
