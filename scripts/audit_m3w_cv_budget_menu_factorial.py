"""Cross frozen query counts with menu width on held development sources."""
from collections import Counter
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_fixed_slate_capacity as old
from scripts import audit_m3w_cv_uncapped_dual_certificate as uncapped
from src.evaluation import m3w_cv_budget_menu_factorial as design

exp = old.exp
np,base,prior = exp.np,exp.base,exp.prior
OUT = exp.OUT.parent/'moving_cv_budget_menu_factorial_v1'
PRIVATE = exp.PRIVATE.parent/OUT.name
ARMS = ('narrow_exact','full_exact','narrow_relaxed','full_relaxed')
SECONDS = 8.


def check(actions,args,anchor,arm):
    c,o,b,y,v,q,*cut = args
    a = np.asarray(actions,dtype=int).reshape(-1,2)
    row,action = a.T
    _,group,counts = np.unique(q,return_inverse=True,return_counts=True)
    if arm.endswith('_exact') and not np.array_equal(
            np.bincount(group[row],minlength=len(counts)),np.bincount(group[anchor>=0],minlength=len(counts))):
        raise ValueError('Witness changed a frozen query count')
    if arm.startswith('narrow'):
        chosen = np.full(len(q),-1,dtype=int)
        chosen[row] = action
        for g in prior.query_groups(q):
            k = int((chosen[g]>=0).sum())
            if not k or np.array_equal(chosen[g],anchor[g]):
                continue
            if not any(len(r)>=k and np.array_equal(chosen[g[r[:k]]],j[:k])
                       for r,j in design.prefix_orders(c[g],o[g],b[g])):
                raise ValueError('Witness selected an out-of-menu slate')
    return old.verify(a,*args,int(counts.max()))


def count_match(cells,pooled):
    by_role = {tuple(c['identity']['held']):c['cut'] for c in pooled['cells']}
    if len(by_role)!=3 or any(by_role.get(tuple(c['identity']['held']))!=c['cut'] for c in cells):
        raise ValueError('Prior full bank used different risk cuts')
    if [c['identity'] for c in cells]!=[c['identity'] for c in pooled['cells']]:
        raise ValueError('Prior full bank used a different source/seed roster')


def result(args,anchor,arm):
    c,o,b,y,v,q,easy,_ = args
    fn = design.narrow if arm.startswith('narrow') else design.full
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message="Unrecognized options detected: .*objective_target.*",category=RuntimeWarning)
        a,status,upper = fn(c,o,b,y,v,q,easy,anchor,arm.endswith('_exact'),SECONDS)
    if a is not None:
        proof = check(a,args,anchor,arm)
        if proof['verified']:
            return dict(status='verified',actions=a.tolist(),edits=proof['edits'],gain_pp=proof['gain_pp'],upper_pp=upper)
    return dict(status=status,actions=None,upper_pp=upper)


def historical_narrow(args,anchor):
    c,o,b,y,v,q,easy,_ = args
    w = exp.weight(q)
    _,ref,offset,cap = old.slate.contributions(c,o,c,y,v,w,easy)
    choices,blocks = [],[]
    for g in prior.query_groups(q):
        k = int((anchor[g]>=0).sum())
        if not k:
            continue
        options = list({a.tobytes():a for a in [anchor[g],*old.slate.menu(c[g],o[g],b[g],k)]}.values())
        choices.append((g,options))
        blocks.append(np.stack([old.slate.contributions(c[g],o[g],old.slate.path_for(c[g],b[g],a),
                                y[g],v[g],w[g],easy)[0] for a in options]))
    take,status = old.solve(blocks,ref,offset,cap)
    if take is not None:
        a = np.full(len(c),-1,dtype=np.int8)
        for (g,options),j in zip(choices,take,strict=True):
            a[g] = options[j]
        ix = np.flatnonzero(a>=0)
        edits = np.column_stack((ix,a[ix]))
        proof = check(edits,args,anchor,'narrow_exact')
        if proof['verified']:
            return dict(status='verified',actions=edits.tolist(),edits=proof['edits'],
                        gain_pp=proof['gain_pp'],upper_pp=None)
    return dict(status=status,actions=None,upper_pp=None)


def cached_full(i,r,args,anchor,previous,continuation):
    parent = uncapped.study.PRIVATE
    z = exp.read(parent/f'capacity_{i}_{r}.json')
    if z['binding']!=dict(cell=i,recording=int(r),parent=previous['parent'],code=previous['code']):
        raise ValueError('Prior full-bank binding changed')
    out = z['witness']['uncapped']
    if out['status']=='unresolved':
        z = exp.read(parent/f'dual_{i}_{r}.json')
        if z['binding']!={k:continuation[k] for k in ('parent','code')}:
            raise ValueError('Prior continuation binding changed')
        z = z['result']
        out = dict(status=z['status'],actions=None,upper_pp=z['upper_pp'])
        if z['witness'] is not None:
            out.update(z['witness'])
    if out['status']=='verified' and not check(out['actions'],args,anchor,'full_relaxed')['verified']:
        raise ValueError('Cached full-bank witness failed original scorer')
    return out


def summarize(rows):
    names = sorted({r['source'] for r in rows})
    if len(names)!=9 or len(rows)!=72:
        raise ValueError('Incomplete source roster')
    v = np.array([[r[k]['status']=='verified' for k in ARMS] for r in rows],float)
    local = np.array([v[[r['source']==s for r in rows]].mean(0) for s in names])
    draw = np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    contrasts = {}
    for label,i,j in (('expand_at_exact',0,1),('relax_with_narrow',0,2),
                      ('expand_at_relaxed',2,3),('relax_with_full',1,3)):
        d = 100.*(local[:,j]-local[:,i])
        contrasts[label] = dict(source_balanced_pp=float(d.mean()),
            interval95_pp=np.quantile(d[draw].mean(1),[.025,.975]).tolist(),
            verified_view_difference=int((v[:,j]-v[:,i]).sum()))
    return dict(contrasts=contrasts,bootstrap_draws=len(draw),source_groups=len(names),views=len(rows),
        counts={k:dict(Counter(r[k]['status'] for r in rows)) for k in ARMS},
        verified_only_median_switch_percent={k:float(np.median(
            [100*r[k]['edits']/r['rows'] for r in rows if r[k]['status']=='verified']))
            if any(r[k]['status']=='verified' for r in rows) else None for k in ARMS})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'factorial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t = time.monotonic()
        p,pool = exp.OUT,uncapped.study.OUT
        files = [p/k for k in ('freeze.json','readout.json','fixed_menu_capacity.json')]
        files += [pool/k for k in ('freeze.json','hindsight_capacity.json','uncapped_continuation.json')]
        f,r,n,po,u,cont = map(exp.read,files)
        if (r['freeze']!=exp.sha(files[0]) or n['parent']['sha256']!=exp.sha(files[1])
                or cont['parent']['sha256']!=exp.sha(files[4])
                or any(z.get('a_labels_opened',False) or z['independent_confirmation'] or not z['original_2pct']
                       for z in (r,n,u,cont))):
            raise ValueError('Require the original development-only comparison')
        count_match(f['cells'],po)
        reg,initial = (OUT/k for k in ('registration_v2.json','registration.json'))
        previous = exp.read(initial)
        if (previous['parents']!=[exp.artifact(x) for x in files]
                or previous['files'][1]!=exp.artifact(Path(design.__file__))
                or previous['seconds_per_mip']!=SECONDS or previous['arms']!=list(ARMS)):
            raise ValueError('Initial middle-arm solver evidence no longer applies')
        first = exp.sha(initial)
        exp.once(reg,exp.stable(dict(parents=[exp.artifact(x) for x in files],arms=ARMS,
            initial=exp.artifact(initial),
            files=[exp.artifact(x) for x in (Path(__file__),Path(design.__file__),Path(old.__file__),OUT/'protocol.md')],
            seconds_per_mip=SECONDS,original_2pct=True,independent_confirmation=False)))
        binding = exp.sha(reg)
        _,donor,_ = base.register()
        d = base.fit_population(lambda **v:None)
        rows = []
        for i,c in enumerate(f['cells']):
            ident = c['identity']
            hi,h = base.subset(d,ident['held'])
            cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
            bank = base.expanded(cv,raw)
            with np.load(exp.checked(c['actions']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],h['ids'])
                anchor = z['frozen']
            for rec in np.unique(h['recording']):
                ix = h['recording']==rec
                a = anchor[ix]
                data = (cv[ix],ols[ix],bank[ix],*(h[k][ix] for k in ('target','valid','query')),*c['cut'])
                path = PRIVATE/f'cell_v2_{i}_{rec}.json'
                if not path.exists():
                    if args.replay:
                        raise ValueError('Missing private capacity checkpoint')
                    old_path = PRIVATE/f'cell_{i}_{rec}.json'
                    if old_path.exists():
                        resume = exp.read(old_path)
                        if resume['registration']!=first:
                            raise ValueError('Initial capacity checkpoint changed')
                        result_arms = {k:resume['results'][k] for k in ('narrow_relaxed','full_exact')}
                    else:
                        result_arms = {k:result(data,a,k) for k in ('narrow_relaxed','full_exact')}
                    result_arms['narrow_exact'] = historical_narrow(data,a)
                    result_arms['full_relaxed'] = cached_full(i,rec,data,a,u,cont)
                    exp.once(path,exp.stable(dict(registration=binding,results=result_arms)))
                receipt = exp.read(path)
                if receipt['registration']!=binding:
                    raise ValueError('Changed capacity checkpoint')
                row = receipt['results']
                for k,z in row.items():
                    if z['status']=='verified' and not check(z['actions'],data,a,k)['verified']:
                        raise ValueError('Frozen witness did not replay')
                for low,high in ((0,1),(0,2),(1,3),(2,3)):
                    if row[ARMS[low]]['status']=='verified' and row[ARMS[high]]['status'] not in ('verified','unresolved'):
                        raise ValueError('A wider feasible set excluded a verified witness')
                source, = np.unique(h['sites'][ix])
                rows.append(dict(row,source=str(source),rows=int(ix.sum())))
                prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',dict(pid=os.getpid(),
                    cell=i,recording=int(rec),views=len(rows),seconds=round(time.monotonic()-t,3),replay=args.replay))
            print(dict(cell=i,views=len(rows)),flush=True)
        report = summarize(rows)
        if (report['counts']['narrow_exact']!=n['counts']
                or report['counts']['full_relaxed']!=cont['final_uncapped']):
            raise ValueError('Historical corner did not reproduce')
        exp.once(OUT/'readout.json',exp.stable(dict(registration=binding,**report,
            labels_hindsight_only=True,a_labels_opened=False,independent_confirmation=False,
            original_2pct=True,submission_ready=False)))
        print(report,flush=True)


if __name__=='__main__':
    main()
