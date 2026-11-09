"""Recompute full-bank count bounds and build training-only teacher plans."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_budget_menu_factorial as parent
from src.evaluation import m3w_cv_minimum_budget as solver

exp,base,prior,np = parent.exp,parent.base,parent.prior,parent.np
teacher = exp.teacher
OUT = parent.OUT.parent/'moving_cv_minimum_budget_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
FRACTIONS = (.05,.10,.25,.50,.75)
SECONDS = 4.


def plan(c,o,b,y,v,q,cut,known):
    design = parent.old.verify
    coeff = parent.design.coefficients(c,o,b,y,v,q,cut[0])
    limit = int(np.unique(q,return_counts=True)[1].max())
    return solver.verified_plan(coeff,known,lambda a:design(a,c,o,b,y,v,q,*cut,limit),SECONDS)


def cached(path,identity,make,replay):
    if not path.exists():
        if replay:
            raise ValueError('Missing recording solve checkpoint')
        exp.once(path,exp.stable(dict(identity=identity,result=make())))
    x = exp.read(path)
    if x['identity']!=identity:
        raise ValueError('Recording checkpoint identity changed')
    return x['result']


def verify(z,c,o,b,y,v,q,cut):
    a = z.get('incumbent')
    if a is not None:
        proof = parent.old.verify(a['actions'],c,o,b,y,v,q,*cut,int(np.unique(q,return_counts=True)[1].max()))
        if not proof['verified'] or proof['edits']!=a['edits']:
            raise ValueError('Minimum-budget incumbent did not replay')
    return a


def training(d,folds,teacher_freeze,reg,replay,beat):
    packed = []
    for fold,cell in enumerate(folds):
        ident = cell['identity']
        tr = teacher.fit_role(d,ident)
        cv,ols,b = teacher.analytic(tr['history'])
        with np.load(exp.checked(teacher_freeze['cells'][fold]['teacher']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],tr['ids'])
            old,verified = z['action'],z['verified']
        selected = np.full(len(old),-1,dtype=np.int8)
        valid_teacher = np.zeros(len(old),dtype=bool)
        count,before,after = 0,0,0
        for rec in np.unique(tr['recording']):
            ix = np.flatnonzero(tr['recording']==rec)
            if not verified[ix].all():
                continue
            g = np.flatnonzero(old[ix]>=0)
            args = (cv[ix],ols[ix],b[ix],*(tr[k][ix] for k in ('target','valid','query')),cell['cut'])
            z = cached(PRIVATE/f'train_{fold}_{rec}.json',dict(registration=reg,fold=fold,recording=int(rec)),
                       lambda:plan(*args,np.column_stack((g,old[ix][g]))),replay)
            a = verify(z,*args)
            if a is None:
                raise ValueError('Lost a verified training plan')
            r,j = np.asarray(a['actions']).T
            selected[ix[r]] = j
            valid_teacher[ix] = True
            count += 1
            before += len(g)
            after += len(r)
            beat(phase='teacher',fold=fold,recording=int(rec),verified=count)
        artifact = prior.study.parent.parent.save_actions(PRIVATE/f'teacher_{fold}.npz',
            dict(ids=tr['ids'],action=selected,verified=valid_teacher))
        packed.append(dict(fold=fold,train=ident['train'],teacher=artifact,recordings=count,old_edits=before,edits=after))
        print(packed[-1],flush=True)
    exp.once(OUT/'teacher_freeze.json',exp.stable(dict(registration=reg,cells=packed,train_only=True)))


def diagnostic(d,donor,cells,reg,replay,beat):
    rows = []
    for i,c in enumerate(cells):
        ident = c['identity']
        hi,h = base.subset(d,ident['held'])
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        b = base.expanded(cv,raw)
        for r in np.unique(h['recording']):
            ix = h['recording']==r
            receipt = exp.read(parent.PRIVATE/f'nested_{i}_{r}.json')
            if receipt['parent']!=exp.sha(parent.OUT/'readout.json'):
                raise ValueError('Changed full-bank checkpoint')
            z = receipt['results']['full_relaxed']
            a = None
            bounds = dict(lower=None)
            if z['status']=='verified':
                args = (cv[ix],ols[ix],b[ix],*(h[k][ix] for k in ('target','valid','query')),c['cut'])
                bounds = cached(PRIVATE/f'held_{i}_{r}.json',dict(registration=reg,cell=i,recording=int(r)),
                                lambda:plan(*args,z['actions']),replay)
                a = verify(bounds,*args)
            site, = np.unique(h['sites'][ix])
            rows.append(dict(source=str(site),n=int(ix.sum()),original=z['status'],lower=bounds['lower'],
                             upper=a['edits'] if a is not None else None))
            beat(phase='held_budget',cell=i,recording=int(r),views=len(rows))
        print(dict(cell=i,views=len(rows)),flush=True)
    eligible = [r for r in rows if r['original']=='verified']
    if (len(eligible),len(rows))!=(53,72):
        raise ValueError('Unexpected full-bank roster')
    table = {}
    for f in FRACTIONS:
        counts = Counter()
        for r in rows:
            cap = int(f*r['n'])
            k = ('undefined_support' if r['original']=='undefined_support' else
                 'verified' if r['upper'] is not None and r['upper']<=cap else
                 'excluded' if r['original']!='verified' or r['lower']>cap else 'unresolved')
            counts[k] += 1
        table[str(f)] = dict(counts)
    out = dict(registration=reg,views=len(rows),capacity=table,
        median_lower_percent=float(np.median([100*r['lower']/r['n'] for r in eligible])),
        median_upper_percent=float(np.median([100*r['upper']/r['n'] for r in eligible])),
        exact_count=sum(r['lower']==r['upper'] for r in eligible),prior_verified=len(eligible),
        independent_confirmation=False,a_labels_opened=False,original_2pct=True)
    exp.once(OUT/'capacity.json',exp.stable(out))
    print(out,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('teacher','capacity'))
    p.add_argument('--replay',action='store_true')
    a = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/f'{a.part}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.monotonic()
        def beat(**v):
            prior.donor.prior.prior.source.atomic(PRIVATE/f'{a.part}_heartbeat.json',
                dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v))
        files = [exp.OUT/'freeze.json',exp.OUT/'teacher_freeze.json',parent.OUT/'nested_verification.json']
        p,t,b = map(exp.read,files)
        if (b['a_labels_opened'] or b['independent_confirmation'] or not b['original_2pct']
                or t['independent_confirmation'] or not t['analytic_only']):
            raise ValueError('Need frozen development-only parents')
        r = OUT/'registration.json'
        exp.once(r,exp.stable(dict(parents=[exp.artifact(f) for f in files],files=[exp.artifact(f) for f in
            (Path(__file__),Path(solver.__file__),OUT/'protocol.md')],seconds_per_mip=SECONDS,fractions=FRACTIONS)))
        _,donor,_ = base.register()
        d = base.fit_population(beat)
        if a.part=='teacher':
            training(d,[next(c for c in p['cells'] if c['identity']['fold']==k) for k in range(3)],t,exp.sha(r),a.replay,beat)
        else:
            diagnostic(d,donor,p['cells'],exp.sha(r),a.replay,beat)
        beat(phase='complete',replay=a.replay)


if __name__=='__main__':
    main()
