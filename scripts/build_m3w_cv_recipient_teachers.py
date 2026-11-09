"""Generate soft recipient targets from verified train-only recording plans."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_minimum_budget as parent
from src.evaluation.m3w_cv_budget_certificates import witness_terms

exp,base,prior,np = parent.exp,parent.base,parent.prior,parent.np
OUT = parent.OUT.parent/'moving_cv_recipient_path_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name


def variants(terms,required,seed):
    floor = required+np.array([1e-8,1e-8,0.,0.,0.,0.])
    n = len(terms)
    rng = np.random.default_rng(seed)
    orders = [np.argsort(terms[:,1],kind='stable'),np.arange(n),np.arange(n)[::-1]]
    orders.extend(rng.permutation(n) for _ in range(6))
    for order in orders:
        total = terms.sum(0)
        keep = np.ones(n,bool)
        for i in order:
            candidate = total-terms[i]
            if (candidate>=floor).all():
                total = candidate
                keep[i] = False
        yield keep


def family(c,o,b,y,v,q,cut,old,new,seed):
    cap = int(np.unique(q,return_counts=True)[1].max())
    unique = {}
    for k,start in enumerate((old,new)):
        terms,required = witness_terms(c,o,b,y,v,q,cut[0],start)
        for action in [start,*(start[m] for m in variants(terms,required,seed+k))]:
            key = action.tobytes()
            if key in unique:
                continue
            proof = parent.parent.old.verify(action,c,o,b,y,v,q,*cut,cap)
            if proof['verified']:
                unique[key] = action
            elif key==start.tobytes():
                raise ValueError('Inherited train plan did not verify')
    if not unique:
        raise ValueError('Expected verified train plans')
    mass = np.zeros(len(c),np.float64)
    for action in unique.values():
        mass[action[:,0]] += 1.
    mass /= len(unique)
    return mass,len(unique),len({tuple(a[:,0]) for a in unique.values()})


def build():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'labels.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = time.monotonic()
        previous = (exp.OUT/'teacher_freeze.json',parent.OUT/'teacher_freeze.json',exp.OUT/'freeze.json')
        orig,minimum,folds = map(exp.read,previous)
        if orig['independent_confirmation'] or not orig['analytic_only'] or not minimum['train_only']:
            raise ValueError('Need analytic training witnesses')
        reg = OUT/'teacher_registration.json'
        exp.once(reg,exp.stable(dict(parents=[exp.artifact(x) for x in previous],files=[exp.artifact(x) for x in
            (Path(__file__),Path(witness_terms.__code__.co_filename),OUT/'protocol.md')],train_only=True)))
        d = base.fit_population(lambda **v:None)
        cells = []
        for fold in range(3):
            cell = next(x for x in folds['cells'] if x['identity']['fold']==fold)
            ident = cell['identity']
            if any(x['cells'][fold]['train']!=ident['train'] for x in (orig,minimum)):
                raise ValueError('Witness roles do not match')
            tr = exp.teacher.fit_role(d,ident)
            c,o,b = exp.teacher.analytic(tr['history'])
            actions = []
            valid = None
            for source in (orig,minimum):
                with np.load(exp.checked(source['cells'][fold]['teacher']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],tr['ids'])
                    if valid is not None:
                        np.testing.assert_array_equal(valid,z['verified'])
                    valid = z['verified']
                    actions.append(z['action'])
            mass = np.zeros(len(c),np.float32)
            counts = []
            for rec in np.unique(tr['recording']):
                ix = np.flatnonzero(tr['recording']==rec)
                if not valid[ix].all():
                    continue
                start_actions = []
                for a in actions:
                    at = np.flatnonzero(a[ix]>=0)
                    start_actions.append(np.column_stack((at,a[ix][at])))
                mass[ix],n,r = family(c[ix],o[ix],b[ix],*(tr[k][ix] for k in ('target','valid','query')),
                    cell['cut'],*start_actions,1949+int(rec))
                counts.append((n,r))
                prior.donor.prior.prior.source.atomic(PRIVATE/'teacher_heartbeat.json',
                    dict(pid=os.getpid(),fold=fold,recording=int(rec),seconds=round(time.monotonic()-start,3)))
            if not counts or not ((mass>0)&(mass<1)).any():
                raise ValueError('No soft recipient supervision in the training role')
            a = prior.study.parent.parent.save_actions(PRIVATE/f'recipient_{fold}.npz',
                dict(ids=tr['ids'],valid=valid,one=actions[1]>=0,soft=mass))
            cells.append(dict(fold=fold,train=ident['train'],labels=a,recordings=len(counts),
                verified_plans=sum(n for n,_ in counts),distinct_recipient_sets=sum(r for _,r in counts),
                ambiguous_rows=int(((mass>0)&(mass<1)).sum())))
            print(cells[-1],flush=True)
        exp.once(OUT/'teacher_freeze.json',exp.stable(dict(registration=exp.sha(reg),cells=cells,train_only=True)))


if __name__=='__main__':
    build()
