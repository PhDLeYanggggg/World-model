"""Generate outcome-aware analytic teachers inside the fitting role only."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_lead_risk as parent
from scripts import audit_m3w_cv_uncapped_dual_certificate as cert

base,prior = parent.base,parent.prior
np,read,sha,artifact,once,stable,checked = (parent.np,parent.read,parent.sha,parent.artifact,
                                           parent.once,parent.stable,parent.checked)
OUT = parent.OUT.parent/'moving_cv_slate_supervision_v1'
PRIVATE = parent.PRIVATE.parent/OUT.name
POS = cert.audit.model.POSITIVE_PP


def analytic(history):
    bank = prior.study.motion.bank(history)
    cv,ols = bank[:,0],bank[:,1]
    return cv,ols,base.expanded(cv,bank[:,1:])


def fit_role(d,old):
    roles = [set(old[k]) for k in ('train','calibration','held')]
    if any(len(k)!=3 for k in roles) or len(set.union(*roles))!=9:
        raise ValueError('Teacher role overlaps evaluation')
    return base.subset(d,old['train'])[1]


def teacher(c,o,b,y,v,q,cut):
    m = cert.audit.model
    design = m.coefficients(c,o,b,y,v,q,cut[0])
    if not design['easy'].any() or design['ref']<=0.:
        return dict(status='undefined_support',actions=None)
    cap = int(design['counts'].max())
    args = (c,o,b,y,v,q,*cut)
    z = cert.certify(design,args,cap)
    if z['status']=='unresolved':
        a,s = m.solve(design,cap,seconds=2.)
        if a is not None:
            picks = np.column_stack((design['row'][a],design['arm'][a])).tolist()
            result = cert.audit.verify(picks,*args,cap)
            if result['verified']:
                return dict(status='verified',actions=picks,edits=result['edits'],gain_pp=result['gain_pp'])
        return dict(status=s['status'],actions=None)
    if z['witness'] is None:
        return dict(status=z['status'],actions=None)
    return dict(status='verified',**z['witness'])


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'teacher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t = time.monotonic()
        parent_fp,parent_rp = parent.OUT/'freeze.json',parent.OUT/'readout.json'
        p,r = read(parent_fp),read(parent_rp)
        if (r['freeze']!=sha(parent_fp) or not p['before_held_scoring'] or r['a_labels_opened']
                or r['independent_confirmation']):
            raise ValueError('Need the original fitting-source experiment')
        reg = OUT/'teacher_registration.json'
        once(reg,stable(dict(parent=artifact(parent_fp),parent_readout=artifact(parent_rp),
            files=[artifact(k) for k in (Path(__file__),Path(cert.__file__),OUT/'protocol.md')],
            original_2pct=True,independent_confirmation=False,analytic_only=True,
            no_calibration_or_held_labels_in_teacher=True)))
        _,_,_ = base.register()
        d = base.fit_population(lambda **kw:None)
        results = []
        for fold in range(3):
            cells = [c for c in p['cells'] if c['identity']['fold']==fold]
            c0 = cells[0]
            ident = c0['identity']
            if (len(cells)!=3 or any(c['cut']!=c0['cut'] or c['identity']['train']!=ident['train'] for c in cells)):
                raise ValueError('Teacher must be deterministic across seeds within fold')
            tr = fit_role(d,ident)
            cv,ols,bank = analytic(tr['history'])
            arrays = dict(ids=tr['ids'],action=np.full(len(cv),-1,dtype=np.int8),
                          verified=np.zeros(len(cv),dtype=bool))
            rec = []
            for j in np.unique(tr['recording']):
                ix = tr['recording']==j
                c,o,b = cv[ix],ols[ix],bank[ix]
                y,v,q = (tr[k][ix] for k in ('target','valid','query'))
                path = PRIVATE/f'teacher_{fold}_{j}.json'
                binding = dict(registration=sha(reg),fold=fold,recording=int(j))
                if not path.exists():
                    z = teacher(c,o,b,y,v,q,c0['cut'])
                    once(path,stable(dict(binding=binding,result=z)))
                z = read(path)
                if z['binding']!=binding:
                    raise ValueError('Saved teacher binding changed')
                a = z['result']
                if a['status']=='verified':
                    cap = np.unique(q,return_counts=True)[1].max()
                    score = cert.audit.verify(a['actions'],c,o,b,y,v,q,*c0['cut'],int(cap))
                    if not score['verified'] or score['edits']!=a['edits']:
                        raise ValueError('Teacher failed original scorer')
                    ix0,arms = np.asarray(a['actions']).T
                    arrays['action'][np.flatnonzero(ix)[ix0]] = arms
                    arrays['verified'][ix] = True
                rec.append(dict(status=a['status'],edits=a.get('edits',0)))
                info = dict(pid=os.getpid(),fold=fold,recording=int(j),seconds=round(time.monotonic()-t,3))
                prior.donor.prior.prior.source.atomic(PRIVATE/'teacher_heartbeat.json',info)
            packed = prior.study.parent.parent.save_actions(PRIVATE/f'teacher_{fold}.npz',arrays)
            row = dict(fold=fold,train=ident['train'],recordings=rec,teacher=packed,
                       selected=int((arrays['action']>=0).sum()),verified=int(sum(x['status']=='verified' for x in rec)))
            results.append(row)
            print({k:row[k] for k in ('fold','selected','verified')},flush=True)
        if any(z['verified']==0 for z in results):
            raise ValueError('At least one verified training record per fold is required')
        once(OUT/'teacher_freeze.json',stable(dict(registration=sha(reg),cells=results,
            analytic_only=True,original_2pct=True,independent_confirmation=False)))


if __name__=='__main__':
    main()
