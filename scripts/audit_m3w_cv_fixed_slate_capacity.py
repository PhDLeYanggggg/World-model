"""Hindsight capacity of the six fixed-count slates, never a deployable policy."""
from collections import Counter
from pathlib import Path
import sys

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_slate_supervision as exp

np,slate,base,prior = exp.np,exp.slate,exp.base,exp.prior
verify = exp.teacher.cert.audit.verify
POSITIVE_PP = exp.teacher.POS


def solve(blocks,ref,offset,cap):
    if not blocks or ref<=0. or not any((b[:,5]>0).any() for b in blocks):
        return None,'undefined_support'
    sizes = [len(b) for b in blocks]
    x = np.concatenate(blocks)
    n = len(x)
    groups = coo_array((np.ones(n),(np.repeat(np.arange(len(sizes)),sizes),np.arange(n))),
                       shape=(len(sizes),n+1))
    units = np.abs(x[:,2:5]).max(0).clip(1.)
    hard = csc_array(np.column_stack((x[:,2:6].T / np.r_[units,1.][:,None],np.zeros(4))))
    utility = csc_array(np.column_stack((-100.*x[:,:2].T/ref,np.ones(2))))
    lim = len(sizes)
    result = milp(np.r_[np.zeros(n),-1.],integrality=np.r_[np.ones(n),0.],
        bounds=Bounds(np.r_[np.zeros(n),-np.inf],np.r_[np.ones(n),np.inf]),
        constraints=LinearConstraint(vstack((groups,hard,utility),format='csc'),
            np.r_[np.ones(lim),-np.inf,-np.inf,-np.inf,1.,-np.inf,-np.inf],
            np.r_[np.ones(lim),0.,0.,cap/units[2],np.inf,0.,100.*offset/ref]),
        options={'time_limit':3.,'mip_rel_gap':0.})
    if result.status==2:
        return None,'risk_infeasible'
    upper = getattr(result,'mip_dual_bound',None)
    if upper is not None and np.isfinite(upper) and -float(upper)<-POSITIVE_PP:
        return None,'nonpositive_bound'
    if result.x is None:
        return None,'unresolved'
    selected = [int(np.argmax(b)) for b in np.split(result.x[:-1],np.cumsum(sizes)[:-1])]
    return selected,'unresolved'


def recording(c,o,b,h,anchor,cut):
    y,v,q = (h[k] for k in ('target','valid','query'))
    w = exp.weight(q)
    _,ref,offset,cap = slate.contributions(c,o,c,y,v,w,cut[0])
    choices,blocks = [],[]
    for g in prior.query_groups(q):
        k = int((anchor[g]>=0).sum())
        if not k:
            continue
        options = list({a.tobytes():a for a in [anchor[g],*slate.menu(c[g],o[g],b[g],k)]}.values())
        choices.append((g,options))
        blocks.append(np.stack([slate.contributions(c[g],o[g],slate.path_for(c[g],b[g],a),
                                y[g],v[g],w[g],cut[0])[0] for a in options]))
    selected,status = solve(blocks,ref,offset,cap)
    if selected is not None:
        action = np.full(len(c),-1,dtype=np.int8)
        for (g,options),j in zip(choices,selected,strict=True):
            action[g] = options[j]
        at = np.flatnonzero(action>=0)
        if len(at)!=int((anchor>=0).sum()):
            raise ValueError('Diagnostic changed the frozen edit count')
        proof = verify(np.column_stack((at,action[at])),c,o,b,y,v,q,*cut,int(np.unique(q,return_counts=True)[1].max()))
        if proof['verified']:
            return 'verified'
    return status


def main():
    f,r = [exp.OUT/k for k in ('freeze.json','readout.json')]
    freeze,result = exp.read(f),exp.read(r)
    if result['freeze']!=exp.sha(f) or result['a_labels_opened'] or result['independent_confirmation']:
        raise ValueError('Require original development-only frozen results')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **kw:None)
    counts,by_cell = Counter(),[]
    for cell in freeze['cells']:
        ident = cell['identity']
        hi,h = base.subset(d,ident['held'])
        c,o,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,ident['seed'],d).items()})
        b = base.expanded(c,raw)
        with np.load(exp.checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],h['ids'])
            anchor = z['frozen']
        local = Counter()
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            local[recording(c[ix],o[ix],b[ix],{k:h[k][ix] for k in ('target','valid','query')},
                            anchor[ix],cell['cut'])] += 1
        counts.update(local)
        by_cell.append(dict(fold=ident['fold'],seed=ident['seed'],counts=dict(local)))
        print(by_cell[-1],flush=True)
    if sum(counts.values())!=72:
        raise ValueError('Missing held recording/seed views')
    exp.once(exp.OUT/'fixed_menu_capacity.json',exp.stable(dict(parent=exp.artifact(r),
        code=exp.artifact(Path(__file__)),counts=dict(counts),by_cell=by_cell,labels_hindsight_only=True,
        a_labels_opened=False,independent_confirmation=False,original_2pct=True)))


if __name__=='__main__':
    main()
