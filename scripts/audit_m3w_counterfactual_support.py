"""Audit exact hindsight capacity with only one query replacement per recording."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_counterfactual_support as exp

run, np = exp.run, exp.np
RISK = ('all_excess','easy_excess','easy_signed')


def components(refs, path, y, valid, cut):
    cv = refs[0]
    risk = exp.old.moments(cv,path,y,valid,cut)
    active = (path != cv).any((1,2))
    gain = [exp.control.paired(ref,path,y,valid)[1].sum(1) for ref in refs]
    return np.column_stack((*gain, *[risk[k]*active for k in RISK], risk['easy'] & active)), risk['ref_easy']


def audit(d, bank, anchor, menus, cuts, fired):
    cv,ols = bank[:2]
    y,valid,rec = d['y'],d['valid'],d['rec']
    old,easy = components((cv,ols),anchor,y,valid,cuts[0])
    _, inv = np.unique(rec.astype(str),return_inverse=True)
    total = np.stack([np.bincount(inv,weights=k) for k in old.T],axis=1)
    cap = .02*np.bincount(inv,weights=easy)
    has = lambda x, limit: ((x[..., :2] > 0.).all(-1) & (x[..., 2:4] <= 1e-12).all(-1)
                           & (x[..., 4] <= limit+1e-12) & (x[..., 5] > 0.))
    base = has(total,cap)
    qualified = base.copy()
    support = total[:,5] > 0.
    restored = support.copy()
    best = {}
    counts = Counter()
    for b in menus:
        a = slice(b['lo'],b['hi'])
        n = len(b['path'])
        idx = np.tile(np.arange(a.start,a.stop),n)
        z,_ = components((cv[idx],ols[idx]),b['path'].reshape(-1,*cv.shape[1:]),y[idx],valid[idx],cuts[0])
        group = inv[a.start]
        if not (inv[a] == group).all():
            raise ValueError('Query spans recordings')
        v = z.reshape(n,a.stop-a.start,-1).sum(1) + (total[group]-old[a].sum(0))
        good = has(v,cap[group])
        restored[group] |= bool((v[:,5] > 0.).any())
        counts['options'] += n
        counts['qualified_options'] += int(good.sum())
        counts['support_loss_options'] += int(((total[group,5] > 0.) & (v[:,5] == 0.)).sum())
        for k in ('local','whole','gain','matched_gain'):
            f = fired[k].get(b['rec'])
            if f is not None and f[0] == b['query']:
                i = f[1]
                counts[f'{k}_changed_in_recoverable_query'] += int(good.any())
                counts[f'{k}_picked_qualified_option'] += int(good[i])
        if good.any() and not base[group]:
            utility = np.where(good,v[:,:2].min(1),-np.inf)
            i = int(utility.argmax())
            if group not in best or utility[i] > best[group][0]:
                best[group] = (float(utility[i]),a,b['path'][i])
    p = anchor.copy()
    for g, (_,a,z) in best.items():
        p[a] = z
        qualified[g] = True
    held = exp.control.parent.score(d,cv,ols,p,cuts)
    original = exp.control.parent.score(d,cv,ols,anchor,cuts)
    if (held['dual_positive'] != int(qualified.sum()) or original['dual_positive'] != int(base.sum())
            or held['edits'] != original['edits']):
        raise ValueError('Hindsight assignment did not pass the original scorer exactly')
    return dict(base_dual=int(base.sum()), oracle_dual=int(qualified.sum()),
                newly_qualified=int((qualified & ~base).sum()),
                easy_support_possible=int(restored.sum()), easy_support_original=int(support.sum()), **counts)


def main():
    reg,source,*_ = exp.register()
    fp,rp = (exp.OUT/k for k in ('freeze.json','readout.json'))
    f,r = map(run.read,(fp,rp))
    if (f['registration'] != reg or r['freeze'] != run.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or r['confirmation_access'] or r['independent_confirmation']):
        raise ValueError('Require frozen development predictions')
    data = exp.m.load(source)
    rows = []
    for c in f['cells']:
        d = data[c['held']]['d']
        bank = exp.control.parent.prior.analytic(d)
        with np.load(run.checked(c['paths']),allow_pickle=False) as z:
            anchor = z['anchor']
        with np.load(run.checked(c['spline']),allow_pickle=False) as z:
            p = exp.old.design(d,bank,anchor,z['comoving_spline'])
        menu = exp.blocks(d,bank[0],p,anchor)
        result = audit(d,bank,anchor,menu,c['cut'],c['fired'])
        rows.append(dict(site=c['held'],seed=c['seed'],result=result))
    summary = dict(sum((Counter(z['result']) for z in rows), Counter()))
    run.once(exp.OUT/'opportunity.json', run.stable(dict(parent=run.artifact(rp),code=run.artifact(Path(__file__)),
        rows=rows,summary=summary,label_aware_only=True,original_2pct=True,confirmation_access=False)))
    print(summary,flush=True)


if __name__ == '__main__':
    main()
