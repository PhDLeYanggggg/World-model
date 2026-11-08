"""Decompose the replayed held-source decision into forecast amplitudes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_ols_bounded_gain as study


def match_amplitudes(scores,legal,queries,reference,number_of_amplitudes):
    np = study.np
    reference = np.asarray(reference)
    take = np.full(len(scores),-1,dtype=np.int16)
    for q in queries:
        amplitudes = sorted(reference[q][reference[q] >= 0]%number_of_amplitudes)
        if len(amplitudes) > 2:
            raise ValueError('Expected at most two fixed amplitude slots per query')
        slots = []
        for a in amplitudes:
            v = np.where(legal[q,a::number_of_amplitudes],scores[q,a::number_of_amplitudes],-np.inf)
            choice = v.argmax(1)
            value = v[np.arange(len(q)),choice]
            order = np.argsort(-value,kind='stable')[:2]
            slots.append([(value[i],q[i],number_of_amplitudes*choice[i]+a) for i in order if np.isfinite(value[i])])
        if len(slots) == 1:
            _,i,b = slots[0][0]; take[i] = b
        elif len(slots) == 2:
            choices = [(l,r) for l in slots[0] for r in slots[1] if l[1] != r[1]]
            if not choices:
                raise ValueError('No legal assignment for the frozen amplitudes')
            for _,i,b in max(choices,key=lambda x:x[0][0]+x[1][0]):
                take[i] = b
        np.testing.assert_array_equal(np.sort(take[q][take[q] >= 0]%number_of_amplitudes),amplitudes)
    return take


def matched_control(queries,ols,bank,debit,w,src,cv_mass,g,h,known,direct,bounded):
    np = study.np
    legal = debit > 1e-6
    mass = 100*w/np.bincount(src,weights=w*cv_mass)[src]
    draws = np.random.default_rng(73107).integers(9,size=(3000,9))
    effects,gains,harms = [],[],[]
    n,m = 0,0
    for d_score,b_score in zip(direct,bounded,strict=True):
        c = study.old.model.choose(d_score,legal,queries)
        b = match_amplitudes(b_score,legal,queries,c,len(study.old.model.AMPLITUDES))
        di,bi = np.flatnonzero(c >= 0),np.flatnonzero(b >= 0)
        dp,bp = ols.copy(),ols.copy()
        dp[di],bp[bi] = bank[di,c[di]],bank[bi,b[bi]]
        effect = -np.linalg.norm(dp-bp,axis=2).max(1)
        effect[known] = 0.
        dk,bk = di[known[di]],bi[known[bi]]
        effect[dk] -= g[dk,c[dk]]; effect[bk] += g[bk,b[bk]]
        effects.append(np.bincount(src,weights=mass*effect,minlength=9))
        gains.append(np.bincount(src[bi],weights=mass[bi]*g[bi,b[bi]],minlength=9))
        harms.append(np.bincount(src[bi],weights=mass[bi]*h[bi,b[bi]],minlength=9))
        n += len(bi); m += int((~known[bi]).sum())
    interval = lambda x:study.old.metric.intervals(np.stack(x,1),draws)
    return dict(paired_vs_direct_pp=interval(effects),vs_ols4_pp=interval(gains),positive_harm_pp=interval(harms),
        switches=n,unknown=m,exact_query_amplitudes=True,posthoc_fitting_only=True,changed_policy=False,
        external_access=False,analysis=study.run.artifact(study.PUBLIC/'analysis.json'))


def main():
    np,run = study.np,study.run
    jobs,old = study.register()
    result = study.prev.receipt(study.PUBLIC,'replay')
    d = study.fitting(jobs,lambda **_:None)
    queries = study.old.fitting_layout(jobs[0]['identity'],d)
    _,_,cv,ols,bank,_,debit = study.old.pack(d)
    mass,g,h,known = study.old.metric.targets(cv,ols,bank,d['target'],d['valid'])
    w = np.empty(len(cv)); src = np.unique(d['sites'],return_inverse=True)[1]
    for q in queries:
        w[q] = 1./len(q)
    unit = 100*w/np.bincount(src,weights=w*mass)[src]/(result['sources']*len(result['seeds']))
    with np.load(run.checked(old['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); direct = z['score'][:,study.old.ARMS.index('direct')]
    frozen = study.prev.receipt(study.PUBLIC,'train')
    with np.load(run.checked(frozen['scores']),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],d['ids']); scores = np.concatenate((direct[:,None],z['scores']),axis=1)
    amounts = study.old.model.AMPLITUDES
    result_table = {}
    for arm,scores_by_seed in zip(study.ARMS,scores.transpose(1,0,2,3),strict=True):
        stats = np.zeros((len(amounts),4))
        for control,score in zip(direct,scores_by_seed,strict=True):
            legal = debit > 1e-6
            c = study.old.model.choose(control,legal,queries)
            p = study.old.model.choose(score,legal,queries,counts=[(c[q] >= 0).sum() for q in queries])
            ix = np.flatnonzero(p >= 0)
            for i in range(len(amounts)):
                k = ix[p[ix]%len(amounts) == i]
                stats[i] += (len(k),int((~known[k]).sum()),unit[k]@h[k,p[k]],unit[k]@g[k,p[k]])
        v = result['arms'][arm]
        np.testing.assert_allclose(stats.sum(0),[v['switches'],v['unknown'],v['harm']['point'],v['gain']['point']],atol=1e-10)
        result_table[arm] = [dict(amplitude=a,switches=int(n),unknown=int(m),harm_pp=float(h),gain_pp=float(g))
                             for a,(n,m,h,g) in zip(amounts,stats,strict=True)]
    out = study.PUBLIC/'amplitudes.json'
    run.once(out,run.stable(dict(analysis=run.artifact(study.PUBLIC/'analysis.json'),arms=result_table,
        posthoc_fitting_only=True,changed_policy=False,external_access=False)))
    run.once(study.PUBLIC/'amplitude_matched_control.json',run.stable(matched_control(
        queries,ols,bank,debit,w,src,mass,g,h,known,direct,scores[:,study.ARMS.index('bounded_full')])))


if __name__ == '__main__':
    main()
