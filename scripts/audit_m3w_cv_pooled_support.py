"""Outcome-aware attribution of lost coverage: risk prediction versus gain."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_pooled_coverage as study

np,torch,base,prior,screen = study.np,study.torch,study.base,study.prior,study.screen
read,sha,artifact,once,stable,checked = study.read,study.sha,study.artifact,study.once,study.stable,study.checked
ARMS = ('predicted','true_risk','true_risk_gain')


def counts(pred,truth):
    return dict(predicted=int(pred.sum()),actual=int(truth.sum()),overlap=int((pred&truth).sum()),
                missed=int((~pred&truth).sum()),unsupported_predicted=int((pred&~truth).sum()))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = study.OUT
    score,fp = out/'readout.json',out/'freeze.json'
    r,f = read(score),read(fp)
    if r['freeze']!=sha(fp) or r['a_labels_opened'] or r['independent_confirmation']:
        raise ValueError('Require a complete held fitting-source readout')
    _,donor,_ = base.register()
    d = base.fit_population(lambda **v:None)
    old = read(base.OUT/'freeze.json')['cells']
    rows, q_summary, row_summary, q_total, row_total = [], [], [], 0, 0
    scorer = prior.prior.parent.readout
    for c,b in zip(f['cells'],old,strict=True):
        ident = c['identity']
        if ident!=b['identity'] or b['fits'][0]['identity']!=dict(ident,head='risk'):
            raise ValueError('Parent checkpoint role changed')
        hi,h = base.subset(d,ident['held'])
        seed = ident['seed']
        cv,ols,raw = prior.bank(h['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
        bank = base.expanded(cv,raw)
        width = prior.study.motion.features(h['history'][:1],'prefix')[0].shape[1]
        net = screen.load_head(b['fits'][0],width+25,8)
        g,risk,e,legal = screen.predictions(net,cv,bank,h['history'])
        scale = prior.model.scale(h['history'])
        y,v,q = (h[k] for k in ('target','valid','query'))
        easy_cut, hard_cut = c['cut']
        cost = prior.utility.risk_excess(cv,bank,y,v,easy_cut)
        ade,_ = scorer.errors(cv,y,v)
        easy = v.any(1)&(ade>0.)&(ade<=easy_cut)
        er = np.broadcast_to(easy[:,None],g.shape)
        _,_,gain,_ = prior.utility.candidate_terms(cv,ols,bank,y,v)
        true_score = gain.mean(2)/v.shape[1]
        donors_pred = (legal & (e>=.10) & (risk<=0.).all(2)).any(1)
        donors_true = (legal & er & (cost<=0.).all(2)).any(1)
        groups = prior.query_groups(q)
        row_summary.append(counts(donors_pred,donors_true))
        q_summary.append(counts(np.array([donors_pred[z].any() for z in groups]),
                                np.array([donors_true[z].any() for z in groups])))
        q_total += len(groups)
        row_total += len(y)
        with np.load(checked(c['actions']),allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'],h['ids'])
            pred = a['pool_full']
        actions = {'predicted':pred,'true_risk':np.full_like(pred,-1),'true_risk_gain':np.full_like(pred,-1)}
        for ix in groups:
            for k,s in (('true_risk',g[ix]*scale[ix,None]),('true_risk_gain',true_score[ix])):
                actions[k][ix] = study.pool_query(s,cost[ix],er[ix],legal[ix],len(ix),.10)
        for rec in np.unique(h['recording']):
            ix = h['recording']==rec
            cp,op = cv[ix],ols[ix]
            qq,yy,vv = q[ix],y[ix],v[ix]
            paths = {base.CV:cp,base.OLS:op,**{k:prior.apply(cp,bank[ix],a[ix]) for k,a in actions.items()}}
            z = scorer.score_recording(paths,{},yy,vv,qq,str(rec),easy_cut=easy_cut,hard_cut=hard_cut)
            _,inv,n = np.unique(qq,return_inverse=True,return_counts=True)
            pairs = {f'{k}_vs_{tag}':(b0,k) for k in ARMS for tag,b0 in (('cv',base.CV),('ols4',base.OLS))}
            z['comparisons'] = {k:prior.study.motion.full_masses(paths[a],paths[b0],cp,yy,vv,1./n[inv])
                                for k,(a,b0) in pairs.items()}
            source, = np.unique(h['sites'][ix])
            rows.append(dict(z,source=str(source),seed=seed))
        print(dict(fold=ident['fold'],seed=seed),flush=True)
    summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
    if summary['views']!=72 or summary['risk']['predicted']['switches']!=244:
        raise ValueError('Expected all original held views')
    combine = lambda a:{k:sum(v[k] for v in a) for k in a[0]}
    once(out/'support_attribution.json',stable(dict(parent=artifact(score),code=artifact(Path(__file__)),
        summary=summary,query_donors=combine(q_summary),row_donors=combine(row_summary),query_views=q_total,
        row_views=row_total,hindsight=True,independent_confirmation=False,original_2pct=True)))


if __name__=='__main__':
    main()
