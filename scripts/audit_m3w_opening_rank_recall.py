"""Check whether a timing rule could recover a useful action at a fixed rank."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opening_stop_wait as study

run,np,anchor = study.run,study.np,study.anchor
K = (1,3,8)


def examine(c,b,x,scale,y,valid,groups,co,w,easy):
    label,atoms,_ = anchor.model.labels(c,b,y,valid,easy)
    useful = label[...,0] & (atoms[...,5]>0.)
    u,_,_,legal = run.model.decision_costs(x,co,b,c,scale)
    j,m = anchor.model.predict(x,w)
    results = {}
    eligible = [g for g in groups if len(g)>1 and legal[g].any()]
    for scope,frames in (('first',eligible[:1]),('first_two',eligible[:2]),('any',eligible)):
        have = {'bank':any(useful[g].any() for g in frames)}
        for name,val in (('joint',j),('marginal',m),('utility',u/scale[:,None])):
            hits = {k:False for k in K}
            for g in frames:
                ix = np.flatnonzero(legal[g].ravel())
                top = ix[np.argsort(-val[g].ravel()[ix],kind='stable')[:max(K)]]
                selected = useful[g].ravel()[top]
                for k in K:
                    hits[k] |= bool(selected[:k].any())
            have.update({f'{name}_top{k}':v for k,v in hits.items()})
        results[scope] = have
    return results


def main():
    pub = study.PUBLIC
    result = pub/'analysis.json'
    if run.read(pub/'replay_execution.json')['artifact']['sha256']!=run.sha(result):
        raise ValueError('Require completed replay')
    data,refs = run.population(),run.register()
    labs = run.old.prior.arrays(('target_eval','valid','frames'))
    records = study.prior.blocks(data,labs['frames'])
    y,valid = labs['target_eval']-data['origin'][:,None],labs['valid']
    cuts = run.cuts_by_source()
    controls = {c['seed']:c['selection'] for c in run.read(anchor.SIZE.PUBLIC/'freeze.json')['cells']}
    af = {(c['seed'],f['identity']['held']):f for c in run.read(anchor.PUBLIC/'freeze.json')['cells']
          for f in c['folds']}
    sources = [c for c in run.read(study.timing.PAST.PUBLIC/'freeze.json')['cells']]
    out = []
    for cell in sources:
        seed = cell['seed']
        c,b = run.candidates(data,refs,seed)
        direct,co = study.timing.parents(data,cell,controls)
        for ix,g in records.values():
            if (direct[ix]>=0).any():
                continue
            s = str(data['sites'][ix[0]])
            x,z = run.model.descriptors(data['history'][ix],b[ix],c[ix],data['crowd'][ix])
            with np.load(run.checked(af[seed,s]['cache']),allow_pickle=False) as a:
                w = a['coefficients'].copy()
            out.append(examine(c[ix],b[ix],x,z,y[ix],valid[ix],g,co[s],w,cuts[s]['easy']))
    if len(out)!=18:
        raise ValueError('Require the complete zero-direct slice')
    doc = run.stable(dict(views=len(out),counts={k:{n:sum(o[k][n] for o in out)
        for n in out[0][k]} for k in out[0]},label_aware_diagnostic=True,
        no_actions_exported=True,independent_outcomes_opened=False,
        analysis_sha256=run.sha(result),selector_retuned=False))
    prior = run.read(anchor.PUBLIC/'opportunity_diagnostic.json')['categories']
    if (doc['counts']['first']['bank']!=prior['first'] or
            doc['counts']['any']['bank']!=prior['first']+prior['later']):
        raise ValueError('Bank opportunities must agree with original scorer check')
    run.once(pub/'rank_recall.json',doc)
    print(doc)


if __name__=='__main__':
    main()
