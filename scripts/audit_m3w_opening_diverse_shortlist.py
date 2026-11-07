"""Compare row/family-diverse recall at the same eight actions per query."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opening_pair_rank as study

run,np,anchor,model = study.run,study.np,study.anchor,study.model


def ordered(values,legal,width):
    ix = np.flatnonzero(legal.ravel())
    raw = ix[np.argsort(-values.ravel()[ix],kind='stable')]
    result = {'flat':raw[:8]}
    for key,group in (('row',lambda i:i//width),('family',lambda i:i%width)):
        seen,first,rest = set(),[],[]
        for i in raw:
            g = group(i)
            if g not in seen:
                seen.add(g)
                first.append(i)
            else:
                rest.append(i)
        result[key] = (first+rest)[:8]
    if len({len(v) for v in result.values()})!=1:
        raise ValueError('All recall comparisons must have identical candidate budgets')
    return result


def main():
    pub = study.PUBLIC
    src = pub/'analysis.json'
    saved = run.read(src)
    if run.read(pub/'replay_execution.json')['artifact']['sha256']!=run.sha(src):
        raise ValueError('Require exact scoring replay')
    d,refs = run.population(),run.register()
    lab = run.old.prior.arrays(('target_eval','valid','frames'))
    records = study.prior.blocks(d,lab['frames'])
    y,valid = lab['target_eval']-d['origin'][:,None],lab['valid']
    cuts = run.cuts_by_source()
    controls = {c['seed']:c['selection'] for c in run.read(anchor.SIZE.PUBLIC/'freeze.json')['cells']}
    old = {(c['seed'],f['identity']['held']):f for c in run.read(anchor.PUBLIC/'freeze.json')['cells']
           for f in c['folds']}
    parent = {c['seed']:c for c in run.read(study.timing.PAST.PUBLIC/'freeze.json')['cells']}
    data = []
    model.torch.set_num_threads(4)
    model.torch.set_num_interop_threads(1)
    for cell in run.read(pub/'freeze.json')['cells']:
        seed = cell['seed']
        cv,bank = run.candidates(d,refs,seed)
        dc,_ = study.timing.parents(d,parent[seed],controls)
        folds = {f['identity']['held']:f for f in cell['folds']}
        for ix,g in records.values():
            if (dc[ix]>=0).any():
                continue
            s = str(d['sites'][ix[0]])
            c,b = cv[ix],bank[ix]
            raw,_ = run.model.descriptors(d['history'][ix],b,c,d['crowd'][ix])
            x = model.features(raw)
            with np.load(run.checked(old[seed,s]['cache']),allow_pickle=False) as a:
                scores = [anchor.model.predict(raw,a['coefficients'])[0]]
            for cp in folds[s]['checkpoints']:
                net = model.network(x.shape[-1],seed)
                state = model.torch.load(run.checked(cp),map_location='cpu',weights_only=True)
                net.load_state_dict(state['network'])
                scores.append(model.predict(net.eval(),x))
            labels,atoms,_ = anchor.model.labels(c,b,y[ix],valid[ix],cuts[s]['easy'])
            good = labels[...,0] & (atoms[...,5]>0.)
            legal = np.linalg.norm(b-c[:,None],axis=-1).max(2)>0.
            q = [j for j in g if len(j)>1 and legal[j].any()]
            row = {}
            for scope,window in (('first',q[:1]),('first_two',q[:2]),('any',q)):
                row[scope] = {}
                for name,v in zip(('old','pointwise','pair'),scores):
                    hit = {a:False for a in ('flat','row','family')}
                    for j in window:
                        for kind,take in ordered(v[j],legal[j],v.shape[1]).items():
                            hit[kind] |= bool(good[j].ravel()[take].any())
                    row[scope][name] = hit
            data.append(row)
    if len(data)!=18:
        raise ValueError('Require all original zero-direct views')
    result = {k:{a:{kind:sum(r[k][a][kind] for r in data) for kind in ('flat','row','family')}
                 for a in ('old','pointwise','pair')} for k in ('first','first_two','any')}
    for k,v in result.items():
        if v['old']['flat']!=saved['old_recall'][k]['joint_top8'] or any(
                v[a]['flat']!=saved['recall'][k][f'{a}_top8'] for a in ('pointwise','pair')):
            raise ValueError('Flat top-eight control differs')
    doc = run.stable(dict(views=18,actions_per_query=8,recall=result,analysis_sha256=run.sha(src),
        label_aware_scoring_only=True,policies_retuned=False,independent_outcomes_opened=False))
    run.once(pub/'diverse_shortlist.json',doc)
    print(doc)


if __name__=='__main__':
    main()
