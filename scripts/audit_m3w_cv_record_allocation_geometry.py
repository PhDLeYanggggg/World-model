"""Compare neighbor displacement on the identical frozen allocation roster."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_record_allocation as study

np, prior, base = study.np, study.prior, study.base
PAIRS = {'offline_gain_vs_local':('local','offline_gain'),
         'offline_risk_vs_local':('local','offline_risk'),
         'offline_risk_vs_gain':('offline_gain','offline_risk'),
         'online_carry_vs_local':('online_local','online_carry')}


def main():
    read, sha = study.read, study.sha
    path = study.OUT/'freeze.json'
    result = study.OUT/'readout.json'
    bundle, parent = read(path),read(result)
    if (bundle['registration']!=sha(study.OUT/'registration.json') or parent['freeze']!=sha(path)
            or not bundle['before_held_scoring'] or parent['a_labels_opened'] or parent['independent_confirmation']):
        raise ValueError('Require complete fitting-only allocation evidence')
    _, donor, _ = base.register()
    data = base.fit_population(lambda **kw:None)
    groups = sorted(set(data['sites']))
    sums = np.zeros((len(groups),len(base.SEEDS),len(PAIRS),2))
    counts = np.zeros((len(groups),len(base.SEEDS),2),dtype=int)
    names = tuple(dict.fromkeys(k for pair in PAIRS.values() for k in pair))
    comparisons = [tuple(names.index(k) for k in pair) for pair in PAIRS.values()]
    for cell in bundle['cells']:
        ident = cell['identity']
        hi,d = base.subset(data,ident['held'])
        seed = ident['seed']
        cv,_,bank = prior.bank(d['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,data).items()})
        bank = base.expanded(cv,bank)
        with np.load(study.checked(cell['actions']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['ids'])
            paths = [prior.apply(cv,bank,z[k]) for k in names]
        for ix in prior.query_groups(d['query']):
            s = groups.index(d['sites'][ix[0]])
            k = base.SEEDS.index(seed)
            _,v,_,(steps,known,_,_) = prior.geometry.query(d['history'][ix],np.stack([p[ix] for p in paths]),
                                                          d['target'][ix],d['valid'][ix],comparisons)
            if steps:
                counts[s,k,1] += 1
                sums[s,k,:,1] += v[:,1]
            if known:
                counts[s,k,0] += 1
                sums[s,k,:,0] += v[:,0]
        print(dict(pid=os.getpid(),fold=ident['fold'],seed=seed),flush=True)
    weights = counts[:,:,None,:]
    means = np.divide(sums,weights,out=np.full_like(sums,np.nan),where=weights>0.)
    draw = np.random.default_rng(73107).integers(len(groups),size=(3000,len(groups)))
    effects = {k:dict(observed_pixel=prior.geometry.bootstrap(means[:,:,i,0],draw),
                      completion_lower_pixel=prior.geometry.bootstrap(means[:,:,i,1],draw))
               for i,k in enumerate(PAIRS)}
    study.once(study.OUT/'geometry.json',study.stable(dict(parent=study.artifact(result),
        source=study.artifact(Path(__file__)),effects=effects,source_groups=len(groups),seeds=base.SEEDS,
        bootstrap_draws=len(draw),queries_with_observed_neighbor_pairs=int(counts[:,:,0].sum()),
        queries_with_neighbor_pairs=int(counts[:,:,1].sum()),a_labels_opened=False,independent_confirmation=False)))


if __name__=='__main__':
    main()
