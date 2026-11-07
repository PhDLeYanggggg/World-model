"""Posthoc diagnostic ceiling for assignment at fixed rows and amplitudes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_incremental_amplitude as study

run, np = study.run,study.np
PUBLIC = study.PUBLIC


def main():
    path = PUBLIC/'analysis.json'
    a = run.read(path)
    if run.read(PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(path):
        raise ValueError('Require exact frozen development readout')
    d,refs = run.population(),run.register()
    groups = study.parent.prior.blocks(d,run.old.prior.arrays(('frames',))['frames'])
    labels = run.old.prior.arrays(('target_eval','valid'))
    target,valid = labels['target_eval']-d['origin'][:,None],labels['valid']
    results = []
    for c in run.read(PUBLIC/'freeze.json')['cells']:
        seed = c['seed']
        cv,bank = run.candidates(d,refs,seed)
        width = bank.shape[1]
        with np.load(run.checked(c['cache']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],d['global_ids'])
            orig,inc = z['choices'][[0,3]]
        for r,(ix,gs) in groups.items():
            source = str(d['sites'][ix[0]])
            count = positive = 0
            gain = known_gain = 0.
            for g in gs:
                rows = ix[g[inc[ix[g]]>=0]]
                if len(rows)!=2:
                    continue
                level = inc[rows]//width
                if level[0]==level[1] or (level[::-1]<orig[rows]//width).any():
                    continue
                count += 1
                changed = inc[rows]%width+width*level[::-1]
                before,after = (study.bank_model.selected(bank[rows],cv[rows],ch)
                                for ch in (inc[rows],changed))
                m = run.parent.readout.comparison_masses(before,after,cv[rows],target[rows],
                                                        valid[rows],1./d['crowd'][rows])
                lower = m['known_delta']-m['unknown_radius']
                known_gain += max(0.,m['known_delta'])
                if lower>0:
                    positive += 1
                    gain += lower
            results.append(dict(seed=seed,source=source,recording=r,
                exchangeable=count,lower_positive_swaps=positive,lower_gain=gain,known_gain=known_gain))
    expected = a['exchangeable_queries']
    if sum(r['exchangeable'] for r in results)!=expected or len(results)!=102:
        raise ValueError('Mismatched exchangeable query roster')
    denominator = {(r['seed'],r['recording']):r['comparisons']['original_vs_cv']['reference'] for r in a['records']}
    sources = sorted({r['source'] for r in results})
    blocks = [[r for r in results if r['source']==s and r['seed']==k]
              for s in sources for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(len(sources),size=(3000,len(sources)))
    def reduce(key):
        data = [100*sum(r[key] for r in g)/sum(denominator[r['seed'],r['recording']] for r in g)
                for g in blocks]
        v = np.asarray(data).reshape(len(sources),3).mean(1)
        return dict(point=float(v.mean()),interval95=np.quantile(v[draw].mean(1),[.025,.975]).tolist())
    out = dict(source=run.artifact(path),exchangeable_queries=expected,
        lower_positive_swaps=sum(r['lower_positive_swaps'] for r in results),
        attainable_conservative_utility=reduce('lower_gain'),
        oracle_known_utility=reduce('known_gain'),
        source_bootstraps=3000,posthoc_diagnostic_only=True,
        oracle_choices_exported=False,independent_outcomes_opened=False)
    run.once(PUBLIC/'ceiling.json',out)
    print(out)


if __name__=='__main__':
    main()
