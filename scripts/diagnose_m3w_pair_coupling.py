"""Post-hoc observed-label ceiling, never an inference input or deployed policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_pair_coupling as run

np, m = run.np, run.model


def oracle_choices(g, y, valid, ids):
    i, j, _, _, di, dj, _ = g
    e, mask = m.observed_residual(g, y, valid)
    unary = np.zeros(len(ids))
    np.add.at(unary, i, ((2*e*di+di**2).sum(-1)*mask).sum(1))
    np.add.at(unary, j, ((-2*e*dj+dj**2).sum(-1)*mask).sum(1))
    mixed = (-2*(di*dj).sum(-1)*mask).sum(1)
    return m.choose(unary, i, j, mixed, ids)


def main():
    run.register()
    parent = run.read(run.PUBLIC/'analysis.json')
    data = run.arrays(('global_ids', 'history', 'origin', 'queries', 'sites'))
    labels = run.arrays(('target_eval', 'valid'))
    results = {}
    for ref, cache in zip(run.read(run.PARENT/'analysis.json')['forecasts'], run.read(run.PUBLIC/'choice_freeze.json')['banks']):
        ix, sites, history, ids, floor, neural, subsets = run.design(data, ref)
        with np.load(run.checked(cache), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ix'], ix); choices = z['choices']
        for q in subsets:
            name = sites[q[0]]; row = results.setdefault(name, dict(queries=0, supported=0, totals=np.zeros(4)))
            row['queries'] += 1
            g = m.geometry(history[q], floor[q], neural[q], ids[q])
            y, valid = (labels[k][ix[q]] for k in ('target_eval', 'valid'))
            if not (valid[g[0]] & valid[g[1]]).any(): continue
            choices_oracle = oracle_choices(g, y, valid, ids[q])
            scores = [m.error(g, y, valid, c) for c in (*choices[q].T, *choices_oracle)]
            score = np.array([v[0]/v[2] for v in scores])
            if score[-1] > score[:-1].min()+1e-10: raise AssertionError('Oracle is not optimal')
            row['totals'] += score; row['supported'] += 1
    rows = []
    for before in parent['sources']:
        r = results[before['source']]; value = r['totals']/r['supported']
        np.testing.assert_allclose(value[:2], before['pair_squared_available'], rtol=1e-13, atol=1e-13)
        rows.append(dict(source=before['source'], queries=r['queries'], supported=r['supported'], scores=value.tolist()))
    mean = np.mean([r['scores'] for r in rows], axis=0)
    result = run.stable(dict(result_source='fresh_posthoc_label_aware_diagnostic',
        parents=[run.artifact(run.PUBLIC/name) for name in ('analysis.json', 'choice_freeze.json')],
        columns=['learned_additive', 'learned_coupled', 'label_aware_additive', 'label_aware_coupled'],
        source_macro_pair_squared=mean.tolist(), oracle_coupling_gain=float(mean[2]-mean[3]),
        learned_to_oracle_coupled_gap=float(mean[1]-mean[3]), sources=rows,
        oracle_actions_saved=False, model_or_metric_changed=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    run.once(run.PUBLIC/'oracle_diagnostic.json', result)
    print({k: result[k] for k in ('source_macro_pair_squared', 'oracle_coupling_gain', 'learned_to_oracle_coupled_gap')})


if __name__ == '__main__':
    main()
