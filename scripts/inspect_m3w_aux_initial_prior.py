"""Post-hoc fitting-metadata hypothesis check, not a registered efficacy test."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_trajectory as run


def constant_bce_excess(p, q):
    if not (0 < p < 1 and 0 <= q <= 1): raise ValueError('Valid Bernoulli probabilities required')
    value = 0.
    if q > 0: value += q*np.log(q/p)
    if q < 1: value += (1-q)*np.log((1-q)/(1-p))
    return float(value)


def main():
    run.registration()
    original = json.loads((run.parent.PUBLIC/'prediction_freeze.json').read_text())
    old_hashes = {r['path']:r for r in original['receipts']}
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text()); rows = []
    for a in freeze['receipts']:
        assert run.artifact(ROOT/a['path']) == a
        doc = json.loads((ROOT/a['path']).read_text()); i = doc['identity']
        if i['arm'] != 'cost_only': continue
        path = (ROOT/doc['original_checkpoint']['path']).parent/'complete.json'
        assert run.artifact(path) == old_hashes[str(path.relative_to(ROOT))]
        old = json.loads(path.read_text()); assert old['identity']['input'] == i['input']
        p, q = old['fit']['prevalence'], i['input']['event_prior']
        rows.append(dict(tag=i['tag'], pair=i['pair'], inherited_easy_prior=p, fitting_cap_prior=q,
            constant_BCE_excess=constant_bce_excess(p, q), original_receipt=run.artifact(path)))
    assert len(rows) == 144
    groups = []
    for pair in ('full', 'motion_only'):
        values = [r for r in rows if r['pair'] == pair]
        groups.append(dict(pair=pair, views=len(values), min_median_max={k:np.quantile([r[k] for r in values], [0,.5,1]).tolist()
            for k in ('inherited_easy_prior', 'fitting_cap_prior', 'constant_BCE_excess')}))
    run.immutable_json(run.PUBLIC/'initial_prior_posthoc.json', dict(
        status='post_hoc_fitting_metadata_hypothesis_not_preregistered_efficacy', rows=rows, groups=groups,
        formula='KL(Bernoulli(fitting_cap_prior)||Bernoulli(inherited_easy_prior))',
        weighting='original_fitting_equal_locality_weights_conditioned_on_known_positive_envelope',
        not_equal_to_random_minibatch_BCE=True, causal_role_proved=False,
        repaired_model_trained=False, independent_role_access=False))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__': main()
