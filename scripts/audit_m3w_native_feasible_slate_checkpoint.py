"""Replay trained rankers and measure the fixed-mask feasibility ceiling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_native_feasible_slate as study


def main():
    run, np = study.run, study.np
    transport, support, cut = study.register()
    freeze = study.PUBLIC/'freeze.json'
    if run.artifact(freeze) != run.read(study.PUBLIC/'fit_execution.json')['artifact']:
        raise ValueError('Require sealed fits')
    d, refs = run.population(), run.register()
    truth = run.old.prior.arrays(('target_eval', 'valid'))
    target, valid = truth['target_eval']-d['origin'][:, None], truth['valid']
    counts = np.zeros(8, dtype=int)
    for cell, tc, sc in zip(run.read(freeze)['cells'], transport['cells'], support['cells'], strict=True):
        seed = cell['seed']
        assert seed == tc['seed'] == sc['seed']
        cv, bank = run.candidates(d, refs, seed)
        x, scale = run.model.descriptors(d['history'], bank, cv, d['crowd'])
        z = study.prev.model.features(d['history'], bank, cv, d['crowd'], d['groups'])
        with np.load(run.checked(tc['cache']), allow_pickle=False) as f:
            original = dict(zip(f['sources'], f['excluded_source_coefficients']))
        with np.load(run.checked(sc['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            active = f['choices'][5] >= 0
        with np.load(run.checked(cell['decisions']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            expected, anchor, prob = (f[k] for k in ('choices', 'anchor', 'p_top'))
        checkpoint = study.joblib.load(run.checked(cell['checkpoint']))
        estimators = dict(zip(checkpoint['names'], checkpoint['estimators'], strict=True))
        score_at = dict(zip(anchor, prob, strict=True))
        actions = np.stack([study.crit.action_targets(cv, bank[:, k], target, valid, cut['easy'])
                            for k in range(bank.shape[1])], 1)
        for q in d['groups']:
            name = d['sites'][q[0]]
            slate, feat, legal = study.slate_for_query(q, d, bank, cv, x, scale, z, original[name])
            p = estimators[name].predict_proba(feat)[:, 1]
            first, gated = study.model.choose(p, legal, bool(active[q].any()))
            np.testing.assert_array_equal(slate[first], expected[0, q])
            np.testing.assert_array_equal(slate[gated] if gated is not None else -1, expected[1, q])
            np.testing.assert_equal(p[first], score_at[q[0]])
            outcome = study.option_targets(actions, q, slate)
            counts += (1, len(slate), outcome.any(), gated is not None,
                       gated is not None and legal.sum() > 1,
                       gated is not None and (outcome & legal).any(),
                       gated is not None and outcome[gated],
                       gated is None and (outcome & legal).any())
    fields = ('seed_queries', 'options', 'any_feasible_unrestricted', 'frozen_gate_active_queries',
              'active_queries_with_multiple_eligible_options', 'active_queries_with_any_feasible_option',
              'active_queries_with_selected_feasible_option',
              'inactive_queries_with_feasible_gate_eligible_option')
    run.once(study.PUBLIC/'checkpoint_audit.json', run.stable(dict(
        freeze=run.artifact(freeze), counts=dict(zip(fields, counts.tolist())), prediction_replay_exact=True,
        diagnostics='posthoc_label_aware_not_operating_policy', external_data_access=False)))


if __name__ == '__main__':
    main()
