"""Posthoc label-aware bounds on the fixed joint proposals, not a policy."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_native_query_critic as study
from src.evaluation.m3w_selection_forecast_readout import errors, policy


def main():
    run, np = study.run, study.np
    result = study.PUBLIC/'analysis.json'
    for phase in ('score', 'replay'):
        assert run.read(study.PUBLIC/f'{phase}_execution.json')['artifact'] == run.artifact(result)
    freeze = study.PUBLIC/'freeze.json'
    assert run.read(study.PUBLIC/'fit_execution.json')['artifact'] == run.artifact(freeze)
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    y, valid = labels['target_eval']-d['origin'][:, None], labels['valid']
    blocks = study.prior.h.previous.prior.blocks(d, labels['frames'])
    cut = run.read(study.prior.REG)['cut']
    views, full_counts = [], None
    for cell in run.read(freeze)['cells']:
        cv, bank = run.candidates(d, refs, cell['seed'])
        ade, _ = errors(cv, y, valid)
        easy = valid.any(1) & (ade > 0.) & (ade <= cut['easy'])
        counts = [int(easy[ix].sum()) for ix, _ in blocks.values()]
        if full_counts is not None:
            assert counts == full_counts
        full_counts = counts
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            old, new = (z['decisions'][k] for k in (0, 3))
        proposed = run.model.selected_paths(old, bank, cv)
        moments = study.critic.action_targets(cv, proposed, y, valid, cut['easy'])
        for rec, (ix, groups) in blocks.items():
            has_proposed = bool(easy[ix][old[ix] >= 0].any())
            witness = False
            for g in groups:
                chosen = g[old[ix[g]] >= 0]
                if len(chosen) == 0:
                    continue
                if not study.critic.accept(study.critic.targets(moments[ix[chosen]], 1.)[None])[0]:
                    continue
                pred = cv[ix].copy()
                pred[chosen] = proposed[ix[chosen]]
                score = policy(pred, cv[ix], cv[ix], y[ix], valid[ix], d['recordings'][ix],
                               easy_cut=cut['easy'], hard_cut=cut['hard'])
                if not score['recordings'][rec]['supported']:
                    raise ValueError('Local witness did not pass original recording scorer')
                witness = True
                break
            views.append((has_proposed, witness, bool((new[ix] >= 0).any()),
                          bool(easy[ix][new[ix] >= 0].any())))
    v = np.asarray(views)
    assert not (v[:, 1] & ~v[:, 0]).any()
    run.once(study.PUBLIC/'proposal_capacity.json', run.stable(dict(
        analysis=run.artifact(result), freeze=run.artifact(freeze),
        result_source='fresh_posthoc_label_aware_capacity_not_policy', views=len(v),
        recordings=len(full_counts), full_reference_easy_rows=sum(full_counts),
        recordings_without_full_easy_reference=sum(n == 0 for n in full_counts),
        views_without_easy_in_any_original_proposal=int((~v[:, 0]).sum()),
        views_with_feasible_single_query_witness=int(v[:, 1].sum()),
        critic_zero_switch_views=int((~v[:, 2]).sum()),
        critic_switches_without_easy_reference_views=int((v[:, 2] & ~v[:, 3]).sum()),
        witness_is_not_an_optimal_or_learned_policy=True, independent_confirmation=False)))


if __name__ == '__main__':
    main()
