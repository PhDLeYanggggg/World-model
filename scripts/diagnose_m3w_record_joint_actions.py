"""Label-free decision replay and attribution of the frozen budget bottleneck."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_record_joint_actions as run

np, m = run.np, run.model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', action='store_true'); args = p.parse_args()
    start = time.monotonic(); refs = run.register(); data = run.population()
    h, sites, rec, crowd = (data[k] for k in ('history', 'sites', 'recordings', 'crowd'))
    results = []
    for f in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = f['seed']; floor, paths = run.candidates(data, refs, seed)
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], data['global_ids'])
            names, coeff, frozen = z['names'], z['coefficients'], z['choices']
        u = np.empty(paths.shape[:2]); a = np.empty_like(u); e = np.empty_like(u)
        scale = np.empty(len(h)); legal = np.empty_like(u, dtype=bool)
        for s, c in zip(names, coeff):
            ix = np.flatnonzero(sites == s)
            x, scale[ix] = m.descriptors(h[ix], paths[ix], floor[ix], crowd[ix])
            u[ix], a[ix], e[ix], legal[ix] = m.decision_costs(x, c, paths[ix], floor[ix], scale[ix])
        choices = np.full_like(frozen, -1)
        for g in data['groups']:
            choices[1:, g] = m.choose_query(h[g], floor[g], paths[g], paths[g, run.NAMES.index(run.parent.OLS)],
                scale[g], u[g], legal[g])
        for r in np.unique(rec):
            ix = np.flatnonzero(rec == r); choice = choices[-1, ix]; j = np.flatnonzero(choice >= 0)
            proposed = (ix[j], choice[j]); positive = u[proposed] > 0
            all_positive = legal[ix] & (u[ix] > 0)
            final = m.budget_prefix(choice, u[ix], a[ix], e[ix], scale[ix]); choices[-1, ix] = final
            selected = final >= 0; jj = (ix[selected], final[selected]); selected_actions = final[selected]
            results.append(dict(seed=seed, recording=str(r), source=str(sites[ix[0]]),
                joint_positive=int(positive.sum()),
                joint_positive_easy_nonpositive=int((positive & (e[proposed] <= 0)).sum()),
                all_actions_positive=int(all_positive.sum()),
                all_actions_positive_easy_nonpositive=int((all_positive & (e[ix] <= 0)).sum()),
                all_actions_positive_single_feasible=int((all_positive & (e[ix] <= 0) & (a[ix] <= 0)).sum()),
                any_legal_easy_nonpositive=bool((legal[ix] & (e[ix] <= 0)).any()),
                budget_selected=int(selected.sum()),
                budget_actions=np.bincount(selected_actions, minlength=len(run.NAMES)).tolist(),
                budget_radii=np.linalg.norm(paths[jj] - floor[ix[selected]], axis=-1).max(1).tolist()))
        np.testing.assert_array_equal(choices, frozen)
        print(dict(seed=seed, action_replay_exact=True), flush=True)
    zero = [r for r in results if not r['budget_selected']]
    counts = dict(no_positive_joint_proposal=sum(r['joint_positive'] == 0 for r in zero),
        positive_joint_but_all_over_easy=sum(r['joint_positive'] > 0 and not r['joint_positive_easy_nonpositive'] for r in zero),
        zero_switch_with_an_easy_feasible_joint_item=sum(r['joint_positive_easy_nonpositive'] > 0 for r in zero),
        zero_switch_with_any_positive_single_feasible=sum(r['all_actions_positive_single_feasible'] > 0 for r in zero),
        zero_switch_with_no_easy_slack_from_any_legal_action=sum(not r['any_legal_easy_nonpositive'] for r in zero),
        zero_switch_views=len(zero))
    path = run.PUBLIC/'budget_diagnosis.json'
    run.once(path, run.stable(dict(views=results, counts=counts, forecast_and_action_order=run.NAMES,
        all_frozen_choices_exact=True, labels_opened=False, new_policy=False)))
    run.once(run.PUBLIC/('budget_diagnosis_replay_execution.json' if args.replay else 'budget_diagnosis_execution.json'),
             dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(path)))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
