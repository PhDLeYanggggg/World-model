"""Posthoc feasibility and gate-retention audit for fixed retrieval decisions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_native_support_retrieval as study


def main():
    run, np = study.run, study.np
    out = study.PUBLIC
    score, freeze = (out/k for k in ('analysis.json', 'freeze.json'))
    assert run.read(out/'replay_execution.json')['artifact'] == run.artifact(score)
    assert run.read(out/'fit_execution.json')['artifact'] == run.artifact(freeze)
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    y, valid = labels['target_eval']-d['origin'][:, None], labels['valid']
    records = study.prior.h.previous.prior.blocks(d, labels['frames'])
    cut = run.read(study.REG)['cut']
    names = ('candidate_easy', 'candidate_feasible', 'gate_emits', 'gate_keeps_feasible',
             'gate_emits_without_easy', 'gate_emits_nonpositive_lower_gain')
    totals = np.zeros((3, len(names)), dtype=int)
    views = []
    for cell in run.read(freeze)['cells']:
        floor, bank = run.candidates(d, refs, cell['seed'])
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = z['choices']
        for _, (ix, groups) in records.items():
            counts = np.zeros_like(totals)
            for i in range(3):
                original, gated = choices[i, ix], choices[i+3, ix]
                proposed = run.model.selected_paths(original, bank[ix], floor[ix])
                actions = study.prev.critic.action_targets(floor[ix], proposed, y[ix], valid[ix], cut['easy'])
                for q in groups:
                    selected = q[original[q] >= 0]
                    if not len(selected):
                        continue
                    t = study.prev.critic.targets(actions[selected], 1.)
                    emits = bool((gated[q] >= 0).any())
                    if emits and not np.array_equal(original[q], gated[q]):
                        raise ValueError('The final gate may only keep or reject a proposal')
                    feasible = bool(study.prev.critic.accept(t[None])[0])
                    counts[i] += (t[-1] > 0., feasible, emits, emits and feasible,
                                  emits and t[-1] == 0., emits and t[0] <= 0.)
            views.append(counts > 0)
            totals += counts
    registered = run.read(score)['counts']
    for i, arm in enumerate(study.MATCHED):
        assert totals[i, 0] == registered[arm]['query_easy_support']
        assert totals[i, 1] == registered[arm]['query_feasible']
    v = np.asarray(views)
    transitions = {}
    for j, key in enumerate(names[:4]):
        a, b = v[:, 1, j], v[:, 2, j]
        transitions[key] = dict(gains=int((b & ~a).sum()), losses=int((a & ~b).sum()))
    run.once(out/'gate_retention.json', run.stable(dict(analysis=run.artifact(score),
        freeze=run.artifact(freeze), queries=dict(zip(study.MATCHED,
            [dict(zip(names, x.tolist())) for x in totals])),
        views_with_at_least_one=dict(zip(study.MATCHED,
            [dict(zip(names, x.tolist())) for x in v.sum(0)])),
        view_transitions_support_vs_gain=transitions,
        scope='posthoc_exposed_development_diagnostic_only', external_outcomes_opened=False)))


if __name__ == '__main__':
    main()
