"""Decompose paired utility and uncertainty across the exposed source groups."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_joint_selector as study
from src.evaluation.m3w_recording_forecast_comparison import merge_comparisons


def main():
    parent = study.OUT / 'readout.json'
    result = study.read(parent)
    details = study.read(study.PRIVATE / 'recording_scores.json')
    if details['readout'] != study.sha(parent) or result['independent_confirmation']:
        raise ValueError('Require bound development-only results')
    rows = details['rows']
    sources = sorted({r['source'] for r in rows})
    pairs = ('joint_vs_additive', 'joint_vs_independent', 'joint_vs_additive_matched')
    score = {}
    for name in pairs:
        a = []
        for i, src in enumerate(sources, 1):
            cells = []
            for seed in study.SEEDS:
                sub = [r for r in rows if r['source'] == src and r['seed'] == seed]
                v = merge_comparisons([r['comparisons'][name] for r in sub])
                if v['known_pp'] is None:
                    raise ValueError('A source comparison has no reference support')
                cells.append([v['known_pp'], v['completion_lower_pp'], 100*v['unknown_radius']/v['reference']])
            known, lower, missing = study.np.mean(cells, axis=0)
            a.append(dict(source=f'S{i:02d}', known_pp=float(known), lower_pp=float(lower),
                          missing_debit_pp=float(missing), all_seed_lower_positive=bool(all(x[1] > 0 for x in cells))))
        expectation = result['agent']['effects'][name]['completion_lower_pp']['point']
        study.np.testing.assert_allclose(study.np.mean([r['lower_pp'] for r in a]), expectation, atol=1e-10)
        score[name] = dict(positive_sources=sum(r['lower_pp'] > 0 for r in a), sources=a,
                           largest_debit_source=max(a, key=lambda r: r['missing_debit_pp'])['source'])
    freeze = study.read(study.OUT / 'freeze.json')
    a, q = study.a_population()
    swaps = []
    for cell in freeze['cells']:
        with study.np.load(study.checked(cell['choices']), allow_pickle=False) as z:
            study.np.testing.assert_array_equal(z['ids'], a['global_ids'])
            joint, additive = z['joint'] >= 0, z['additive_pair'] >= 0
            if not study.np.array_equal(z['joint'][joint & additive], z['additive_pair'][joint & additive]):
                raise ValueError('Coupling changed a shared recipient path')
            swaps.append(dict(seed=cell['seed'], changed_queries=len(set(q[joint != additive])),
                              swapped_recipients=int((joint & ~additive).sum()), queries=len(set(q)),
                              total_recipients=int(joint.sum())))
    study.once(study.OUT / 'source_diagnostic.json', study.stable(dict(parent=study.artifact(parent),
        source_count=len(sources), analysis=score, swaps=swaps, independent_confirmation=False, post_hoc=True)))
    print({k: v['positive_sources'] for k, v in score.items()}, swaps)


if __name__ == '__main__':
    main()
