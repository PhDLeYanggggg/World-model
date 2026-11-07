"""Explain what happened to the box model's first-query rescue probes."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_box_visibility as exp

run, np, h = exp.run, exp.np, exp.h


def main():
    analysis, freeze = (exp.PUBLIC/n for n in ('analysis.json', 'freeze.json'))
    for p, phase in ((analysis, 'replay'), (freeze, 'fit')):
        if run.read(exp.PUBLIC/f'{phase}_execution.json')['artifact']['sha256'] != run.sha(p):
            raise ValueError('Require exact replay of frozen scores')
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells, controls, ranks = h.cost.banks()
    frozen = {c['seed']: c for c in run.read(freeze)['cells']}
    cuts, p = run.cuts_by_source(), run.parent
    counts = Counter()
    source, recordings = set(), set()
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        top = np.full((2, len(m.floor)), -1, np.int16)
        choice = np.full_like(top, -1)
        cut = {}
        for f in frozen[m.seed]['folds']:
            s = f['identity']['held']
            at = m.ix[s]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                np.testing.assert_array_equal(z['choices'][0], m.direct[at])
                top[:, at], choice[:, at] = z['top'], z['choices'][1:]
            cut[s] = f['thresholds']
        for rec, (ix, g) in records.items():
            s = str(d['sites'][ix[0]])
            e, hard = cuts[s]['easy'], cuts[s]['hard']
            dc, c, b = m.direct[ix], m.floor[ix], m.bank[ix]
            a = exp.anchor.model.labels(c, b, y[ix], valid[ix], e)[1]
            target, mass, _ = exp.learner.supervision(a, dc)
            picks = []
            for arm in top[:, ix]:
                i = np.flatnonzero(arm >= 0)
                if len(i) > 1:
                    raise ValueError('Require one probe per view')
                picks.append((int(i[0]), int(arm[i[0]])) if len(i) else None)
            if any(v is None for v in picks):
                if not all(v is None for v in picks):
                    raise ValueError('Probe eligibility is not matched')
                continue
            support = tuple(a[i, k, 1] > 0. for i, k in picks)
            counts['support_gained'] += int(not support[0] and support[1])
            counts['support_lost'] += int(support[0] and not support[1])
            row, k = picks[1]
            if not target[row, k, 1]:
                continue
            source.add(s)
            recordings.add(rec)
            counts['box_first_rescues'] += 1
            counts['box_first_rescues_missing'] += int(mass[1] == 0.)
            candidate = dc.copy()
            candidate[row] = k
            output = p.readout.score_recording({p.CV: c,
                'direct': run.model.selected_paths(dc, b, c),
                'witness': run.model.selected_paths(candidate, b, c)}, {'pair': ('direct', 'witness', p.CV)},
                y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=hard)
            pair = output['comparisons']['pair']
            if (not output['metrics']['witness']['risk']['supported']
                    or pair['known_delta']-pair['unknown_radius'] < -1e-8):
                raise ValueError('Box probe did not pass original scoring')
            label = ('no_donor_cutoff' if cut[s][1] is None else
                'same_insertion' if choice[1, ix[row]] == k else
                'different_insertion' if (choice[1, ix] != dc).any() else 'cutoff_did_not_trigger')
            counts[label] += 1
        del m
    expected = run.read(analysis)['probe']['trajectory_boxes']
    if (counts['box_first_rescues'] != expected['all_rescue']
            or counts['box_first_rescues_missing'] != expected['missing_rescue']):
        raise ValueError('Probe audit does not match frozen score')
    run.once(exp.PUBLIC/'probe_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), freeze_sha256=run.sha(freeze), counts=dict(counts),
        rescue_source_groups=len(source), rescue_distinct_recordings=len(recordings),
        thresholds_changed=False, oracle_choices_exported=False, independent_confirmation=False)))


if __name__ == '__main__':
    main()
