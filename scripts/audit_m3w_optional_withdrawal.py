"""Check whether fixed learned insertions admit a non-loss withdrawal."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_optional_withdrawal as exp

run, np, h = exp.run, exp.np, exp.h


def feasible(atoms, known, dc, q, crowd, i, k):
    base, full = exp.previous.mass(atoms, dc)
    if exp.previous.previous.model.supported(base, full):
        return False, None
    add = atoms[i, k]
    support = exp.previous.previous.model.supported
    lower = add[-1]/crowd[i]
    solo = bool(support(base+add, full) and lower >= 0.)
    j = np.flatnonzero((dc >= 0) & (q >= q[i]))
    drop = atoms[j, dc[j]]
    gain = lower+np.where(known[j], -drop[:, -1], drop[:, -1])/crowd[j]
    good = support(base+add-drop, full) & (gain >= 0.)
    return solo, int(j[np.flatnonzero(good)[0]]) if good.any() else None


def predict_frozen(f, x, center, spread):
    torch = exp.model.torch
    state = torch.load(run.checked(f['checkpoints'][0]), map_location='cpu', weights_only=True)
    nn = exp.model.nn
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 1))
    net.load_state_dict(state['model'])
    return exp.model.predict(net.eval(), np.clip((x-center)/spread, -8., 8.).astype(np.float32))


def main():
    analysis, freeze = (exp.PUBLIC/n for n in ('analysis.json', 'freeze.json'))
    for path, phase in ((analysis, 'replay'), (freeze, 'fit')):
        if run.read(exp.PUBLIC/f'{phase}_execution.json')['artifact']['sha256'] != run.sha(path):
            raise ValueError('Require verified frozen experiment')
    exp.model.torch.set_num_threads(4)
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    motion = h.previous.model.motion(d['history'], d['groups'])
    cells, controls, ranks = h.cost.banks()
    saved = {c['seed']: c for c in run.read(freeze)['cells']}
    cuts, p = run.cuts_by_source(), run.parent
    names = ('first_unthresholded', 'insert_only', 'matched_insert')
    count = {n: Counter() for n in names}
    missing = {n: Counter() for n in names}
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        choice = np.full((4, len(m.floor)), -1, np.int16)
        folds = {}
        for f in saved[m.seed]['folds']:
            s = f['identity']['held']
            at = m.ix[s]
            with np.load(run.checked(f['cache']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
                choice[:, at] = z['choices']
                folds[s] = (f, z['insert_center'], z['insert_spread'])
        np.testing.assert_array_equal(choice[0], m.direct)
        for rec, (ix, groups) in records.items():
            s = str(d['sites'][ix[0]])
            e, hard = cuts[s]['easy'], cuts[s]['hard']
            c, b = m.floor[ix], m.bank[ix]
            dc, add, _ = exp.design(m, ix, groups, m.co[s], m.rank(s), motion)
            np.testing.assert_array_equal(dc, m.direct[ix])
            i, k, q, x = add
            scores = predict_frozen(folds[s][0], x, folds[s][1], folds[s][2])
            probe = exp.model.first(np.arange(len(i)), q, scores, 0.)
            probes = [None if probe is None else (i[probe], k[probe])]
            for arm in choice[1:3, ix]:
                at = np.flatnonzero((arm >= 0) & (dc < 0))
                probes.append(None if not len(at) else (at[0], arm[at[0]]))
            a = exp.anchor.model.labels(c, b, y[ix], valid[ix], e)[1]
            base, _ = exp.previous.mass(a, dc)
            path = run.model.selected_paths(dc, b, c)
            ordinal = np.empty(len(ix), dtype=int)
            for v, g in enumerate(groups):
                ordinal[g] = v
            for name, move in zip(names, probes):
                buckets = (count[name], missing[name]) if base[1] == 0. else (count[name],)
                if move is None:
                    continue
                row, k = move
                solo, drop = feasible(a, valid[ix].any(1), dc, ordinal, d['crowd'][ix], row, k)
                if drop is not None:
                    z = dc.copy()
                    z[[row, drop]] = [k, -1]
                    output = p.readout.score_recording({p.CV: c, 'direct': path,
                        'witness': run.model.selected_paths(z, b, c)}, {'pair': ('direct', 'witness', p.CV)},
                        y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=hard)
                    pair = output['comparisons']['pair']
                    if (not output['metrics']['witness']['risk']['supported']
                            or pair['known_delta']-pair['unknown_radius'] < -1e-8):
                        raise ValueError('Withdrawal witness failed original scorer')
                for bucket in buckets:
                    bucket['insertions'] += 1
                    bucket['observed_easy_reference'] += int(a[row, k, 1] > 0.)
                    bucket['solo_nonloss_rescue'] += int(solo)
                    bucket['repairable_only_after_withdrawal'] += int(drop is not None and not solo)
                    bucket['repairable_only_after_later_withdrawal'] += int(
                        drop is not None and not solo and ordinal[drop] > ordinal[row])
        del m
    expected = run.read(analysis)['risk']
    for arm in names[1:]:
        if count[arm]['insertions'] != expected[arm]['insertions']:
            raise ValueError('Mismatch to frozen insertion counts')
    run.once(exp.PUBLIC/'insertion_diagnostic.json', run.stable(dict(
        analysis_sha256=run.sha(analysis), freeze_sha256=run.sha(freeze),
        all={k: dict(v) for k, v in count.items()}, missing={k: dict(v) for k, v in missing.items()},
        hindsight_only=True, thresholds_changed=False, choices_exported=False,
        independent_confirmation=False)))


if __name__ == '__main__':
    main()
