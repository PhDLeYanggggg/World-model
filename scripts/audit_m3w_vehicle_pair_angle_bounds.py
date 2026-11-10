"""Separate missing-label uncertainty from observed relative-motion effects."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_angle as study

old, short, np = study.old, study.short, study.np
read, write, sha, screen, action = study.read, study.write, study.sha, study.screen, study.action
MODES = ('anchor_difference', 'pointwise', 'incumbent')
FIELDS = ('lower', 'upper', 'observed', 'missing_debit')


def pair_bounds(left, right, target, valid):
    i, j = np.triu_indices(len(left), 1)
    left, right = left[i]-left[j], right[i]-right[j]
    known = valid[i] & valid[j]
    truth = np.where(known[..., None], target[i]-target[j], 0.)
    gain = np.linalg.norm(right-truth, axis=-1)-np.linalg.norm(left-truth, axis=-1)
    shift = np.linalg.norm(left-right, axis=-1)
    obs = np.where(known, gain, 0.).sum()/known.size
    debit = np.where(known, 0., shift).sum()/known.size
    affected = shift > 1e-10
    return np.array([obs-debit, obs+debit, obs, debit]), dict(
        changed=int(affected.sum()), observed_changed=int((affected & known).sum()))


def audit(p, label, cut, z, pair, prev, direct_threshold):
    masks = old.masks(z)
    count = len(z['shortlist'])
    a = pair['direct_agreement']
    outputs = {q: {tag: np.zeros(len(FIELDS)) for tag in MODES} for q in masks}
    support = {q: {tag: Counter() for tag in MODES} for q in masks}
    for k, (paths, y, valid, _, _) in enumerate(short.truth(p, label, cut, z['shortlist'])):
        for q, keep in masks.items():
            if not keep[k]:
                continue
            t = direct_threshold[q][0]
            index = a[k] if t is not None and pair['direct_score'][k] >= t else 0
            comparisons = (('anchor_difference', a[k], pair['anchor_difference_agreement'][k]),
                           ('pointwise', a[k], prev['agreement'][k]), ('incumbent', index, 0))
            for tag, i, j in comparisons:
                if i == j:
                    continue
                x, n = pair_bounds(paths[i+1], paths[j+1], y, valid)
                outputs[q][tag] += x/count
                support[q][tag].update(**n, swaps_between_arms=1)
    return outputs, support


def main():
    key, data, _, parent, _, _, original, prior, _ = study.register()
    path, result = (study.OUT/k for k in ('freeze.json', 'readout.json'))
    f, r = read(path), read(result)
    if (f['registration'] != key or r['freeze'] != sha(path)
            or not f['before_held_scoring'] or r['external_outcomes_read']):
        raise ValueError('Require fully scored frozen models')
    by_site = defaultdict(list)
    counts = {q: {tag: Counter() for tag in MODES} for q in old.QUOTAS}
    for name, c in parent['cells'].items():
        row = f['cells'][name]
        for s, packet in c['packets'].items():
            cal, = set(c['packets'])-{s}
            d = action.labels(data[s]['file'])
            with (screen.packet_file(short.ctx.study.LOCAL, packet) as p,
                  screen.packet_file(short.LOCAL, original['cells'][name]['packets'][s]) as z,
                  screen.packet_file(study.LOCAL, row['packets'][s]) as a,
                  screen.packet_file(old.LOCAL, prior['directions'][f'{name}_{s}']) as prev):
                x, n = audit(p, d, c['cut'], z, a, prev, row['cutoffs'][cal]['direct'])
            by_site[s].append((c['seed'], x))
            for q in n:
                for tag in n[q]:
                    counts[q][tag].update(n[q][tag])
    sites = study.base.SITES
    seeds = study.base.SEEDS
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))
    effects = {}
    for q in counts:
        effects[q] = {}
        for tag in counts[q]:
            loc = np.asarray([np.mean([np.mean([v[q][tag] for seed, v in by_site[s] if seed == j], axis=0)
                                       for j in seeds], axis=0) for s in sites])
            if tag == 'incumbent':
                expected = r['effects']['direct'][q]['calibrated_agreement']['relative']
            else:
                expected = r['direct_minus'][tag][q]['agreement']['relative']
            np.testing.assert_allclose(loc[:, 0], [expected['by_site'][s] for s in sites], rtol=0., atol=1e-12)
            ci = np.quantile(loc[draws].mean(1), [.025, .975], axis=0).T
            effects[q][tag] = {k: dict(point=float(loc[:, j].mean()), ci95=ci[j].tolist())
                               for j, k in enumerate(FIELDS)}
    report = dict(result_source='fresh_postscore_development_audit', readout=sha(result),
                  freeze=sha(path), counts=counts, effects=effects, original_2pct=True,
                  external_outcomes_read=False, deployment_changed=False)
    write(study.OUT/'relative_bounds.json', report, immutable=True)
    print(report, flush=True)


if __name__ == '__main__':
    main()
