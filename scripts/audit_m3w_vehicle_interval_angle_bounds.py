"""Replay the frozen interval-angle comparison with two-sided missing-label bounds."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_vehicle_pair_angle_bounds as bound
from scripts import run_m3w_vehicle_interval_angle as study

np, read, write, sha, screen, short, old = (study.np, study.read, study.write, study.sha,
                                           study.screen, study.short, study.old)
ARMS = ('agreement', 'calibrated_agreement')


def run():
    key, data, _, parent, _, _, direct, _ = study.register()
    f, r = (study.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    freeze, result = read(f), read(r)
    if (freeze['registration'] != key or result['freeze'] != sha(f) or not result['original_2pct']
            or result['external_outcomes_read'] or result['directions'] != 126):
        raise ValueError('Require complete, unchanged development result')
    shortlist = read(short.OUT/'freeze.json')['cells']
    by_site = defaultdict(list)
    support = {q: {tag: Counter() for tag in ARMS} for q in old.QUOTAS}
    for name, cell in parent['cells'].items():
        for site, pref in cell['packets'].items():
            cal, = set(cell['packets'])-{site}
            ar = freeze['cells'][name]['packets'][site]
            cr = direct['cells'][name]['packets'][site]
            zref = shortlist[name]['packets'][site]
            cut_a = read(study.LOCAL/f'{name}_{cal}_cutoff.json')
            if cut_a['identity']['freeze'] != result['freeze'] or cut_a['identity']['site'] != cal:
                raise ValueError('Opposite-site calibration receipt does not match')
            cut_b = direct['cells'][name]['cutoffs'][cal]['direct']
            label = study.action.labels(data[site]['file'])
            with (screen.packet_file(short.ctx.study.LOCAL, pref) as p, screen.packet_file(short.LOCAL, zref) as z,
                  screen.packet_file(study.LOCAL, ar) as a, screen.packet_file(study.prior.LOCAL, cr) as b):
                masks = old.masks(z)
                n = len(z['shortlist'])
                vals = {q: {tag: np.zeros(4) for tag in ARMS} for q in masks}
                for i, (paths, y, valid, _, _) in enumerate(short.truth(p, label, cell['cut'], z['shortlist'])):
                    for q, at in masks.items():
                        if not at[i]:
                            continue
                        left, right = a['agreement'][i], b['direct_agreement'][i]
                        x, t = cut_a['value'][q][0], cut_b[q][0]
                        candidates = ((left, right), (left if x is not None and a['score'][i] >= x else 0,
                                                     right if t is not None and b['direct_score'][i] >= t else 0))
                        for tag, (l, rr) in zip(ARMS, candidates, strict=True):
                            if l == rr:
                                continue
                            value, counts = bound.pair_bounds(paths[l+1], paths[rr+1], y, valid)
                            vals[q][tag] += value/n
                            support[q][tag].update(counts, different_views=1)
            by_site[site].append((cell['seed'], vals))
    draws = np.random.default_rng(2917).integers(7, size=(3000, 7))
    effects = {}
    for q in old.QUOTAS:
        effects[q] = {}
        for tag in ARMS:
            site = np.asarray([np.mean([np.mean([v[q][tag] for seed, v in by_site[s] if seed == j], axis=0)
                                        for j in study.base.SEEDS], axis=0) for s in study.base.SITES])
            expected = result['interval_minus_direct'][q][tag]['relative']['site_means']
            np.testing.assert_allclose(site[:, 0], expected, rtol=0., atol=1e-12)
            ci = np.quantile(site[draws].mean(1), [.025, .975], axis=0).T
            effects[q][tag] = {k: dict(point=float(site[:, j].mean()), ci95=ci[j].tolist(), site_means=site[:, j].tolist())
                               for j, k in enumerate(bound.FIELDS)}
    write(study.OUT/'relative_bounds.json', dict(result_source='fresh_postscore_development_replay',
        score_sha256=sha(r), freeze_sha256=sha(f), effects=effects, support=support,
        original_2pct=True, external_outcomes_read=False, deployment_changed=False), immutable=True)
    print('Replayed all seven site-level lower endpoints; saved two-sided contrasts.', flush=True)


if __name__ == '__main__':
    run()
