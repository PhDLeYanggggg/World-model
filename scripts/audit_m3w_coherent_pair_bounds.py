"""Audit shared missing positions on the existing OpenDD development choices."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_vehicle_pair_angle_bounds as prior
from src.evaluation import m3w_coherent_pair_bounds as model

study, old, short, np = prior.study, prior.old, prior.short, prior.np
read, write, sha, screen, action = prior.read, prior.write, prior.sha, prior.screen, prior.action
OUT = study.OUT.parent/'opendd_pair_coherent_bounds_v1'
METRICS = ('observed', 'old_lower', 'old_upper', 'coherent_observed', 'new_lower', 'new_upper', 'one_missing_debit',
           'two_missing_debit', 'debit_reduction')
TOL = 1e-12


def audit(p, label, cut, z, pair, prev, direct_threshold):
    masks = old.masks(z)
    count = len(z['shortlist'])
    a = pair['direct_agreement']
    score = {q: {tag: np.zeros(len(METRICS)) for tag in prior.MODES} for q in masks}
    support = {q: {tag: Counter() for tag in prior.MODES} for q in masks}
    for k, (paths, y, valid, _, _) in enumerate(short.truth(p, label, cut, z['shortlist'])):
        saved = {}
        for q, keep in masks.items():
            if not keep[k]:
                continue
            t = direct_threshold[q][0]
            index = a[k] if t is not None and pair['direct_score'][k] >= t else 0
            comps = (('anchor_difference', a[k], pair['anchor_difference_agreement'][k]),
                     ('pointwise', a[k], prev['agreement'][k]), ('incumbent', index, 0))
            for tag, i, j in comps:
                if i == j:
                    continue
                if (i, j) not in saved:
                    saved[i, j] = (model.compare(paths[i+1], paths[j+1], y, valid),
                                   prior.pair_bounds(paths[i+1], paths[j+1], y, valid)[0])
                v, archive = saved[i, j]
                x, before, after = (v[t] for t in ('observed', 'pairwise_debit', 'coherent_debit'))
                if abs(x-archive[2]) > 1e-6 or abs(before-archive[3]) > 1e-10:
                    raise ValueError('Float64 calculation changed more than numerical precision')
                score[q][tag] += np.array([archive[2], archive[0], archive[1], x, x-after, x+after, v['one_missing_debit'],
                                           v['two_missing_debit'], before-after])/count
                ambiguous = x-before < -TOL and x+before > TOL
                unresolved = x-after < -TOL and x+after > TOL
                witness = v['witness_lower'] < -TOL and v['witness_upper'] > TOL
                if (witness and not unresolved) or (unresolved and not ambiguous):
                    raise ValueError('Inconsistent nested bounds and witnesses')
                support[q][tag].update(different_views=1, missing_debit_views=int(before > TOL),
                    tightened_views=int(before-after > TOL), old_straddles_zero=int(ambiguous),
                    new_straddles_zero=int(unresolved), strict_sign_resolved=int(ambiguous and (
                        x-after > TOL or x+after < -TOL)), both_signs_witnessed=int(witness),
                    **{k: v[k] for k in ('informative_stars', 'contracting_stars', 'single_anchor_stars',
                                         'near_translation_stars', 'other_multianchor_stars')})
    return score, support


def run():
    key, data, _, parent, _, _, original, previous, _ = study.register()
    frozen, score_path, bounds_path = (study.OUT/f'{k}.json' for k in ('freeze', 'readout', 'relative_bounds'))
    roster_path = short.OUT/'readout.json'
    f, r, base = (read(p) for p in (frozen, score_path, bounds_path))
    roster = read(roster_path)
    if (f['registration'] != key or r['freeze'] != sha(frozen) or base['readout'] != sha(score_path)
            or r['external_outcomes_read'] or base['external_outcomes_read'] or not base['original_2pct']
            or r['directions'] != 126):
        raise ValueError('Need the fixed OpenDD angle audit')
    by_site = defaultdict(list)
    counts = {q: {tag: Counter() for tag in prior.MODES} for q in old.QUOTAS}
    n = views = 0
    for name, c in parent['cells'].items():
        row = f['cells'][name]
        for s, packet in c['packets'].items():
            cal, = set(c['packets'])-{s}
            d = action.labels(data[s]['file'])
            with (screen.packet_file(short.ctx.study.LOCAL, packet) as p,
                  screen.packet_file(short.LOCAL, original['cells'][name]['packets'][s]) as z,
                  screen.packet_file(study.LOCAL, row['packets'][s]) as a,
                  screen.packet_file(old.LOCAL, previous['directions'][f'{name}_{s}']) as prev):
                x, y = audit(p, d, c['cut'], z, a, prev, row['cutoffs'][cal]['direct'])
                views += len(z['shortlist'])
            by_site[s].append((c['seed'], x))
            for q in y:
                for tag in y[q]:
                    counts[q][tag].update(y[q][tag])
            n += 1
            if n % 18 == 0:
                print('Audited', n, '/ 126 frozen directions', flush=True)
    if (n != 126 or views != roster['totals']['views'] or set(by_site) != set(study.base.SITES)
            or any(len(v) != 18 for v in by_site.values())):
        raise ValueError('Lost frozen source directions')
    draw = np.random.default_rng(2917).integers(7, size=(3000, 7))
    estimates = {}
    for q in old.QUOTAS:
        estimates[q] = {}
        for tag in prior.MODES:
            if counts[q][tag]['different_views'] != base['counts'][q][tag].get('swaps_between_arms', 0):
                raise ValueError('Audit did not preserve the exact choices')
            site = np.asarray([np.mean([np.mean([v[q][tag] for seed, v in by_site[s] if seed == k], axis=0)
                                       for k in study.base.SEEDS], axis=0) for s in study.base.SITES])
            ci = np.quantile(site[draw].mean(1), [.025, .975], axis=0).T
            result = {k: dict(point=float(site[:, i].mean()), ci95=ci[i].tolist(), site_means=site[:, i].tolist())
                      for i, k in enumerate(METRICS)}
            for before, after in (('observed', 'observed'), ('lower', 'old_lower'), ('upper', 'old_upper')):
                b = base['effects'][q][tag][before]
                v = result[after]
                np.testing.assert_allclose([b['point'], *b['ci95']], [v['point'], *v['ci95']], rtol=0., atol=1e-12)
            debit = result['one_missing_debit']['point']+result['two_missing_debit']['point']
            estimates[q][tag] = dict(metrics=result,
                width_reduction_fraction=result['debit_reduction']['point']/debit if debit > 0. else None)
    paths = (Path(__file__), Path(model.__file__), OUT/'protocol.md', frozen, score_path, bounds_path, roster_path)
    root = study.base.prior.ROOT
    return dict(result_source='fresh_fixed_choice_development_audit', counts=counts, estimates=estimates,
        source_hashes={str(p.resolve().relative_to(root)): sha(p) for p in paths},
        directions=n, recordings=roster['recordings'], repeated_views=views,
        original_bound_replayed=True, bootstrap_locations=7, bootstrap_draws=3000, no_new_fit=True,
        original_2pct=True, external_outcomes_read=False, deployment_changed=False,
        witnesses_query_specific_only=True, submission_ready=False)


if __name__ == '__main__':
    d = run()
    write(OUT/'readout.json', d, immutable=True)
    print(d['counts'], flush=True)
