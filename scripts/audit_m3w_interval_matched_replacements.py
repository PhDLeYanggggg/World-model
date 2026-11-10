"""Measure ranking at equal realized switch counts on exposed OpenDD data."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_vehicle_pair_angle_bounds as pair
from scripts import run_m3w_vehicle_interval_angle as study
from src.evaluation.m3w_angle_budget_match import match, qualified_expectation

np, read, write, sha, old, screen, short = (study.np, study.read, study.write, study.sha,
                                             study.old, study.screen, study.short)
OUT = study.OUT.parent/'opendd_interval_matched_replacements_v1'
NAMES = ('interval', 'direct')
EFFECTS = ('qualified_per_100_views', 'conservative_trajectory_pp', 'observed_fde_pp',
           'relative_lower', 'relative_upper', 'relative_observed')


def measures(p, label, cut, z, interval, direct):
    masks = old.masks(z)
    flag, gain = old.compare(p, label, cut, z, masks['ego_count'] | masks['fixed_15pct'])
    qualified = flag[..., 1] & (gain[..., :2] > 1e-10).all(-1)
    result, policies = {}, {}
    a, b = interval['agreement'], direct['direct_agreement']
    ix = np.arange(len(a))
    at = np.isin(label['recording'], p['records'])
    w = label['weight'][at]
    err = np.where(label['valid'][at], np.linalg.norm(p['cv'][at]-label['future'][at], axis=-1), 0.)
    cv = np.array([w @ err.sum(1), w @ err[:, -1]])/100.
    if not (cv > 0.).all():
        raise ValueError('Require positive, unchanged CV denominators')
    for q, keep in masks.items():
        k, options, pools = match(keep, a, b, interval['score'], direct['direct_score'])
        count = Counter(matched_swaps=k, zero_quota_directions=int(k == 0), rotated=int(keep.sum()),
                        hindsight_qualified=min(k, int(qualified[keep].any(1).sum())))
        left, right = options
        count['same_recordings'] = int(((left > 0) & (right > 0)).sum())
        count['same_recording_and_angle'] = int(((left == right) & (left > 0)).sum())
        for tag, original, selected, pool in zip(NAMES, (a, b), options, pools, strict=True):
            count[f'{tag}_original_swaps'] = len(pool)
            count[f'{tag}_original_qualified'] = int(qualified[pool, original[pool]].sum())
            count[f'{tag}_qualified'] = int(qualified[ix, selected].sum())
            count[f'{tag}_risk_passes'] = int(flag[ix[keep], selected[keep], 0].sum())
            count[f'{tag}_strict_passes'] = int(flag[ix[keep], selected[keep], 3].sum())
            count[f'{tag}_random_qualified'] = qualified_expectation(qualified, original, pool, k)
        diff = count['interval_qualified']-count['direct_qualified']
        result[q] = (count, np.array([100.*diff/len(a), 0., 0., 0., 0., 0.]))
        policies[q] = options
    for i, (paths, y, valid, weights, _) in enumerate(short.truth(p, label, cut, z['shortlist'])):
        for q, (l, r) in policies.items():
            if l[i] == r[i]:
                continue
            left, right = paths[l[i]+1], paths[r[i]+1]
            g, lower = study.base.parent.paired(right, left, y, valid)
            rel, _ = pair.pair_bounds(left, right, y, valid)
            result[q][1][1:] += np.r_[
                (weights @ lower.sum(1))/cv[0], (weights @ g[:, -1])/cv[1], rel[:3]/len(a)]
    return result


def run():
    key, data, _, parent, _, _, direct, _ = study.register()
    f, r = (study.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    freeze, result = read(f), read(r)
    if (freeze['registration'] != key or result['freeze'] != sha(f) or not result['original_2pct']
            or result['external_outcomes_read'] or result['directions'] != 126):
        raise ValueError('Require previously scored, immutable model comparison')
    shortlist = read(short.OUT/'freeze.json')['cells']
    by_site = defaultdict(list)
    totals = {q: Counter() for q in old.QUOTAS}
    site_counts = defaultdict(lambda: {q: Counter() for q in old.QUOTAS})
    for name, cell in parent['cells'].items():
        for s, upstream in cell['packets'].items():
            a, b = freeze['cells'][name]['packets'][s], direct['cells'][name]['packets'][s]
            zref = shortlist[name]['packets'][s]
            if a['shortlist'] != zref['sha256'] or b['shortlist'] != zref['sha256']:
                raise ValueError('Different frozen menus')
            y = study.action.labels(data[s]['file'])
            with (screen.packet_file(short.ctx.study.LOCAL, upstream) as p,
                  screen.packet_file(short.LOCAL, zref) as z, screen.packet_file(study.LOCAL, a) as left,
                  screen.packet_file(study.prior.LOCAL, b) as right):
                m = measures(p, y, cell['cut'], z, left, right)
            for q, (count, _) in m.items():
                totals[q].update(count)
                site_counts[s][q].update(count)
            by_site[s].append((cell['seed'], {q: effect for q, (_, effect) in m.items()}))
    sites, seeds = study.base.SITES, study.base.SEEDS
    draws = np.random.default_rng(2917).integers(len(sites), size=(3000, len(sites)))
    effects = {}
    for q, count in totals.items():
        for tag in NAMES:
            expected = result['counts'][tag][q]['agreement']
            if (expected['swaps'], expected['qualified_joint_gain'], expected['rotated']) != (
                    count[f'{tag}_original_swaps'], count[f'{tag}_original_qualified'], count['rotated']):
                raise ValueError('Original unrestricted counts did not replay')
        loc = np.array([np.mean([np.mean([v[q] for sd, v in by_site[s] if sd == j], axis=0)
                                 for j in seeds], axis=0) for s in sites])
        ci = np.quantile(loc[draws].mean(1), [.025, .975], axis=0).T
        effects[q] = {k: dict(point=float(loc[:, j].mean()), ci95=ci[j].tolist(), site_means=loc[:, j].tolist())
                      for j, k in enumerate(EFFECTS)}
    write(OUT/'readout.json', dict(result_source='fresh_matched_budget_development_replay',
        freeze_sha256=sha(f), scored_readout_sha256=sha(r), protocol_sha256=sha(OUT/'protocol.md'),
        directions=sum(map(len, by_site.values())), bootstrap_locations=len(sites), bootstrap_draws=len(draws),
        totals=totals, anonymous_site_counts=[site_counts[s] for s in sites], interval_minus_direct=effects,
        original_2pct=True, external_outcomes_read=False,
        models_refit=False, deployment_changed=False), immutable=True)
    print(dict(totals), flush=True)


if __name__ == '__main__':
    run()
