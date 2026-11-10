"""Hold movement and count fixed when comparing OpenDD joint release scores."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_fixed_joint_release as prior
from src.evaluation import m3w_matched_release_budget as budget

np, pair, learn, sha, write, read = (getattr(prior, k) for k in ('np', 'pair', 'learn', 'sha', 'write', 'read'))
OUT = prior.OUT.parent/'opendd_fixed_joint_budget_v1'
LOCAL = prior.LOCAL.parent/OUT.name
ARMS = prior.ARMS
MODES = ('original_quota', 'movement_matched')


def register():
    key, cells, _, cohorts = prior.register()
    paths = (prior.OUT/'score_freeze.json', prior.OUT/'calibration.json', prior.OUT/'readout.json')
    f, c, r = map(read, paths)
    if (f['registration'] != key or r['registration'] != key or f['outcome_fields_loaded']
            or c['score_freeze_sha256'] != sha(paths[0]) or r['score_freeze_sha256'] != sha(paths[0])
            or r['calibration_sha256'] != sha(paths[1]) or r['directions'] != 126
            or r['independent_confirmation'] or r['external_outcomes_read'] or not r['original_2pct']):
        raise ValueError('Require the original fixed-joint experiment with unchanged roles')
    refs = (*paths, Path(__file__), Path(budget.__file__), OUT/'protocol.md')
    p = OUT/'registration.json'
    write(p, dict(inputs={str(k.relative_to(prior.ROOT)): sha(k) for k in refs},
                  budget_tolerance=budget.TOLERANCE, original_2pct=True, fit=False), immutable=True)
    return sha(p), cells, cohorts, f, r


def frozen_masks(name, site, key, score, costs, seconds):
    path, receipt = (LOCAL/f'{name}_{site}.{ext}' for ext in ('npz', 'json'))
    if not receipt.exists():
        base = np.stack([prior.policy.quota(s) for s in score])
        matched = np.stack([base[0], *(budget.select(s, costs, base[0], seconds) for s in score[1:])])
        saved = prior.base.parent.indices.save_npz(path, dict(original_quota=base, movement_matched=matched,
                                                              costs=costs))
        write(receipt, dict(registration=key, packet_sha256=saved), immutable=True)
    r = read(receipt)
    if r['registration'] != key or r['packet_sha256'] != sha(path):
        raise ValueError('Changed cached movement mask')
    return r['packet_sha256']


def freeze(key, cells, cohorts, scores, seconds):
    weights = {site: d['weight'] for site, d in prior.cohort_rows(cohorts, ('weight',))}
    packets = {}
    for n, (name, cell) in enumerate(cells.items(), 1):
        for site, ref in cell['packets'].items():
            with prior.prior.packet(learn.LOCAL, ref) as z:
                cost = budget.movement(z['cv'], z['joint_rows'], z['joint_paths'], weights[site])
            values = prior.scores(name, site, scores)
            packets[f'{name}:{site}'] = frozen_masks(name, site, key, values, cost, seconds)
        beat('freeze_budget_masks', n)
    write(OUT/'mask_freeze.json', dict(registration=key, packets=packets, outcomes_read=False,
                                      exact_count=True, optimal_witnesses=2*len(packets)), immutable=True)


def beat(phase, n):
    if n == 1 or n % 10 == 0 or n == 63:
        import os
        write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, completed=n, total=63))
        print(f'{phase}: {n}/63', flush=True)


def measure(z, d, masks, outcome, costs, mode):
    cv, rows, paths = (z[k] for k in ('cv', 'joint_rows', 'joint_paths'))
    mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
    k = masks[0].sum()
    anchor = prior.policy.prediction(cv, rows, paths, masks[0])
    result = {}
    for arm, use in zip(ARMS, masks, strict=True):
        ratios = costs[:, use].sum(axis=1)/costs[:, masks[0]].sum(axis=1)
        if use.sum() != k or (mode == 'movement_matched' and np.any(abs(ratios-1.) > budget.TOLERANCE+1e-8)):
            raise ValueError('Changed quota or movement at scoring')
        pred = prior.policy.prediction(cv, rows, paths, use)
        val = pair.paired(pred, anchor, d, z['bounds'], mass) if arm != ARMS[0] else dict(
            ade_available_pp=0., ade_lower_pp=0., relative_lower_native=0.)
        result[arm] = dict(**val, missing_debit_pp=val['ade_available_pp']-val['ade_lower_pp'],
                           move_ratio=ratios[0].item(), weighted_move_ratio=ratios[1].item(),
                           retained_original=int((use & masks[0]).sum()), selected=int(k),
                           risk_pass=int((use & outcome['risk']).sum()), dual=int((use & outcome['dual']).sum()),
                           undefined_easy=int((use & outcome['missing']).sum()))
    return result


def summary(views, previous):
    draws = np.random.default_rng(12773).integers(7, size=(3000, 7))
    names = prior.base.SITES
    out = {}
    counts = ('retained_original', 'selected', 'risk_pass', 'dual', 'undefined_easy')
    for mode in MODES:
        totals, contrasts, range_move = {}, {}, {}
        for a in ARMS:
            totals[a] = {k: sum(r[mode][a][k] for r in views) for k in counts}
            contrasts[a], range_move[a] = {}, {}
            for k in ('move_ratio', 'weighted_move_ratio'):
                x = [r[mode][a][k] for r in views]
                range_move[a][k] = [min(x), max(x)]
            for k in ('ade_available_pp', 'ade_lower_pp', 'relative_lower_native', 'missing_debit_pp',
                      'move_ratio', 'weighted_move_ratio', 'dual_rate_pp'):
                def val(r):
                    x, base = r[mode][a], r[mode]['original']
                    return 100.*(x['dual']-base['dual'])/x['selected'] if k == 'dual_rate_pp' else x[k]
                x = np.array([np.mean([val(r) for r in views if r['site'] == s]) for s in names])
                contrasts[a][k] = dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                                       location_values=x.tolist())
            if mode == 'original_quota':
                ref = previous['summary']['fixed_quarter']
                if any(totals[a][k] != ref['counts'][a][k] for k in counts if k != 'retained_original'):
                    raise ValueError('Old matched-count totals changed')
                if a != 'original':
                    for k in ('ade_available_pp', 'ade_lower_pp', 'relative_lower_native', 'dual_rate_pp'):
                        np.testing.assert_allclose(contrasts[a][k]['location_values'],
                            ref['contrasts'][a][k]['location_values'], atol=1e-12, rtol=0.)
        out[mode] = dict(counts=totals, contrasts=contrasts, movement_ratio_range=range_move)
    return out


def score(key, cells, cohorts, previous):
    p = OUT/'mask_freeze.json'
    prior.require_published(p)
    f = read(p)
    if f['registration'] != key or len(f['packets']) != 126 or f['outcomes_read']:
        raise ValueError('Require the complete published outcome-blind mask freeze')
    data = dict(prior.cohort_rows(cohorts, ('future', 'valid', 'recording', 'weight')))
    views = []
    for n, (name, c) in enumerate(cells.items(), 1):
        for site, ref in c['packets'].items():
            file = LOCAL/f'{name}_{site}.npz'
            if f['packets'][f'{name}:{site}'] != sha(file):
                raise ValueError('Frozen mask changed')
            d = data[site]
            with prior.prior.packet(learn.LOCAL, ref) as z, np.load(file, allow_pickle=False) as m:
                costs = budget.movement(z['cv'], z['joint_rows'], z['joint_paths'], d['weight'])
                np.testing.assert_array_equal(costs, m['costs'])
                np.testing.assert_array_equal(m[MODES[0]][0], m[MODES[1]][0])
                y = pair.evaluate(pair.prediction(z, 'joint'), z, d, c['cut'])
                views.append(dict(site=site, **{k: measure(z, d, m[k], y, costs, k) for k in MODES}))
        beat('score_budget_masks', n)
    result = summary(views, previous)
    write(OUT/'readout.json', dict(result_source='fresh_run_on_exposed_development', registration=key,
          mask_freeze_sha256=sha(p), directions=len(views), locations=7, seeds=prior.base.SEEDS,
          summary=result, original_quota_reproduced=True, original_2pct=True, deployment_changed=False,
          new_training=False, new_external_labels_read=False, independent_confirmation=False), immutable=True)
    for k, v in result.items():
        print(k, v['counts'], v['movement_ratio_range'], v['contrasts']['supported_merit'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    parser.add_argument('--solve-seconds', type=float, default=30.)
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cells, cohorts, f, original = register()
        if args.phase == 'freeze':
            freeze(key, cells, cohorts, f, args.solve_seconds)
        else:
            score(key, cells, cohorts, original)
