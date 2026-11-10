"""Diagnose the WEPDTOF fixed-count pair penalty without reading targets."""

from collections import Counter, defaultdict
from itertools import combinations, product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_wepdtof_frozen_capacity as freeze_check
from scripts import run_m3w_wepdtof_external_readout as frozen
from src.evaluation import m3w_penalty_breakpoint as math

study, base, np, torch, method = frozen.study, frozen.base, frozen.np, frozen.torch, frozen.method
OUT = study.OUT.with_name('wepdtof_penalty_breakpoint_v1')


def reconstruct(est, legal, q, h, cv, bank):
    utility = est[..., :2]
    risk = est[..., 2:4]-.02*est[..., 4:6]
    chosen = {k: cv.copy() for k in ('fixed_no_pair', 'fixed_joint')}
    cases = []
    for query in np.unique(q):
        ix = np.flatnonzero(q == query)
        eligible = ix[legal[ix].all(1)]
        n = min(2, len(eligible), len(ix)-1)
        if n < 1:
            continue
        local = max(float(np.median(np.linalg.norm(np.diff(h[ix], axis=1), axis=-1)))*12., 1e-8)
        menu, gains, penalties = [], [], []
        for recipients in combinations(eligible, n):
            near = np.exp(-np.linalg.norm(h[recipients[0], -1]-h[recipients[-1], -1])/local)
            for arms in product(range(bank.shape[1]), repeat=n):
                a, b = np.array(tuple(zip(recipients, arms))).T
                u = utility[a, b].sum(0).min()-np.maximum(risk[a, b].sum(0), 0.).sum()
                offsets = bank[a, b]-cv[a]
                p = near*np.linalg.norm(offsets[0]-offsets[-1], axis=1).mean()
                menu.append((a, b))
                gains.append(u)
                penalties.append(p)
        r = math.breakpoint(gains, penalties, method.PAIR_WEIGHT)
        for name, pick in (('fixed_no_pair', 'base'), ('fixed_joint', 'winner')):
            a, b = menu[r[pick]]
            chosen[name][a] = bank[a, b]
        cases.append(dict(edits=n, candidates=len(menu), threshold=r['threshold'],
                          chosen_penalty=r['penalty_at_base'], option_changed=r['base'] != r['winner']))
    return chosen, cases


def summarize(cases):
    counts = Counter()
    ratios = []
    for r in cases:
        n, t = r['edits'], r['threshold']
        counts.update(queries=1, edits=n, candidates=r['candidates'], changed_options=int(r['option_changed']))
        if n == 1:
            if t is not None or r['chosen_penalty'] != 0.:
                raise ValueError('A one-edit query must have zero pair penalty')
            counts['one_edit_zero_pair_term'] += 1
        elif t is None:
            counts['two_edit_no_lower_penalty_alternative'] += 1
        else:
            counts['two_edit_finite_crossover'] += 1
            counts['strict_crossover_below_registered_weight'] += int(t < method.PAIR_WEIGHT)
            counts['crossover_equal_registered_weight'] += int(t == method.PAIR_WEIGHT)
            ratios.append(t)
    out = dict(counts)
    out['finite_crossover_quantiles'] = (np.quantile(ratios, [0., .25, .5, .75, 1.]).tolist() if ratios else None)
    return out


def run():
    f_path = study.OUT/'augmented_freeze.json'
    f = json.loads(freeze_check.checked(f_path, freeze_check.FREEZE))
    reg, cells, _ = study.register()
    meta = study.OUT/'evaluation_registration.json'
    freeze_check.checked(meta, f['evaluation_registration'])
    approved = study.read(meta)
    if (approved['training_registration'] != reg or not f['outcomes_unopened']
            or not f['original_2pct'] or not approved['original_2pct'] or approved['selection_on_external']
            or len(f['rows']) != 15 or len({r['source'] for r in f['rows']}) != 13 or method.PAIR_WEIGHT != .10):
        raise ValueError('Not the exact original prospective WEPDTOF menu')
    for a in approved['files']:
        study.checked(a)
    camera, windows = study.frozen_camera()
    if study.checked(f['camera']) != camera or set(windows) != {r['clip'] for r in f['rows']}:
        raise ValueError('The forecast-only camera roster changed')
    nets = study.models(reg, cells)
    group, cases = defaultdict(list), []
    identical = 0
    views = 0
    for row in f['rows']:
        clip = row['clip']
        h, q, ids = study.camera_arrays(windows[clip])
        with np.load(study.checked(row['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'], q)
            np.testing.assert_array_equal(z['ids'], ids)
            for s, (net, _) in nets.items():
                cv, ols, spline = (z[f'{s}_{k}'] for k in ('cv', 'ols4', 'spline'))
                bank = method.bank(cv, ols, spline, q)
                x, scale, legal, _ = study.costs.features(h, cv, ols, bank)
                with torch.inference_mode():
                    estimate = net(torch.from_numpy(x)).numpy()*scale[:, None, None]
                recon, result = reconstruct(estimate, legal, q, h, cv, bank)
                for arm, pred in recon.items():
                    np.testing.assert_array_equal(pred, z[f'{s}_{arm}'])
                identical += int(bool(result) and np.array_equal(*recon.values()))
                views += 1
                cases.extend(result)
                group[row['source']].extend(result)
    summary = summarize(cases)
    if (summary['queries'], summary['edits'], summary['one_edit_zero_pair_term'], views, identical) != (198, 366, 30, 45, 36):
        raise ValueError('Forecast-only reconstruction differs from the original edited roster')
    receipt = dict(summary=summary, source_summaries=[summarize(group[k]) for k in sorted(group)],
        clips=15, sources=13, views=views, edited_identity_views=identical, registered_pair_weight=method.PAIR_WEIGHT,
        frozen_paths_exactly_reproduced=True, original_2pct=True, outcomes_opened=False,
        new_fit=False, weight_changed=False, release_changed=False,
        bindings=dict(forecast_freeze=freeze_check.FREEZE, registration=base.sha(meta), donor_training=reg,
            donor_heads=base.sha(study.OUT/'head_freeze.json'), protocol=base.sha(OUT/'protocol.md'),
            driver=base.sha(Path(__file__)), math=base.sha(Path(math.__file__))))
    base.immutable(OUT/'readout.json', receipt)
    print(summary, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run()
