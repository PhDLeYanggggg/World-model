"""Freeze and score an equal-step displacement sensitivity on opened Songdo flights."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_songdo_fixed_recipient_paths as old
from src.evaluation import m3w_step_budget_match as budget

np, store, sha, read = old.np, old.storage, old.sha, old.read
OUT = old.OUT.parent/'songdo_step_budget_v1'
LOCAL = old.LOCAL.parent/OUT.name
ARMS = ('independent', 'joint')
VARIANTS = ('original', 'smaller_peak', 'matched_steps')
KEYS = ('known_pp', 'lower_pp', 'upper_pp', 'relative_available_native', 'relative_lower_native')
SEEDS = tuple(old.donor.base.SEEDS)


def scope():
    data, groups, _, source, _ = old.base.check_scope()
    p, r = (old.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    original, outcome = read(p), read(r)
    if (original['binding'] != old.phase_binding(source) or set(original['packets']) != set(groups)
            or outcome['proposal_freeze_sha256'] != sha(p) or not outcome['original_diagonal_reproduced']
            or outcome['new_cohort_opened'] or outcome['calibration_outcomes_opened']):
        raise ValueError('Require the original crossed proposals from the opened 160 flights')
    binding = dict(parent_freeze=sha(p), previous=sha(r), driver=sha(Path(__file__)),
                   protocol=sha(OUT/'protocol.md'), budget=sha(Path(budget.__file__)))
    return data, groups, original['packets'], outcome, binding


def transform(cv, packet):
    out = {}
    for seed in SEEDS:
        p = f'seed{seed}_'
        lo, hi = packet[p+'bounds']
        rows = np.stack([packet[p+c+'_rows'] for c in ('II', 'JJ')])
        if rows.shape != (2, 2) or np.any(rows[:, 0] == rows[:, 1]) or np.any((rows < lo) | (rows >= hi)):
            raise ValueError('Frozen recipient is outside its matched query')
        paths = np.stack([packet[p+c+'_paths'] for c in ('II', 'JJ')])
        delta = paths-cv[rows]
        values = budget.match(delta)
        out.update({p+'bounds': np.array([lo, hi]), p+'rows': rows, p+'original': paths,
                    p+'matched': np.array(values is not None)})
        if values is not None:
            variants, radius, scale = values
            out.update({p+k: cv[rows]+variants[k] for k in VARIANTS[1:]})
            out.update({p+'radius': radius, p+'peak_retained': scale, p+'movement_retained':
                        4.*radius.sum()/np.linalg.norm(delta, axis=-1).sum()})
    return out


def freeze(data, groups, packets, binding):
    rows = {}
    for name in sorted(groups):
        stem = Path(name).stem
        _, _, _, bank = data.observed(name)
        f = old.base.primary.checked(old.LOCAL/f'{stem}.npz', packets[name])
        p, r = LOCAL/f'{stem}.npz', LOCAL/f'{stem}.json'
        identity = dict(binding=binding, original=packets[name], bank=sha(bank))
        if not r.exists():
            with np.load(bank, allow_pickle=False) as b, np.load(f, allow_pickle=False) as original:
                values = transform(b['cv'], original)
            store.write_json(r, dict(identity=identity, sha256=store.save_npz(p, values)))
        row = read(r)
        if row['identity'] != identity or row['sha256'] != sha(p):
            raise ValueError('Changed transformed forecasts')
        rows[name] = row['sha256']
        old.heartbeat('step_freeze', len(rows), len(groups))
    old.base.published.secondary.write(OUT/'freeze.json', dict(binding=binding, packets=rows,
        future_coordinates_read=False, previous_roles_unchanged=True, reranking=False))


def readout(data, name, packet, cut, previous):
    item, cache, past, bank = data.observed(name)
    evaluator = old.base.primary.protocol.readout
    with (np.load(past, allow_pickle=False) as p, np.load(bank, allow_pickle=False) as b,
          np.load(packet, allow_pickle=False) as z):
        cv, ols, offsets = b['cv'], b['ols4'], b['offsets']
        np.testing.assert_array_equal(offsets, p['offsets'])
        y, valid = data.targets(name, cache, p, item)
        w = np.repeat(1./np.diff(offsets), np.diff(offsets))
        mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        full, _ = evaluator.coverage(valid, offsets)
        result = {}
        for seed in SEEDS:
            k = f'seed{seed}_'
            lo, hi = z[k+'bounds']
            rows = z[k+'rows']
            matched = bool(z[k+'matched'])
            values = {}
            for arm in VARIANTS if matched else ('original',):
                pred = {a: cv.copy() for a in ARMS}
                for a, r, path in zip(ARMS, rows, z[k+arm], strict=True):
                    pred[a][r] = path
                j, i = pred['joint'], pred['independent']
                forward = evaluator.compare(j, i, y, valid, w, mass, (lo, hi))
                backward = evaluator.compare(i, j, y, valid, w, mass, None)
                direct = dict(forward, upper_pp=-backward['lower_pp'])
                if arm == 'original':
                    prior = previous[str(seed)]['contrasts']['diagonal']
                    for key in KEYS:
                        np.testing.assert_allclose(direct[key], prior[key], atol=1e-12, rtol=0.)
                risk, error = {}, {}
                for a, val in pred.items():
                    r = evaluator.policy(val, cv, cv, y, valid, np.repeat('flight', len(cv)),
                                         easy_cut=cut[0], hard_cut=cut[1])['recordings']['flight']
                    if r['selected'] != 2:
                        raise ValueError('Matched path transformation changed the edit count')
                    gains = [evaluator.compare(val, ref, y, valid, w, mass, None)['lower_pp'] for ref in (cv, ols)]
                    risk[a] = dict(supported=r['supported'], dual=bool(r['supported'] and min(gains) > 0.))
                    error[a] = evaluator.absolute(val, y, valid, full)
                values[arm] = dict(direct=direct, risk=risk, absolute=error)
            result[str(seed)] = dict(variants=values, matched=matched,
                peak_retained=float(z[k+'peak_retained']) if matched else None,
                movement_retained=float(z[k+'movement_retained']) if matched else None)
    return result


def summarize(groups, records, original):
    all_views = [v for r in records.values() for v in r.values()]
    sources = {g: sorted(n for n, s in groups.items() if s == g) for g in sorted(set(groups.values()))}
    draws = np.random.default_rng(1439).integers(len(sources), size=(3000, len(sources)))
    def effect(getter, matched):
        groups = []
        for names in sources.values():
            flights = []
            for n in names:
                v = list(records[n].values())
                if matched and not all(r['matched'] for r in v):
                    continue
                x = [getter(r) for r in v]
                if all(k is not None for k in x):
                    flights.append(float(np.mean(x)))
            groups.append(float(np.mean(flights)) if flights else None)
        if any(x is None for x in groups):
            return dict(point=None, ci95=None, source_means=groups)
        v = np.array(groups)
        return dict(point=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist(),
                    source_means=v.tolist())
    original_direct = {k: effect(lambda r: r['variants']['original']['direct'][k], False) for k in KEYS}
    for k in KEYS:
        np.testing.assert_allclose(original_direct[k]['source_means'], original['effects']['diagonal'][k]['source_means'],
                                   atol=1e-12, rtol=0.)
    variants = {}
    for arm in VARIANTS:
        group = [r['variants'][arm] for r in all_views if arm in r['variants']]
        variants[arm] = dict(direct=original_direct if arm == 'original' else
            {k: effect(lambda r: r['variants'][arm]['direct'][k], True) for k in KEYS},
            risk={a: {k: sum(r['risk'][a][k] for r in group) for k in ('supported', 'dual')} for a in ARMS},
            errors={a: {k: effect(lambda r: r['variants'][arm]['absolute'][a][k], arm != 'original')
                        for k in ('available_ade_native', 'available_fde_native', 'available_ade_p95_native')}
                    for a in ARMS}, evaluated_views=len(group))
    for a, k in zip(ARMS, ('II', 'JJ'), strict=True):
        if variants['original']['risk'][a] != original['risk'][k]:
            raise ValueError('Original risk counts did not reproduce')
    retained = {k: effect(lambda r: r[k], True) for k in ('peak_retained', 'movement_retained')}
    sensitivity = {k: effect(lambda r: r['variants']['matched_steps']['direct'][k]-
                          r['variants']['smaller_peak']['direct'][k], True) for k in KEYS[:3]}
    return dict(variants=variants, retained=retained, step_minus_smaller_peak_endpoint_change=sensitivity,
                unmatched_views=sum(not r['matched'] for r in all_views), flights=len(records),
                source_components=len(sources), bootstrap_draws=len(draws), diagonal_reproduced=True)


def score(data, groups, original, binding):
    f = OUT/'freeze.json'
    old.require_published(f)
    frozen = read(f)
    if (frozen['binding'] != binding or frozen['future_coordinates_read'] or set(frozen['packets']) != set(groups)):
        raise ValueError('Require the published retrospective transformation freeze')
    published = old.base.published
    cal = read(published.PUBLIC/'calibration_receipt.json')
    records = {}
    for n in sorted(groups):
        stem = Path(n).stem
        p = old.base.primary.checked(LOCAL/f'{stem}.npz', frozen['packets'][n])
        before = old.LOCAL/f'{stem}_score.json'
        dest = LOCAL/f'{stem}_readout.json'
        identity = dict(freeze=sha(f), variant=frozen['packets'][n], previous=sha(before))
        if not dest.exists():
            # Use only the frozen scalar cutoffs, not any calibration targets.
            first = read(published.RUN/'confirmation'/f'{stem}.json')
            if (first['easy_cut'], first['hard_cut']) != (cal['easy_cut'], cal['hard_cut']):
                raise ValueError('Readout cutoffs changed')
            result = readout(data, n, p, (first['easy_cut'], first['hard_cut']), read(before)['results'])
            store.write_json(dest, dict(identity=identity, results=result))
        row = read(dest)
        if row['identity'] != identity:
            raise ValueError('A score belongs to another transformation')
        records[n] = row['results']
        old.heartbeat('step_score', len(records), len(groups))
    result = summarize(groups, records, original)
    published.secondary.write(OUT/'readout.json', dict(result, binding=binding, freeze=sha(f),
        result_source='fresh_retrospective_sensitivity', previously_opened_outcomes_reread=True,
        calibration_outcomes_read=False, new_cohort_opened=False, original_2pct=True,
        deployment_changed=False, independent_confirmation=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'score'))
    args = p.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, groups, packets, original, binding = scope()
        if args.phase == 'freeze':
            freeze(data, groups, packets, binding)
        else:
            score(data, groups, original, binding)


if __name__ == '__main__':
    main()
