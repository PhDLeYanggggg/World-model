"""Read-only OpenDD comparison with uniform allocation at the same spend."""

import fcntl
import os
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_full_grid_count_motion as parent
from src.evaluation import m3w_fixed_spend_uniform as uniform

run, prior, base, before, opt, np, torch = (parent.run, parent.prior, parent.base,
                                         parent.before, parent.optimizer, parent.np, parent.torch)
OUT = parent.OUT.with_name('opendd_fixed_spend_uniform_v1')
PRIVATE = parent.PRIVATE.with_name(OUT.name)
EXACT, DRAWS = 4096, 8192


def evaluate(inputs, idx, old, full, salt):
    _, _, _, d, bank, menu, _, _, cuts = inputs
    ai, q, rows = idx
    _, legal, all_rows, chosen, _ = menu
    legal = legal[q]
    cv, ols = bank[0][ai], bank[1][ai]
    y, valid = d['y'][ai], d['valid'][ai]
    total, cap = old['edits'], old['cap']
    movement = base.model.movement_prefix(bank[0], chosen, all_rows)[q]
    space = uniform.prefix_space(legal, total, base.model.within_cap(movement, cap))
    known_positive = full['robust_status'] == parent.VERIFIED
    if not space.size or full['robust_status'].startswith('unresolved'):
        raise ValueError('Expected a legal frozen spend and an exact hindsight result')
    plan = np.asarray(old['score_plan'], dtype=int)
    original_path, original = before.measure(d, ai, bank, chosen, rows, plan, cuts)
    official = prior.certificate(cv, original_path, y, valid, cuts[0], original)[0]
    frozen = official and min(original['vs_cv'], original['vs_ols4']) > opt.POSITIVE_PP
    out = dict(edits=total, frozen_original_grid=int(official), frozen_strict=int(frozen),
               hindsight=int(known_positive), prefix_space_size=str(space.size))
    if not known_positive:
        if frozen:
            raise ValueError('Frozen success contradicts the verified exclusion')
        return dict(out, mode='excluded_or_no_edit', proposals=0, legal=0, passed=0, probability=0.)

    g, risk, easy, _, offset, easy_cap, mass = base.old.prefix_terms(cv, ols, chosen[ai], y, valid, rows, legal, cuts[0])
    grid = opt.prefix_grid_terms(cv, chosen[ai], y, valid, rows, cuts[0])
    risk = np.concatenate((risk, grid[..., :2]), axis=2)
    bounds = np.asarray([0., 0., easy_cap, 0., 0.])
    tol = 1e-10 * max(1., total * np.abs(risk).max(), abs(easy_cap))

    def direct(x, k):
        local_queries = np.arange(len(legal))
        n = np.zeros(len(legal), dtype=int)
        n[x] = k
        if (n.sum() != total or not legal[local_queries, n].all()
                or not opt.respects_tokens(n, legal[:, 1])
                or not base.model.within_cap(movement[local_queries, n].sum(), cap)):
            raise ValueError('Invalid exact-budget sample')
        p, score = before.measure(d, ai, bank, chosen, rows, n, cuts)
        return prior.certificate(cv, p, y, valid, cuts[0], score)[0] and min(score['vs_cv'], score['vs_ols4']) > opt.POSITIVE_PP

    def coeff(x, k):
        if not easy[x, k].any() or not grid[x, k, 2].any():
            return False
        r = risk[x, k].sum(0) - bounds
        if (r > tol).any():
            return False
        gain = 100. * (offset + g[x, k].sum(0)) / mass
        gap = float(gain.min()) - opt.POSITIVE_PP
        if abs(gap) <= 1e-9 or ((np.abs(r) <= tol).any() and gap > 0.):
            return direct(x, k)
        return gap > 0.

    for choice, expected in ((plan, frozen), (np.asarray(full['witness']), True)):
        x = np.flatnonzero(choice)
        if coeff(x, choice[x]) != expected:
            raise ValueError('Prefix coefficients disagree on the archived allocation')

    exact = space.size <= EXACT
    rng = Random(salt)
    ranks = range(space.size) if exact else (rng.randrange(space.size) for _ in range(DRAWS))
    checked, legal_count, passed = set(), 0, 0
    for rank in ranks:
        x, k = space.unrank(rank)
        if not base.model.within_cap(movement[x, k].sum(), cap):
            continue
        good = coeff(x, k)
        if good not in checked:
            if direct(x, k) != good:
                raise ValueError('Sampled plan disagrees with direct evaluation')
            checked.add(good)
        legal_count += 1
        passed += good
    if exact and not passed:
        raise ValueError('Exhaustive uniform space lost a known full-grid witness')
    return dict(out, mode='exact' if exact else 'sampled',
                proposals=space.size if exact else DRAWS, legal=legal_count, passed=passed,
                probability=passed/legal_count if legal_count else None)


def summarize(groups):
    rows = [x for seeds in groups.values() for view in seeds.values() for x in view]
    agg = {k: sum(r[k] for r in rows) for k in
           ('edits', 'frozen_original_grid', 'frozen_strict', 'hindsight', 'proposals', 'legal', 'passed')}
    agg.update(views=len(rows), modes={k: sum(r['mode'] == k for r in rows) for k in
               ('excluded_or_no_edit', 'exact', 'sampled')},
               no_accepted_draws=sum(r['probability'] is None for r in rows))
    agg['expected_passes_low'] = sum(r['probability'] or 0. for r in rows)
    agg['expected_passes_high'] = agg['expected_passes_low'] + agg['no_accepted_draws']
    agg['conditional_hindsight_median'] = float(np.median(
        [r['probability'] for r in rows if r['hindsight'] and r['probability'] is not None]))
    sampled = [r for r in rows if r['mode'] == 'sampled' and r['legal']]
    agg['sampling_se_passes'] = float(np.sqrt(sum(
        ((r['passed']+.5)/(r['legal']+1))*((r['legal']-r['passed']+.5)/(r['legal']+1))/r['legal'] for r in sampled)))
    locations = {}
    for site, seeds in groups.items():
        locations[site] = []
        for seed, view in seeds.items():
            p = sum(r['probability'] or 0. for r in view)
            locations[site].append(dict(seed=seed, views=len(view), frozen=sum(r['frozen_strict'] for r in view),
                                        hindsight=sum(r['hindsight'] for r in view), uniform_low=p,
                                        uniform_high=p+sum(r['probability'] is None for r in view)))
    draws = np.random.default_rng(817).integers(len(locations), size=(3000, len(locations)))
    effects = {}
    for arm in ('frozen', 'hindsight'):
        lo, hi = [np.array([np.mean([100*(r[arm]-r[f'uniform_{side}'])/r['views'] for r in v])
                            for v in locations.values()]) for side in ('high', 'low')]
        effects[arm+'_minus_uniform'] = dict(point_bounds_pp=[float(lo.mean()), float(hi.mean())],
            location_ci95_pp=[float(np.quantile(lo[draws].mean(1), .025)), float(np.quantile(hi[draws].mean(1), .975))])
    return agg, locations, effects


def main():
    prereqs = (parent.OUT/'registration.json', parent.OUT/'readout.json', base.OUT/'registration.json')
    fixed, archived, original = (run.sha(p) for p in prereqs)
    verify = run.read(prereqs[1])
    if (verify['registration'] != fixed or verify['summary']['unresolved'] != 0
            or verify['new_external_outcome_access'] or not verify['original_2pct']):
        raise ValueError('OpenDD fixed-spend checks are incomplete')
    reg = OUT/'registration.json'
    inputs = [*prereqs, Path(__file__), Path(uniform.__file__), OUT/'protocol.md']
    run.once(reg, run.stable(dict(inputs=[run.artifact(p) for p in inputs], exact_limit=EXACT, draws=DRAWS,
                                  original_2pct=True, new_external_outcome_access=False)))
    binding = run.sha(reg)
    groups = {}
    for inputs, _ in prior.matched.context_cells():
        site, seed, _, d, _, menu, _, _, _ = inputs
        rows = []
        for i, rec in enumerate(sorted(set(d['rec']))):
            tag = f'{site}_{seed}_{i:04d}.json'
            a, b, p = base.PRIVATE/tag, parent.PRIVATE/tag, PRIVATE/tag
            old, full = run.read(a), run.read(b)
            expected = dict(registration=binding, original=run.sha(a), full=run.sha(b))
            if (old['identity'] != dict(registration=original, site=site, seed=seed, ordinal=i)
                    or full['identity'] != dict(registration=fixed, original=expected['original'])):
                raise ValueError('Unbound view')
            if not p.exists():
                result = evaluate(inputs, before.indices(d, rec, menu[2]), old, full,
                                  int(expected['full'][:16], 16))
                run.once(p, run.stable(dict(identity=expected, result=result)))
            saved = run.read(p)
            if saved['identity'] != expected:
                raise ValueError('Foreign uniform checkpoint')
            rows.append(saved['result'])
        groups.setdefault(site, {})[seed] = rows
        beat = dict(pid=os.getpid(), site=site, seed=seed, completed_cells=sum(map(len, groups.values())))
        run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', beat)
        print(beat, flush=True)
    if len(groups) != 7 or any(set(v) != {17, 29, 43} for v in groups.values()):
        raise ValueError('Missing registered location or seed')
    summary, by_location, contrasts = summarize(groups)
    for a, b in (('views', 'views'), ('edits', 'edits'), ('hindsight', 'full_grid'), ('frozen_original_grid', 'blind_full_grid')):
        if summary[a] != verify['summary'][b]:
            raise ValueError('Verified fixed-spend outcome changed')
    run.once(OUT/'readout.json', run.stable(dict(registration=binding, verified_fixed_spend_hash=archived,
             summary=summary, by_location=by_location, contrasts=contrasts, original_2pct=True,
             hindsight_diagnostic=True, new_external_outcome_access=False, training=False, deployment_changed=False)))
    print(summary, contrasts, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
