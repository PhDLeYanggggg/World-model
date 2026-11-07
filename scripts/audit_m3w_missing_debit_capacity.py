"""Separate risk and utility debits for wholly missing rows in development."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_dual_reference_capacity as dual
from src.evaluation.m3w_dual_reference_capacity import action_masses, solve
from src.evaluation.m3w_joint_path_audit import bootstrap
from src.evaluation.m3w_missing_debit_capacity import ARMS, evidence, variant_masses
from src.evaluation.m3w_ols_blend_capacity import prune_unknown

run, prev, np = dual.run, dual.prev, dual.np
PUBLIC = dual.PUBLIC.parent/'native_missing_debit_capacity_v1'
PRIVATE = dual.PRIVATE.parent/'native_missing_debit_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(evidence.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_missing_debit_capacity.py')
STATUS = ('verified_both', 'risk_count_infeasible', 'negative_maximin_upper_bound', 'unresolved')
POS, INFEASIBLE, NEG, UNKNOWN = STATUS
ORIG, UTIL, RISK, BOTH = ARMS


def register():
    cut, _ = dual.register()
    old = prev.receipt(dual.PUBLIC, 'replay')
    refs = (*FILES, dual.REG, dual.PUBLIC/'replay_execution.json', Path(dual.__file__),
            Path(prune_unknown.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in refs], cuts=cut,
        arms=ARMS, seeds=prev.SEEDS, limits=[3., 12.], external_access=False,
        original_risk_unchanged=True, exact_query_counts=True)))
    return cut, old


def rescore(cv, ols, bank, y, valid, rec, cut, inverse, counts, w, rows, cols, arm):
    if len(set(rows)) != len(rows):
        raise ValueError('Solver selected multiple paths for one person')
    pred = cv.copy()
    pred[rows] = bank[rows, cols]
    changed = (pred != cv).any(axis=(1, 2))
    np.testing.assert_array_equal(np.bincount(inverse[changed], minlength=len(counts)), counts)
    return evidence(cv, ols, pred, y, valid, rec, cut, w, arm)


def solve_view(cv, ols, bank, y, valid, rec, cut, inv, counts, w, original):
    def checked(p, arm):
        r, c = (np.asarray(v, dtype=int) for v in p['witness'])
        return rescore(cv, ols, bank, y, valid, rec, cut, inv, counts, w, r, c, arm)

    if original['status'] == POS:
        out = {a: dict(status=POS, evidence=checked(original, a), witness=original['witness'],
                       inherited_from=ORIG) for a in (UTIL, RISK, BOTH)}
        return dict(out, original=original)

    reduced = prune_unknown(cv, ols, bank, valid)
    base = action_masses(cv, ols, reduced, y, valid, w, cut['easy'])
    tol = 1e-7*base['scale']
    out = {ORIG: original}
    for arm in (UTIL, RISK, BOTH):
        if arm == UTIL and original['status'] == INFEASIBLE:
            out[arm] = dict(status=INFEASIBLE, inherited_from=ORIG)
            continue
        parent = next((a for a in (UTIL, RISK) if out[a]['status'] == POS), None) if arm == BOTH else None
        if parent is not None:
            out[arm] = dict(status=POS, evidence=checked(out[parent], arm),
                            witness=out[parent]['witness'], inherited_from=parent)
            continue
        m = variant_masses(base, cv, ols, reduced, y, valid, w, arm)
        for seconds in (3., 12.):
            take, s = solve(m, inv, counts, seconds)
            out[arm] = dict(s, status=UNKNOWN, seconds_budget=seconds)
            if take is not None:
                rows, cols = m['row'][take], m['arm'][take]
                actual = rescore(cv, ols, bank, y, valid, rec, cut, inv, counts, w, rows, cols, arm)
                np.testing.assert_allclose(actual['gains'], m['offset']+m['utility'][:, take].sum(1),
                                           rtol=1e-10, atol=tol)
                if actual['supported'] and min(actual['gains']) > tol:
                    out[arm].update(status=POS, evidence=actual, witness=[rows.tolist(), cols.tolist()])
                    break
            if s['infeasible']:
                out[arm]['status'] = INFEASIBLE
                break
            if s['solver_status'] in (0, 1) and s['upper_gain_mass'] is not None and s['upper_gain_mass'] < -tol:
                out[arm]['status'] = NEG
                break
    return out


def summarize(rows, old, switches):
    groups = sorted({r['source'] for r in rows})
    if len(rows) != 102 or switches != 9675 or len(groups) != 14:
        raise ValueError('Expected complete exact-count roster')
    totals = {a: {s: sum(r[a]['status'] == s for r in rows) for s in STATUS} for a in ARMS}
    if totals[ORIG] != {k: old['risk_capacity']['full_forecast_bank'][k] for k in STATUS}:
        raise ValueError('Strict comparator did not replay')
    patterns = {c: Counter() for c in (NEG, INFEASIBLE)}
    for r in rows:
        strict, u, h, both = (r[a]['status'] for a in ARMS)
        if (POS in (strict, u, h) and both != POS) or (strict == POS and (u, h) != (POS, POS)):
            raise ValueError('Relaxation lost a checked strict witness')
        if strict not in patterns:
            continue
        if UNKNOWN in (u, h, both):
            name = 'unresolved_attribution'
        elif u == h == POS:
            name = 'either_omission_suffices'
        elif u == POS:
            name = 'only_utility_omission_suffices_alone'
        elif h == POS:
            name = 'only_risk_omission_suffices_alone'
        else:
            name = 'both_omissions_required' if both == POS else 'still_not_positive'
        patterns[strict][name] += 1
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    for a in (UTIL, RISK, BOTH):
        added = [r for r in rows if r[a]['status'] == POS and r[ORIG]['status'] != POS]
        rates = [[np.mean([100.*(r[a]['status'] == POS)-100.*(r[ORIG]['status'] == POS)
                           for r in rows if (r['source'], r['seed']) == (g, s)])
                  for s in prev.SEEDS] for g in groups]
        contrasts[a] = dict(new_positive_views=len(added), new_positive_recordings=len({r['rec'] for r in added}),
            new_positive_source_groups=len({r['source'] for r in added}),
            source_balanced_verified_rate_difference_pp=bootstrap(rates, draw))
    return dict(views=len(rows), matched_switches=switches, source_groups=len(groups),
                capacity=totals, contrasts=contrasts, original_failure_attribution=patterns,
                only_diagnostic=True, deployment=False, external_access=False, independent_confirmation=False,
                stage5c_executed=False, smc_enabled=False)


def score(d, refs, lab, cut, frozen, old, *, replay, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    found, switches = [], 0
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0] >= 0
        for rec, (ix, _) in records.items():
            _, inverse, sizes = np.unique(d['queries'][ix], return_inverse=True, return_counts=True)
            counts = np.bincount(inverse[selected[ix]], minlength=len(sizes))
            np.testing.assert_array_equal(counts, np.minimum(2, sizes-1))
            switches += int(counts.sum())
            w = 1./sizes[inverse]
            local = (cv[ix], bank[ix, run.NAMES.index(dual.parent.OLS)], bank[ix], y[ix], valid[ix],
                     rec, cut, inverse, counts, w)
            p = dual.PRIVATE/f'{seed}_{rec}.json'
            prior = run.read(p)
            if prior['binding'] != dict(registration=run.sha(dual.REG), frozen=f['artifact']['sha256']):
                raise ValueError('Strict recording lineage differs')
            original = prior['result']['full_forecast_bank']
            if original.get('inherited') or original['status'] == UNKNOWN:
                raise ValueError('Require resolved strict native result')
            binding = dict(registration=run.sha(REG), original=run.sha(p))
            mark = PRIVATE/p.name
            if not mark.exists() and not replay:
                computed = solve_view(*local, original)
                run.once(mark, run.stable(dict(binding=binding, result=computed)))
            cell = run.read(mark)
            if cell['binding'] != binding or cell['result'][ORIG] != original:
                raise ValueError('Ablation checkpoint lineage differs')
            for arm, v in cell['result'].items():
                if v['status'] == POS:
                    r, c = (np.asarray(t, dtype=int) for t in v['witness'])
                    actual = rescore(*local, r, c, arm)
                    if actual != v['evidence'] or not actual['supported'] or min(actual['gains']) <= 0.:
                        raise ValueError('Witness did not replay exactly')
            found.append(dict(cell['result'], source=str(d['sites'][ix[0]]), rec=rec, seed=seed))
            if len(found) % 10 == 0 or len(found) == 102:
                beat(seed=seed, completed=len(found))
    return summarize(found, old, switches)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, old = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit this diagnostic before real-data scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    t = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'replay':
            prev.receipt(PUBLIC, 'score')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-t, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        frozen = prev.receipt(prev.PUBLIC, 'freeze')
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(score(d, refs, lab, cut, frozen, old, replay=phase == 'replay', beat=beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Missing-debit replay differs')


if __name__ == '__main__':
    main()
