"""Audit whether fixed OLS4 blends expand dual-reference development capacity."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_dual_reference_capacity as dual
from src.evaluation.m3w_joint_path_audit import bootstrap
from src.evaluation.m3w_ols_blend_capacity import FRACTIONS, MIN_NEW_CHANGE, add_blends, prune_unknown

run, prev, np = dual.run, dual.prev, dual.np
PUBLIC = dual.PUBLIC.parent/'native_ols_blend_capacity_v1'
PRIVATE = dual.PRIVATE.parent/'native_ols_blend_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(add_blends.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_ols_blend_capacity.py')
ARMS = ('native', 'add_causal_blends', 'add_causal_and_learned_blends')
LEARNED = ('eqmotion', 'pct', 'self')
OLS = dual.parent.OLS
CAUSAL = tuple(k for k in run.NAMES if k not in (*LEARNED, OLS))
STATUS = ('verified_both', 'risk_count_infeasible', 'negative_maximin_upper_bound', 'unresolved')


def register():
    cut, _ = dual.register()
    old = prev.receipt(dual.PUBLIC, 'replay')
    if len(CAUSAL) != 4 or (old['views'], old['matched_switches']) != (102, 9675):
        raise ValueError('Expected full frozen bank and original roster')
    refs = (*FILES, dual.REG, dual.PUBLIC/'replay_execution.json', Path(dual.__file__),
            Path(bootstrap.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in refs], cuts=cut,
        arms=ARMS, causal=CAUSAL, learned=LEARNED, seeds=prev.SEEDS, fractions=FRACTIONS,
        minimum_new_change_pixel=MIN_NEW_CHANGE, external_access=False, risk=.02,
        exact_query_counts=True, original_scorer=True)))
    return cut, old


def summarize(rows, switches, old):
    if len(rows) != 102 or switches != 9675:
        raise ValueError('Expected complete three-seed, exact-count roster')
    groups = sorted({r['source'] for r in rows})
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    totals = {a: {s: sum(r[a]['status'] == s for r in rows) for s in STATUS} for a in ARMS}
    if totals['native'] != {k: old['risk_capacity']['full_forecast_bank'][k] for k in STATUS}:
        raise ValueError('Original native comparator did not replay')
    contrasts = {}
    for a, b in zip(ARMS[:-1], ARMS[1:], strict=True):
        added = [r for r in rows if r[b]['status'] == 'verified_both' and r[a]['status'] != 'verified_both']
        cells = [[np.mean([100.*(r[b]['status'] == 'verified_both')-100.*(r[a]['status'] == 'verified_both')
                           for r in rows if (r['source'], r['seed']) == (g, s)])
                  for s in prev.SEEDS] for g in groups]
        contrasts[b+'_vs_'+a] = dict(new_verified_views=len(added),
            new_verified_recordings=len({r['recording'] for r in added}),
            new_verified_source_groups=len({r['source'] for r in added}),
            newly_verified_after_resolved_negative=sum(r[a]['status'] != 'unresolved' for r in added),
            source_balanced_verified_rate_difference_pp=bootstrap(cells, draws))
    return dict(views=len(rows), matched_switches=switches, source_groups=len(groups), risk_capacity=totals,
        contrasts=contrasts, forecasts_not_retrained=True, choices_exported=False, independent_confirmation=False,
        external_access=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def score(d, refs, lab, cut, frozen, old, *, replay, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    found, switches = [], 0
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0] >= 0
        for rec, (ix, _) in records.items():
            q = d['queries'][ix]
            _, inverse, sizes = np.unique(q, return_inverse=True, return_counts=True)
            counts = np.bincount(inverse[selected[ix]], minlength=len(sizes))
            np.testing.assert_array_equal(counts, np.minimum(2, sizes-1))
            w = 1./sizes[inverse]
            switches += int(counts.sum())
            floor, y, mask = cv[ix], target[ix], valid[ix]
            ols = bank[ix, run.NAMES.index(OLS)]
            menus = {ARMS[0]: bank[ix]}

            def menu(a):
                if a not in menus:
                    k = ARMS.index(a)
                    sources = CAUSAL if k == 1 else LEARNED
                    menus[a] = add_blends(floor, ols, menu(ARMS[k-1]), [run.NAMES.index(s) for s in sources])
                return menus[a]

            parent_path = dual.PRIVATE/f'{seed}_{rec}.json'
            original = run.read(parent_path)
            if original['binding'] != dict(registration=run.sha(dual.REG), frozen=f['artifact']['sha256']):
                raise ValueError('Original dual-reference recording cache differs')
            original = dict(original['result']['full_forecast_bank'], witness_menu=ARMS[0])
            if original['status'] == 'unresolved' or original.get('inherited'):
                raise ValueError('Expected resolved native parent with native witnesses')
            binding = dict(registration=run.sha(REG), native=run.sha(parent_path))
            mark = PRIVATE/f'{seed}_{rec}.json'
            if not mark.exists() and not replay:
                row = {ARMS[0]: original}
                for a, b in zip(ARMS[:-1], ARMS[1:], strict=True):
                    if row[a]['status'] == 'verified_both':
                        row[b] = row[a]
                    else:
                        options = prune_unknown(floor, ols, menu(b), mask)
                        got = dual.assess(floor, ols, options, y, mask, q, rec, cut, inverse, counts, w)
                        row[b] = dict(got, witness_menu=b)
                        if (row[a]['status'] == 'negative_maximin_upper_bound'
                                and got['status'] == 'risk_count_infeasible'):
                            raise ValueError('Expanded menu cannot lose original feasibility')
                        if (row[a]['status'] != 'unresolved' and got['status'] == 'verified_both'
                                and max(got['witness'][1]) < menu(a).shape[1]):
                            raise ValueError('New witness contradicts the smaller menu bound')
                run.once(mark, run.stable(dict(binding=binding, result=row)))
            checkpoint = run.read(mark)
            if checkpoint['binding'] != binding or checkpoint['result'][ARMS[0]] != original:
                raise ValueError('Expanded menu checkpoint binding differs')
            row = checkpoint['result']
            for a, item in row.items():
                if item['status'] == 'verified_both':
                    r, k = (np.asarray(t, dtype=int) for t in item['witness'])
                    actual = dual.evidence(floor, ols, menu(item['witness_menu']), y, mask,
                                           q, rec, cut, inverse, counts, w, r, k)
                    if actual != item['evidence'] or not actual['supported'] or min(actual['gains']) <= 0.:
                        raise ValueError(f'Original witness score changed for {a}')
            found.append(dict(row, seed=seed, source=str(d['sites'][ix[0]]), recording=rec))
            if len(found) % 10 == 0 or len(found) == 102:
                beat(seed=seed, completed=len(found))
    return summarize(found, switches, old)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, old = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the experiment before running real scoring')
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
            raise ValueError('Expanded menu replay differs')


if __name__ == '__main__':
    main()
