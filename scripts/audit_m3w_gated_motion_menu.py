"""Evaluate frozen trained motion proposals as optional exact-budget actions."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_gated_motion_residual as motion
from src.evaluation.m3w_joint_path_audit import bootstrap
from src.evaluation.m3w_ols_blend_capacity import prune_unknown

dual, run, prev, np = motion.dual, motion.run, motion.prev, motion.np
PUBLIC = motion.PUBLIC.parent/'native_gated_motion_menu_v1'
PRIVATE = motion.PRIVATE.parent/'native_gated_motion_menu_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), PUBLIC/'protocol.md')
ARMS = ('native', 'add_ungated', 'add_gated', 'add_both')
STATES = ('verified_both', 'negative_maximin_upper_bound', 'risk_count_infeasible', 'unresolved')
POS, NEG, INFEASIBLE, UNKNOWN = STATES


def register():
    cut, _ = motion.register()
    scored = prev.receipt(motion.PUBLIC, 'replay')
    if not scored['original_recipient_masks_equal'] or scored['both_head_substitutions'] != 9675:
        raise ValueError('Expected completed matched new forecaster comparison')
    old = prev.receipt(dual.PUBLIC, 'replay')
    frozen = prev.receipt(motion.PUBLIC, 'infer')
    if not frozen['uses_past_only'] or frozen['registration'] != run.sha(motion.REG):
        raise ValueError('Require frozen past-only forecasts')
    binds = (*FILES, motion.REG, motion.PUBLIC/'infer_execution.json', motion.PUBLIC/'replay_execution.json',
             dual.REG, dual.PUBLIC/'replay_execution.json', Path(prune_unknown.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in binds], cut=cut, arms=ARMS,
        exact_query_counts=True, original_risk_unchanged=True, external_access=False, seeds=prev.SEEDS)))
    return cut, old, frozen


def score(d, refs, lab, cut, old, predictions, *, replay, beat):
    sources = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    frozen = prev.receipt(prev.PUBLIC, 'freeze')
    seen, switches = [], 0
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        new = {}
        for f in predictions['forecasts']:
            if f['seed'] == seed:
                with np.load(run.checked(f['forecast']), allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], d['global_ids'])
                    new[f['arm']] = z['prediction'].astype(float)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            selected = z['choice'][0] >= 0
        for rec, (ix, _) in sources.items():
            q, floor, base = d['queries'][ix], cv[ix], bank[ix]
            ols, y, m = base[:, run.NAMES.index(motion.OLS)], target[ix], valid[ix]
            _, inv, sizes = np.unique(q, return_inverse=True, return_counts=True)
            counts = np.bincount(inv[selected[ix]], minlength=len(sizes))
            np.testing.assert_array_equal(counts, np.minimum(2, sizes-1))
            w = 1./sizes[inv]
            switches += int(counts.sum())
            p = dual.PRIVATE/f'{seed}_{rec}.json'
            parent = run.read(p)
            if parent['binding'] != dict(registration=run.sha(dual.REG), frozen=cell['artifact']['sha256']):
                raise ValueError('Strict native parent cache differs')
            native = dict(parent['result']['full_forecast_bank'], menu='native')
            if native.get('inherited') or native['status'] == UNKNOWN:
                raise ValueError('Require resolved native capacity')
            additions = ((), ('ungated',), ('gated',), ('ungated', 'gated'))

            def menu(a):
                paths = [new[b][ix] for b in additions[ARMS.index(a)]]
                for path in paths:
                    identity = np.linalg.norm(path-floor, axis=2).max(1) <= 1e-6
                    path[identity] = floor[identity]
                return np.concatenate((base, *(path[:, None] for path in paths)), axis=1) if paths else base

            bind = dict(registration=run.sha(REG), native=run.sha(p))
            mark = PRIVATE/p.name
            if not mark.exists() and not replay:
                results = {'native': native}
                for a in ARMS[1:]:
                    inherited = native if native['status'] == POS else None
                    if inherited is None and a == 'add_both':
                        inherited = next((r for r in results.values() if r['status'] == POS), None)
                    if inherited is not None:
                        results[a] = inherited
                        continue
                    row = dual.assess(floor, ols, prune_unknown(floor, ols, menu(a), m),
                                      y, m, q, rec, cut, inv, counts, w)
                    smaller = results.values() if a == 'add_both' else (native,)
                    if (row['status'] == POS and max(row['witness'][1]) < base.shape[1]
                            or row['status'] == INFEASIBLE and any(v['status'] == NEG for v in smaller)):
                        raise ValueError('Expansion contradicts native capacity bound')
                    results[a] = dict(row, menu=a)
                run.once(mark, run.stable(dict(binding=bind, result=results)))
            result = run.read(mark)
            if result['binding'] != bind or result['result']['native'] != native:
                raise ValueError('Expanded training-menu checkpoint differs')
            for r in result['result'].values():
                if r['status'] == POS:
                    rows, arms = (np.asarray(v, dtype=int) for v in r['witness'])
                    val = dual.evidence(floor, ols, menu(r['menu']), y, m, q, rec, cut, inv, counts, w, rows, arms)
                    if val != r['evidence'] or not val['supported'] or min(val['gains']) <= 0.:
                        raise ValueError('Frozen positive witness did not replay')
            seen.append(dict(result['result'], rec=rec, seed=seed, source=str(d['sites'][ix[0]])))
            if len(seen) % 10 == 0 or len(seen) == 102:
                beat(seed=seed, completed=len(seen))
    totals = {a: {s: sum(r[a]['status'] == s for r in seen) for s in STATES} for a in ARMS}
    if len(seen) != 102 or switches != 9675 or totals['native'] != {
            s: old['risk_capacity']['full_forecast_bank'][s] for s in STATES}:
        raise ValueError('Original roster, counts, or native summary changed')
    groups = sorted({r['source'] for r in seen})
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    for a in ARMS[1:]:
        found = [r for r in seen if r[a]['status'] == POS and r['native']['status'] != POS]
        cells = [[np.mean([100.*((r[a]['status'] == POS)-(r['native']['status'] == POS))
                           for r in seen if (r['source'], r['seed']) == (g, s)]) for s in prev.SEEDS] for g in groups]
        contrasts[a] = dict(new_verified_views=len(found), new_recordings=len({r['rec'] for r in found}),
            new_sources=len({r['source'] for r in found}), source_balanced_verified_rate_pp=bootstrap(cells, draws))
    return dict(capacity=totals, contrasts_vs_native=contrasts, views=len(seen), matched_switches=switches,
                external_access=False, independent_confirmation=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    args = p.parse_args()
    cut, old, frozen = register()
    if args.phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the diagnostic before capacity scoring')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'replay':
            prev.receipt(PUBLIC, 'score')

        def beat(**v):
            event = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(score(run.population(), run.register(), lab, cut, old, frozen,
                                      replay=args.phase == 'replay', beat=beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Capacity witness replay changed')


if __name__ == '__main__':
    main()
