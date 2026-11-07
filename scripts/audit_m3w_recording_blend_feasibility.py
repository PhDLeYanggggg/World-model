"""Bound recording-level feasibility of fixed-recipient motion blends."""
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
from scripts import audit_m3w_pair_blend_direction as direction
from src.evaluation.m3w_recording_blend_feasibility import audit, safe_segment

run, prev, blend, np, pair = direction.run, direction.prev, direction.blend, direction.np, direction.pair
PUBLIC = blend.PUBLIC.parent/'native_recording_blend_feasibility_v1'
PRIVATE = blend.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(audit.__code__.co_filename),
         PUBLIC/'protocol.md', ROOT/'tests/test_m3w_recording_blend_feasibility.py')
BASE, OLS, GAIN, REF = blend.BASE, blend.OLS, blend.GAIN, blend.REF
SAFE, JOINT = 'individual_safe_ceiling', 'pair_safe_ray'
ARMS = (BASE, OLS, REF, SAFE, JOINT)
COMPARISONS = ((SAFE, JOINT), (REF, JOINT), (BASE, JOINT), (OLS, JOINT))
PAIRS = {f'{b}_vs_{a}': (a, b, BASE) for a, b in COMPARISONS}


def register():
    cut, frozen = direction.register()
    prev.receipt(direction.PUBLIC, 'replay')
    bindings = (*FILES, direction.REG, direction.PUBLIC/'replay_execution.json',
                blend.relative.PUBLIC/'infer_execution.json',
                Path(run.parent.readout.__file__),
                ROOT/'src/evaluation/m3w_selection_forecast_readout.py',
                ROOT/'src/evaluation/m3w_recording_switch_ceiling.py')
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cut=cut,
        seeds=prev.SEEDS, risk_budget=.02, fixed_recipients=True, external_access=False,
        pair_ray_grid=[0., .25, .5, .75, 1.], future_labels_used_only_for_hindsight=True)))
    return cut, frozen


def geometry(queries, history, y, v, paths, comparisons):
    table = [pair.query(history[q], np.stack([p[q] for p in paths]), y[q], v[q], comparisons)[1]
             for q in queries]
    return pair.mean_finite(table)


def fit_ray(queries, h, y, v, cv, ref, b, a, mask, rec, cut):
    candidate = ref.copy()
    for q in queries:
        if mask[q].any():
            candidate[q] = direction.evaluate(h[q], ref[q], b[q], cv[q], y[q], v[q],
                                              mask[q], pair.local_edges)[1][2]
    t, _ = safe_segment(a, candidate, cv, y, v, mask, rec, cut['easy'], cut['hard'])
    best = (0., 0., a)
    for f in (.25, .5, .75, 1.):
        p = a+(f*t)*(candidate-a)
        if not np.array_equal((p != cv).any((1, 2)), mask):
            continue
        score = run.parent.readout.policy(p, cv, cv, y, v, np.full(len(cv), rec),
                                          easy_cut=cut['easy'], hard_cut=cut['hard'])
        if not score['recordings'][rec]['supported']:
            continue
        val = geometry(queries, h, y, v, (a, p), [(0, 1)])[0, 1]
        if np.isfinite(val) and val > best[0]:
            best = (float(val), f*t, p)
    return best


def score(cut, frozen, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    state, witnesses, counts, exclusions = [], [], Counter(), Counter()
    switches = 0
    expected = prev.receipt(direction.PUBLIC, 'replay')['matched_switches']
    for c in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        a, b = (x.astype(float) for x in blend.anchors(d, frozen, seed))
        with np.load(run.checked(c['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice = z['choice'][0]
        for rec, (ix, queries) in records.items():
            floor, y, v, h = cv[ix], target[ix], valid[ix], d['history'][ix]
            ols = bank[ix, run.NAMES.index(OLS)]
            gain = run.model.selected_paths(choice[ix], bank[ix], floor)
            active = (gain != floor).any((1, 2))
            switches += int(active.sum())
            left = np.where(active[:, None, None], a[ix], gain)
            right = np.where(active[:, None, None], b[ix], gain)
            info, witness = audit(floor, left, right, y, v, active, rec, cut['easy'], cut['hard'])
            counts[info['status']] += 1
            if info.get('unknown_radius_alone_excludes_easy'):
                exclusions['unknown_radius_alone_excludes_easy'] += 1
            for name, failed in zip(('all', 'easy', 'signed_easy'), info.get('excluding_channels', ())):
                exclusions[name] += int(failed)
            state.append(dict(recording=rec, seed=seed, **info))
            if witness is None:
                continue
            utility, t, joint = fit_ray(queries, h, y, v, floor, left, right, witness, active, rec, cut)
            paths = dict(zip(ARMS, (floor, ols, left, witness, joint)))
            readout = run.parent.readout.score_recording(paths, PAIRS, y, v, d['queries'][ix], rec,
                                                        easy_cut=cut['easy'], hard_cut=cut['hard'])
            indices = [(ARMS.index(a), ARMS.index(b)) for a, b in COMPARISONS]
            geom = geometry(queries, h, y, v, tuple(paths.values()), indices)
            if not all(readout['metrics'][arm]['risk']['supported'] for arm in (SAFE, JOINT)):
                raise ValueError('Feasible witness fails original recording rule')
            witnesses.append(dict(recording=rec, seed=seed, pair_ray_fraction=t, pair_gain_vs_safe_px=utility,
                pair_grid_ade_lower_px={k: float(x) if np.isfinite(x) else None for k, x in zip(PAIRS, geom[:, 1])},
                conservative_cv_gain_percent={arm: readout['metrics'][arm]['completion_lower_gain_floor_percent']
                                               for arm in (SAFE, JOINT)}))
        beat(state='feasibility_scored', seed=seed, views=len(state), statuses=dict(counts))
    if switches != expected or len(state) != 102:
        raise ValueError('Original switch population changed')
    return dict(views=len(state), switches=switches, counts=dict(counts), exclusion_channels=dict(exclusions),
        feasible_with_positive_pair_ray=sum(x['pair_gain_vs_safe_px'] > 1e-9 for x in witnesses),
        feasible_with_positive_pair_vs_cv=sum((x['pair_grid_ade_lower_px'][f'{JOINT}_vs_{BASE}'] or 0.) > 1e-9
                                              for x in witnesses),
        feasible_with_positive_pair_vs_ols=sum((x['pair_grid_ade_lower_px'][f'{JOINT}_vs_{OLS}'] or 0.) > 1e-9
                                               for x in witnesses),
        by_recording_seed=state, feasible_witnesses=witnesses,
        pair_search_global=False, deployable=False, hindsight_only=True, independent_confirmation=False,
        external_access=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'score', 'replay'))
    args = parser.parse_args()
    cut, frozen = register()
    if args.phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the diagnostic before scoring')
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

        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(score(cut, frozen, beat)))
        mark = PUBLIC/f'{args.phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Recording feasibility replay differs')


if __name__ == '__main__':
    main()
