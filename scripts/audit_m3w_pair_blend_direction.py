"""Check continuous query-local pair headroom without learning another gate."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_completion_motion_blend as blend
from src.evaluation.m3w_pair_blend_direction import evaluate, TOL

run, prev, np, pair = blend.run, blend.prev, blend.np, blend.pair
PUBLIC = blend.PUBLIC.parent/'native_pair_blend_direction_v1'
PRIVATE = blend.PRIVATE.parent/'native_pair_blend_direction_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(evaluate.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_pair_blend_direction.py')
BASE, OLS, GAIN, REF = blend.BASE, blend.OLS, blend.GAIN, blend.REF
SHARED, SEPARATE = 'shared_hindsight', 'separate_hindsight'
ARMS = (SHARED, SEPARATE)
STATES = ('verified_gain', 'no_gain_to_tolerance', 'unresolved')
PAIRS = {f'{b}_vs_{a}': (a, b, BASE) for a, b in (
    (REF, SHARED), (REF, SEPARATE), (SHARED, SEPARATE), (GAIN, SEPARATE), (OLS, SEPARATE), (BASE, SEPARATE))}


def register():
    cut, _, _, frozen = blend.register()
    old = prev.receipt(blend.PUBLIC, 'replay')
    if old['common_substitutions'] != 9675:
        raise ValueError('Expected the same frozen pair-blend cohort')
    bindings = (*FILES, blend.REG, blend.PUBLIC/'replay_execution.json',
                blend.relative.PUBLIC/'infer_execution.json', Path(pair.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cut=cut, tolerance=TOL,
        arms=ARMS, seeds=prev.SEEDS, exact_recipients=True, no_new_training=True, external_access=False)))
    return cut, frozen


def score(cut, frozen, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    rows, cells, jointly_only = [], {}, 0
    counts = {arm: dict.fromkeys(STATES, 0) for arm in ARMS}
    active_queries, active_switches = 0, 0
    for c in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        a, b = (x.astype(float) for x in blend.anchors(d, frozen, seed))
        with np.load(run.checked(c['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice = z['choice'][0]
        stat = {g: [] for g in groups}
        for rec, (ix, queries) in records.items():
            floor, local = cv[ix], bank[ix]
            gain = run.model.selected_paths(choice[ix], local, floor)
            switch = (gain != floor).any(axis=(1, 2))
            active_switches += int(switch.sum())
            if any((np.linalg.norm(p[ix][switch]-floor[switch], axis=2).max(1) <= 1e-6).any() for p in (a, b)):
                raise ValueError('Expected both previous nonidentity proposals on every original recipient')
            reference = np.where(switch[:, None, None], a[ix], gain)
            proposal = np.where(switch[:, None, None], b[ix], gain)
            paths = {BASE: floor, OLS: local[:, run.NAMES.index(OLS)], GAIN: gain, REF: reference,
                     SHARED: reference.copy(), SEPARATE: reference.copy()}
            for q in queries:
                active = switch[q]
                if not active.any():
                    continue
                active_queries += 1
                h, y, v = d['history'][ix[q]], target[ix[q]], valid[ix[q]]
                r = reference[q]
                answers = evaluate(h, r, proposal[q], floor[q], y, v, active, pair.local_edges)
                for arm, (state, improvement, p) in zip(ARMS, answers, strict=True):
                    counts[arm][state] += 1
                    paths[arm][q] = p
                    if state == 'verified_gain':
                        _, check, _, _ = pair.query(h, np.stack((r, p)), y, v, [(0, 1)])
                        np.testing.assert_allclose(improvement, check[0, 1], atol=1e-8)
                        if check[0, 1] <= TOL:
                            raise ValueError('Witness does not improve the original pair lower score')
                if answers[0][0] == 'no_gain_to_tolerance' and answers[1][0] == 'verified_gain':
                    jointly_only += 1
                stat[roster[rec]].append([float(s == 'verified_gain') for s, _, _ in answers] +
                                         [answers[0][1], answers[1][1]])
            for arm in ARMS:
                np.testing.assert_array_equal((paths[arm] != floor).any(axis=(1, 2)), switch)
            out = run.parent.readout.score_recording(paths, PAIRS, target[ix], valid[ix], d['queries'][ix], rec,
                                                    easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(out, seed=seed, source=roster[rec]))
        for g in groups:
            if not stat[g]:
                raise ValueError('A whole source has no active-query support')
            cells[g, seed] = np.mean(stat[g], axis=0)
        beat(state='directional_witnesses_scored', seed=seed, active_queries=active_queries, views=len(rows))
    grouped = prev.old.prior.grouped_readout(rows, roster, prev.SEEDS)
    if active_switches != 9675 or grouped['risk'][GAIN] != prev.receipt(prev.PUBLIC, 'replay')['grouped']['risk'][GAIN]:
        raise ValueError('The fixed-recipient original scorer changed')
    if any(sum(x.values()) != active_queries for x in counts.values()):
        raise ValueError('Incomplete directional classification')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    grid = np.asarray([[cells[g, s] for s in prev.SEEDS] for g in groups])
    result = {a: dict(verified_rate_percent=pair.bootstrap(100.*grid[:, :, i], draw),
                      witnessed_gain_per_active_query_px=pair.bootstrap(grid[:, :, 2+i], draw))
              for i, a in enumerate(ARMS)}
    return dict(active_queries=active_queries, matched_switches=active_switches, counts=counts,
                separate_verified_with_shared_no_gain=jointly_only, active_query_bootstrap=result,
                recording_readout=grouped, hindsight_only=True, independent_confirmation=False,
                external_access=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    args = p.parse_args()
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
            raise ValueError('Directional witness replay differs')


if __name__ == '__main__':
    main()
