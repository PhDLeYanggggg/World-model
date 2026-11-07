"""Test new past-only motion shapes with fixed recipients and exact counts."""
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
from src.evaluation.m3w_damped_motion_options import HALF_LIVES, options, fixed_recipient_paths
from src.evaluation.m3w_joint_path_audit import bootstrap
from src.evaluation.m3w_ols_blend_capacity import prune_unknown

run, prev, np = dual.run, dual.prev, dual.np
PUBLIC = dual.PUBLIC.parent/'native_damped_motion_v1'
PRIVATE = dual.PRIVATE.parent/'native_damped_motion_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(options.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_damped_motion_options.py')
ARMS = ('native', 'add_speed', 'add_speed_and_turn')
CONTROL, SPEED, TURN = 'gain', 'speed_trend', 'speed_and_turn'
BASE, OLS = dual.parent.BASE, dual.parent.OLS
PAIRS = {f'{a}_vs_{b}': (b, a, BASE) for a, b in
         ((SPEED, CONTROL), (TURN, SPEED), (TURN, CONTROL), (SPEED, OLS), (TURN, OLS),
          (SPEED, BASE), (TURN, BASE))}
STATUS = ('verified_both', 'risk_count_infeasible', 'negative_maximin_upper_bound', 'unresolved')


def register():
    cut, _ = dual.register()
    old = prev.receipt(dual.PUBLIC, 'replay')
    original = prev.receipt(prev.PUBLIC, 'replay')
    refs = (*FILES, dual.REG, dual.PUBLIC/'replay_execution.json', Path(dual.__file__),
            prev.PUBLIC/'replay_execution.json', Path(prune_unknown.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in refs], cuts=cut, arms=ARMS,
        seeds=prev.SEEDS, half_lives=HALF_LIVES, fixed_recipient_pairs=PAIRS,
        risk=.02, exact_query_counts=True, external_access=False)))
    return cut, old, original


def summarize(rows, actual, old, original, roster):
    if len(rows) != 102 or sum(r['switches'] for r in rows) != 9675:
        raise ValueError('Expected all original matched-count recording views')
    capacity = {a: {s: sum(r[a]['status'] == s for r in rows) for s in STATUS} for a in ARMS}
    if capacity['native'] != {k: old['risk_capacity']['full_forecast_bank'][k] for k in STATUS}:
        raise ValueError('Original native capacity did not replay')
    groups = sorted(set(roster.values()))
    draws = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    for a, b in zip(ARMS[:-1], ARMS[1:]):
        novel = [r for r in rows if r[b]['status'] == 'verified_both' and r[a]['status'] != 'verified_both']
        cells = [[np.mean([100.*((r[b]['status'] == 'verified_both')-(r[a]['status'] == 'verified_both'))
                           for r in rows if (r['source'], r['seed']) == (g, s)])
                  for s in prev.SEEDS] for g in groups]
        contrasts[b+'_vs_'+a] = dict(new_views=len(novel), new_recordings=len({r['recording'] for r in novel}),
            new_sources=len({r['source'] for r in novel}), source_balanced_verified_rate_pp=bootstrap(cells, draws))
    paired = prev.old.prior.grouped_readout(actual, roster, prev.SEEDS)
    if paired['risk'][CONTROL] != original['grouped']['risk'][CONTROL]:
        raise ValueError('Original causal control did not replay')
    return dict(views=len(rows), matched_switches=9675, risk_capacity=capacity, capacity_contrasts=contrasts,
        past_only=paired, path_substitutions={a: sum(r['substitutions'][a] for r in rows) for a in (SPEED, TURN)},
        new_training=False, external_access=False, independent_confirmation=False, deployment=False,
        stage5c_executed=False, smc_enabled=False)


def score(d, refs, lab, cut, frozen, old, original, *, replay, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows, actual = [], []
    for f in frozen['cells']:
        seed = f['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(f['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choices = z['choice'][0]
        for rec, (ix, _) in records.items():
            q = d['queries'][ix]
            floor, base = cv[ix], bank[ix]
            ols = base[:, run.NAMES.index(OLS)]
            straight, turning, has_speed, has_turn = options(d['history'][ix], ols, floor)
            origin = run.model.selected_paths(choices[ix], base, floor)
            speed, turn = fixed_recipient_paths(origin, floor, straight[:, 0], turning[:, 0], has_speed, has_turn)
            selected = (origin != floor).any(axis=(1, 2))
            for p in (speed, turn):
                np.testing.assert_array_equal((p != floor).any(axis=(1, 2)), selected)
            substitutions = {SPEED: int((speed != origin).any(axis=(1, 2)).sum()),
                             TURN: int((turn != speed).any(axis=(1, 2)).sum())}
            _, inv, sizes = np.unique(q, return_inverse=True, return_counts=True)
            counts = np.bincount(inv[selected], minlength=len(sizes))
            np.testing.assert_array_equal(counts, np.minimum(2, sizes-1))
            w = 1./sizes[inv]
            y, mask = target[ix], valid[ix]
            read = run.parent.readout.score_recording({BASE: floor, OLS: ols, CONTROL: origin, SPEED: speed, TURN: turn},
                                                       PAIRS, y, mask, q, rec, easy_cut=cut['easy'], hard_cut=cut['hard'])
            actual.append(dict(read, seed=seed, source=roster[rec]))

            menus = {ARMS[0]: base}

            def menu(a):
                if a not in menus:
                    k = ARMS.index(a)
                    menus[a] = np.concatenate((menu(ARMS[k-1]), straight if k == 1 else turning), axis=1)
                return menus[a]

            p = dual.PRIVATE/f'{seed}_{rec}.json'
            parent = run.read(p)
            if parent['binding'] != dict(registration=run.sha(dual.REG), frozen=f['artifact']['sha256']):
                raise ValueError('Original capacity recording binding differs')
            native = dict(parent['result']['full_forecast_bank'], menu=ARMS[0])
            if native.get('inherited') or native['status'] == STATUS[-1]:
                raise ValueError('Expected resolved native capacity')
            binding = dict(registration=run.sha(REG), parent=run.sha(p))
            mark = PRIVATE/p.name
            if not mark.exists() and not replay:
                found = {ARMS[0]: native}
                for a, b in zip(ARMS[:-1], ARMS[1:]):
                    if found[a]['status'] == STATUS[0]:
                        found[b] = found[a]
                        continue
                    result = dual.assess(floor, ols, prune_unknown(floor, ols, menu(b), mask),
                                         y, mask, q, rec, cut, inv, counts, w)
                    if result['status'] == STATUS[0] and found[a]['status'] != STATUS[-1]:
                        if max(result['witness'][1]) < menu(a).shape[1]:
                            raise ValueError('New witness cannot use only the smaller menu')
                    if result['status'] == STATUS[1] and found[a]['status'] == STATUS[2]:
                        raise ValueError('Expanded menu cannot lose risk feasibility')
                    found[b] = dict(result, menu=b)
                run.once(mark, run.stable(dict(binding=binding, capacity=found)))
            cache = run.read(mark)
            if cache['binding'] != binding or cache['capacity'][ARMS[0]] != native:
                raise ValueError('Capacity checkpoint binding differs')
            for item in cache['capacity'].values():
                if item['status'] == STATUS[0]:
                    r, c = (np.asarray(v, dtype=int) for v in item['witness'])
                    val = dual.evidence(floor, ols, menu(item['menu']), y, mask, q, rec, cut, inv, counts, w, r, c)
                    if val != item['evidence'] or not val['supported'] or min(val['gains']) <= 0.:
                        raise ValueError('Capacity witness did not re-score exactly')
            rows.append(dict(cache['capacity'], recording=rec, source=roster[rec], seed=seed,
                             switches=int(counts.sum()), substitutions=substitutions))
            if len(rows) % 10 == 0 or len(rows) == 102:
                beat(seed=seed, completed=len(rows))
    return summarize(rows, actual, old, original, roster)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, old, original = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the experiment before real-data scoring')
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
        run.once(out, run.stable(score(d, refs, lab, cut, frozen, old, original,
                                      replay=phase == 'replay', beat=beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Damped-motion replay differs')


if __name__ == '__main__':
    main()
