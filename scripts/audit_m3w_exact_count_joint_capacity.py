"""Check joint risk feasibility at the exact observed per-query budgets."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recipient_shortfall as prev
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_recording_switch_ceiling import matched_count, moments, policy

run, np, BASE, OLS = prev.run, prev.np, prev.BASE, prev.OLS
PUBLIC = prev.PUBLIC.parent/'native_exact_count_joint_capacity_v1'
PRIVATE = prev.PRIVATE.parent/'native_exact_count_joint_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_exact_count_joint_capacity.py')
ARMS = ('fixed_nomination', 'full_forecast_bank')


def register():
    cut = prev.register()
    prev.receipt(prev.PUBLIC, 'replay')
    parents = (prev.REG, prev.PUBLIC/'freeze_execution.json', prev.PUBLIC/'replay_execution.json',
               Path(prev.__file__), Path(matched_count.__code__.co_filename),
               Path(comparison_masses.__code__.co_filename), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in parents], arms=ARMS, seeds=prev.SEEDS,
        risk=.02, exact_query_counts=True, objective='cv_conservative_gain', seconds=[3., 12.],
        external_access=False)))
    return cut


def dominant_paths(cv, bank, y, valid, cut):
    best = np.full(len(cv), -np.inf)
    pred = cv.copy()
    for arm in range(bank.shape[1]):
        value = moments(cv, bank[:, arm], y, valid, cut)
        take = (value['radius'] > 0.) & (value['utility'] > best)
        pred[take], best[take] = bank[take, arm], value['utility'][take]
    return pred


def checked_outcome(cv, ols, path, y, valid, queries, recording, cut, weight, seconds):
    m = moments(cv, path, y, valid, cut['easy'])
    take, solver = matched_count(m, queries, weight, seconds)
    out = dict(solver=solver, verified=False, positive_cv=False, positive_ols4_witness=False)
    if take is not None:
        pred = np.where(take[:, None, None], path, cv)
        check = policy(pred, cv, cv, y, valid, np.full(len(cv), recording),
                       easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][recording]
        if check['supported']:
            lower = [comparison_masses(r, pred, cv, y, valid, weight) for r in (cv, ols)]
            out.update(verified=True, positive_cv=lower[0]['known_delta'] > lower[0]['unknown_radius'],
                       positive_ols4_witness=lower[1]['known_delta'] > lower[1]['unknown_radius'])
        else:
            out['solver']['optimal'] = False
            out['postcheck_failed'] = True
    return out


def assess(cv, ols, bank, nominee, selected, y, valid, queries, recording, cut):
    _, inv, count = np.unique(queries, return_inverse=True, return_counts=True)
    k = np.bincount(inv[selected], minlength=len(count))
    np.testing.assert_array_equal(k, np.minimum(2, count-1))
    weight = 1./count[inv]
    nominees = run.model.selected_paths(nominee, bank, cv)
    result = {}
    for key, p in zip(ARMS, (nominees, dominant_paths(cv, bank, y, valid, cut['easy']))):
        r = checked_outcome(cv, ols, p, y, valid, queries, recording, cut, weight, 3.)
        refined = not (r['verified'] or r['solver']['infeasible'])
        if refined:
            r = checked_outcome(cv, ols, p, y, valid, queries, recording, cut, weight, 12.)
        result[key] = dict(r, refined=refined)
    if result[ARMS[0]]['verified'] and result[ARMS[1]]['solver']['infeasible']:
        raise ValueError('Dominance reduction lost a checked fixed-nomination witness')
    return result


def score(d, refs, lab, cut, frozen, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    seen = []
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            nominee, selected = z['nominee'], z['choice'][0] >= 0
        for rec, (ix, _) in records.items():
            checkpoint = PRIVATE/f'{seed}_{rec}.json'
            if not checkpoint.exists():
                result = assess(cv[ix], bank[ix, run.NAMES.index(OLS)], bank[ix], nominee[ix],
                                selected[ix], y[ix], valid[ix], d['queries'][ix], rec, cut)
                run.once(checkpoint, run.stable(dict(registration=run.sha(REG), forecast=cell['artifact']['sha256'],
                    result=result)))
            info = run.read(checkpoint)
            if info['registration'] != run.sha(REG) or info['forecast'] != cell['artifact']['sha256']:
                raise ValueError('Incompatible capacity checkpoint')
            seen.append(info['result'])
            if len(seen) % 10 == 0 or len(seen) == 102:
                beat(seed=seed, completed=len(seen))
    if len(seen) != 102:
        raise ValueError('Require all recording-seed views')
    result = {}
    for key in ARMS:
        rows = [s[key] for s in seen]
        result[key] = dict(verified_feasible=sum(r['verified'] for r in rows),
            proved_infeasible=sum(r['solver']['infeasible'] for r in rows),
            unresolved=sum(not (r['verified'] or r['solver']['infeasible']) for r in rows),
            cv_optimum_proved=sum(r['solver']['optimal'] for r in rows),
            positive_cv_witness=sum(r['positive_cv'] for r in rows),
            positive_ols4_witness=sum(r['positive_ols4_witness'] for r in rows),
            refined=sum(r['refined'] for r in rows))
    return dict(views=len(seen), risk_capacity=result, no_hindsight_choices_exported=True,
                independent_confirmation=False, external_access=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the design before execution')
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
        run.once(out, run.stable(score(d, refs, lab, cut, frozen, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-t, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Capacity replay differs')


if __name__ == '__main__':
    main()
