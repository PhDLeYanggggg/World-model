"""Check both strong references in the exact-count development capacity audit."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_exact_count_joint_capacity as parent
from src.evaluation.m3w_dual_reference_capacity import action_masses, solve
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import policy

run, prev, np = parent.run, parent.prev, parent.np
PUBLIC = parent.PUBLIC.parent/'native_dual_reference_capacity_v1'
PRIVATE = parent.PRIVATE.parent/'native_dual_reference_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(solve.__code__.co_filename), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_dual_reference_capacity.py')


def register():
    cut = parent.register()
    old = prev.receipt(parent.PUBLIC, 'replay')
    sources = (*FILES, parent.REG, parent.PUBLIC/'replay_execution.json',
               prev.PUBLIC/'freeze_execution.json', Path(comparison_masses.__code__.co_filename),
               Path(policy.__code__.co_filename))
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in sources], cut=cut,
        seconds=[3., 12.], positive_tolerance_fraction=1e-7, risk=.02, seeds=prev.SEEDS,
        exact_query_counts=True, objective='maximin_conservative_gain_cv_ols4',
        independent_confirmation=False, external_access=False)))
    return cut, old


def evidence(cv, ols, bank, y, valid, queries, rec, cut, inverse, counts, weight, rows, arms):
    if len(set(rows)) != len(rows):
        raise ValueError('Solver selected two paths for one person')
    pred = cv.copy()
    pred[rows] = bank[rows, arms]
    changed = (pred != cv).any(axis=(1, 2))
    if not np.array_equal(np.bincount(inverse[changed], minlength=len(counts)), counts):
        raise ValueError('Solver did not reproduce genuine query counts')
    risk = policy(pred, cv, cv, y, valid, np.full(len(cv), rec),
                  easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][rec]
    vals = [comparison_masses(p, pred, cv, y, valid, weight) for p in (cv, ols)]
    gains = [v['known_delta']-v['unknown_radius'] for v in vals]
    return dict(supported=risk['supported'], gains=gains, denominator=vals[0]['reference'])


def assess(cv, ols, bank, y, valid, queries, rec, cut, inverse, counts, weight):
    m = action_masses(cv, ols, bank, y, valid, weight, cut['easy'])
    tol = 1e-7*m['scale']
    result = {}
    for seconds in (3., 12.):
        take, s = solve(m, inverse, counts, seconds)
        result = dict(s, status='unresolved', solve_seconds_budget=seconds)
        if take is not None:
            rows, arms = m['row'][take], m['arm'][take]
            actual = evidence(cv, ols, bank, y, valid, queries, rec, cut,
                              inverse, counts, weight, rows, arms)
            np.testing.assert_allclose(actual['gains'], m['offset']+m['utility'][:, take].sum(1),
                                       rtol=1e-10, atol=tol)
            if actual['supported'] and min(actual['gains']) > tol:
                result.update(status='verified_both', evidence=actual,
                              witness=[rows.tolist(), arms.tolist()])
                break
        if s['infeasible']:
            result.update(status='risk_count_infeasible')
            break
        if s['upper_gain_mass'] is not None and s['upper_gain_mass'] < -tol:
            result.update(status='negative_maximin_upper_bound')
            break
    return result


def score(d, refs, lab, cut, frozen, old, *, replay, beat):
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    found = []
    switches = 0
    for cell in frozen['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            nominee, selected = z['nominee'], z['choice'][0] >= 0
        nominated = run.model.selected_paths(nominee, bank, cv)[:, None]
        for rec, (ix, _) in records.items():
            q = d['queries'][ix]
            _, inverse, sizes = np.unique(q, return_inverse=True, return_counts=True)
            counts = np.bincount(inverse[selected[ix]], minlength=len(sizes))
            np.testing.assert_array_equal(counts, np.minimum(2, sizes-1))
            weight = 1./sizes[inverse]
            switches += int(counts.sum())
            floor, ols = cv[ix], bank[ix, run.NAMES.index(parent.OLS)]
            local_banks = (nominated[ix], bank[ix])
            mark = PRIVATE/f'{seed}_{rec}.json'
            binding = dict(registration=run.sha(REG), frozen=cell['artifact']['sha256'])
            if not mark.exists() and not replay:
                computed = {key: assess(floor, ols, b, y[ix], valid[ix], q, rec, cut, inverse, counts, weight)
                            for key, b in zip(parent.ARMS, local_banks, strict=True)}
                run.once(mark, run.stable(dict(binding=binding, result=computed)))
            data = run.read(mark)
            if data['binding'] != binding:
                raise ValueError('Incompatible recording capacity checkpoint')
            for key, b in zip(parent.ARMS, local_banks, strict=True):
                p = data['result'][key]
                if p['status'] == 'verified_both':
                    r, a = (np.asarray(x, dtype=int) for x in p['witness'])
                    check = evidence(floor, ols, b, y[ix], valid[ix], q, rec, cut, inverse, counts, weight, r, a)
                    if check != p['evidence'] or not check['supported'] or min(check['gains']) <= 0.:
                        raise ValueError('Recording witness does not re-score exactly')
            fixed, full = (data['result'][a] for a in parent.ARMS)
            if fixed['status'] == 'verified_both' and full['status'] != 'verified_both':
                if full['status'] != 'unresolved':
                    raise ValueError('Full bank cannot lose a verified nomination witness')
                data['result'][parent.ARMS[1]] = dict(fixed, inherited=True)
            found.append(data['result'])
            if len(found) % 10 == 0 or len(found) == 102:
                beat(seed=seed, completed=len(found))
    if len(found) != 102 or switches != 9675:
        raise ValueError('Missing registered recording views or genuine switches')
    states = ('verified_both', 'risk_count_infeasible', 'negative_maximin_upper_bound', 'unresolved')
    rows = {}
    for key in parent.ARMS:
        vals = [v[key] for v in found]
        p = {s: sum(v['status'] == s for v in vals) for s in states}
        prior = old['risk_capacity'][key]
        if p['risk_count_infeasible'] > prior['proved_infeasible']:
            raise ValueError('Maximin objective cannot remove feasible risk/count solutions')
        rows[key] = dict(p, previous_cv_maximizer_both=prior['positive_ols4_witness'],
                         inherited_witnesses=sum(v.get('inherited', False) for v in vals))
    return dict(views=len(found), matched_switches=switches, risk_capacity=rows, choices_exported=False,
                independent_confirmation=False, external_access=False, deployment=False,
                stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score', 'replay'))
    phase = p.parse_args().phase
    cut, old = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the design before scoring')
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
            raise ValueError('Capacity replay differs')


if __name__ == '__main__':
    main()
