"""Resume only unresolved all-recipient solves without changing their objective."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_recipient_budget_capacity as prior

run, np, cal = prior.run, prior.np, prior.cal
PUBLIC = prior.PUBLIC
REG = PUBLIC/'refinement_registration.json'
SECONDS = 20.


def prune(cv, ols, paths, valid):
    """A wholly unknown switch needs positive paired value and an undominated radius."""
    allow = np.ones(paths.shape[:2], dtype=bool)
    at = ~valid.any(1)
    a, b = cv[at], ols[at]
    before = np.linalg.norm(a-b, axis=-1).max(1)
    after = np.linalg.norm(paths[at]-b[:, None], axis=-1).max(2)
    radius = np.linalg.norm(paths[at]-a[:, None], axis=-1).max(2)
    keep = (after < before[:, None]) & (radius > 0.)
    for j in range(paths.shape[1]):
        better = (after <= after[:, j, None]) & (radius <= radius[:, j, None])
        strict = (after < after[:, j, None]) | (radius < radius[:, j, None])
        earlier = np.arange(paths.shape[1]) < j
        keep[:, j] &= ~((better & (strict | earlier)) & (radius > 0.)).any(1)
    allow[at] = keep
    return allow


def main():
    cut, ctrl = prior.register()
    p = PUBLIC/'analysis.json'
    run.checked(run.read(PUBLIC/'execution.json')['artifact'])
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in (Path(__file__), p, prior.REG)],
        seconds=SECONDS, rerun='only_previous_unresolved_all_recipients',
        pruning='unknown_options_dominated_by_abstention_or_same_recipient', external_access=False)))
    if sys.argv[1:] == ['register']:
        return
    if sys.argv[1:] != ['score']:
        raise ValueError('Use register or score')
    for f in (Path(__file__), REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit refinement before running it')
    private = prior.PRIVATE/'refinement'
    private.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (private/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d, refs = run.population(), run.register()
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
        rows, rerun = [], 0
        for c in ctrl['cells']:
            seed = c['seed']
            cv, bank = run.candidates(d, refs, seed)
            with np.load(run.checked(c['cache']), allow_pickle=False) as f:
                coef = dict(zip(f['sources'], f['excluded_source_coefficients']))
            for r, (ix, groups) in records.items():
                old = run.read(prior.PRIVATE/f'{seed}_{r}.json')
                if old['registration'] != run.sha(prior.REG):
                    raise ValueError('Incompatible initial result')
                v = old['outcomes']['all_recipients']
                if not (v['proved_infeasible'] or v['optimum_proved']):
                    rerun += 1
                    out = private/f'{seed}_{r}.json'
                    if not out.exists():
                        loc = ix[np.concatenate(groups)]
                        offs = np.r_[0, np.cumsum(list(map(len, groups)))]
                        paths, cap, _ = prior.query_design(d, loc, offs, cv[loc], bank[loc], coef[old['source']])
                        ols = paths[:, run.NAMES.index(cal.OLS)]
                        y, valid = lab['target_eval'][loc]-d['origin'][loc, None], lab['valid'][loc]
                        mask = prune(cv[loc], ols, paths, valid)
                        result = prior.model.solve(cv[loc], ols, paths, y, valid, offs, cap, mask, r,
                                                   cut['easy'], cut['hard'], SECONDS)
                        if result['proved_infeasible'] and (v['feasible'] or old['outcomes']['same_recipients']['feasible']):
                            raise ValueError('Exact pruning removed a feasible solution')
                        run.once(out, run.stable(dict(registration=run.sha(REG), result=result)))
                    saved = run.read(out)
                    if saved['registration'] != run.sha(REG):
                        raise ValueError('Stale refinement output')
                    result = saved['result']
                    candidates = [a for a in (v, result, old['outcomes']['same_recipients']) if a['feasible']]
                    best = max(candidates, key=lambda a: a['witness_mass'], default=None)
                    if best and result['upper_mass'] is not None and best['witness_mass'] > result['upper_mass']+1e-5:
                        raise ValueError('Upper bound below a previously verified witness')
                    if best:
                        result = dict(result, **{k: best[k] for k in ('feasible', 'positive', 'witness_mass')})
                    old['outcomes']['all_recipients'] = result
                    beat = dict(pid=os.getpid(), seed=seed, refined_views=rerun, status=result['status'])
                    run.kin.prior.source.atomic(private/'heartbeat.json', beat)
                    print(beat, flush=True)
                rows.append(old)
        if len(rows) != 102 or rerun != run.read(p)['summary']['slates']['all_recipients']['unresolved']:
            raise ValueError('Incomplete initial or refinement roster')
        dest = PUBLIC/'refinement.json'
        run.once(dest, run.stable(dict(registration=run.sha(REG), refined_views=rerun, summary=prior.summarize(rows),
            source='fresh_solver_refinement_of_previous_unresolved_views', deployment=False,
            external_data_opened=False, independent_confirmation=False)))
        receipt = PUBLIC/'refinement_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(dest), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(dest):
            raise ValueError('Refinement result differs from prior execution')


if __name__ == '__main__':
    main()
