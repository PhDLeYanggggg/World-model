"""Locate lost recording opportunity between retrieval, decoding, and the gate."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_recording_witness_retrieval as witness

run, np, cal = witness.run, witness.np, witness.cal
base = witness.prior.prior.g.model.base
PUBLIC = witness.PUBLIC
PRIVATE = witness.PRIVATE/'retention'
REG = PUBLIC/'retention_registration.json'
STAGES = ('retrieved_bank', 'decoded_queries', 'admitted_queries')


def register():
    cut, _ = witness.register()
    for k in ('fit', 'score', 'replay'):
        run.checked(run.read(PUBLIC/f'{k}_execution.json')['artifact'])
    parents = (Path(__file__), Path(base.__file__), witness.REG, PUBLIC/'analysis.json',
               PUBLIC/'replay_execution.json')
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], stages=STAGES,
        solver_seconds=4., original_risk=True, no_changes_to_frozen_choices=True, external_access=False)))
    return cut


def audit(cv, ols, bank, y, valid, choices, off, recording, cut):
    if not np.all((choices[1] == -1) | (choices[1] == choices[0])):
        raise ValueError('Gate changed an admitted action')
    slots = np.full((2, len(off)-1, 2), -1, int)
    paths = np.zeros((*slots.shape, *cv.shape[1:]))
    for j, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
        for a, c in enumerate(choices):
            at = lo+np.flatnonzero(c[lo:hi] >= 0)
            slots[a, j, :len(at)] = at
            paths[a, j, :len(at)] = bank[at, c[at]]
    p = base.prepare(cv, ols, y, valid, off, slots, paths, cut['easy'])
    results = []
    for arm in range(2):
        take, info = base.solve(p, 'ols4', (arm,), seconds=4.)
        feasible = positive = False
        if take is not None:
            pred = cv.copy()
            for k in p['query'][take]:
                use = slots[arm, k] >= 0
                pred[slots[arm, k, use]] = paths[arm, k, use]
            risk = base.policy(pred, cv, cv, y, valid, np.full(len(cv), recording),
                               easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][recording]
            m = base.comparison_masses(ols, pred, cv, y, valid, p['weights'])
            val = m['known_delta']-m['unknown_radius']
            np.testing.assert_allclose(val, info['witness_gain_mass'], atol=1e-7, rtol=1e-10)
            feasible = risk['supported']
            positive = feasible and val > 0.
        results.append(dict(feasible=feasible, positive=positive,
                            unresolved=not (info['proved_infeasible'] or (feasible and info['optimum_proved']))))
    return results


def readout(cut):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    cells = []
    for c in run.read(witness.PRIVATE/'freeze.json')['cells']:
        seed = c['seed']
        cv, ols, bank = witness.prior.bank_for(d, refs, seed)
        with np.load(run.checked(c['artifact']), allow_pickle=False) as f:
            np.testing.assert_array_equal(f['ids'], d['global_ids'])
            choices = f['chosen'][[1, 3]]
        for r, (ix, groups) in records.items():
            p = PRIVATE/f'{seed}_{r}.json'
            if not p.exists():
                loc = ix[np.concatenate(groups)]
                off = np.r_[0, np.cumsum(list(map(len, groups)))]
                old = run.read(witness.PRIVATE/f'capacity_{seed}_{r}.json')
                if old['registration'] != run.sha(witness.REG) or old['fit'] != c['artifact']['sha256']:
                    raise ValueError('Stale retrieved shortlist score')
                new = audit(cv[loc], ols[loc], bank[loc], y[loc], valid[loc], choices[:, loc], off, r, cut)
                prev = old['result']
                cell = [dict(feasible=prev['feasible'], positive=prev['positive'],
                    unresolved=not (prev['optimum_proved'] or prev['proved_infeasible'])), *new]
                for a, b in zip(cell, cell[1:]):
                    if (not a['unresolved'] and not a['feasible'] and b['feasible']) or (
                            not a['unresolved'] and not a['positive'] and b['positive']):
                        raise ValueError('A nested restriction increased the proved capacity')
                run.once(p, run.stable(dict(registration=run.sha(REG), stages=cell)))
            row = run.read(p)
            if row['registration'] != run.sha(REG):
                raise ValueError('Stale retention checkpoint')
            cells.append(row['stages'])
        beat = dict(pid=os.getpid(), seed=seed, completed=len(cells))
        run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', beat)
        print(beat, flush=True)
    if len(cells) != 102:
        raise ValueError('Incomplete stage comparison')
    return {k: {v: sum(c[i][v] for c in cells) for v in ('feasible', 'positive', 'unresolved')}
            for i, k in enumerate(STAGES)}


def main():
    cut = register()
    if sys.argv[1:] == ['register']:
        return
    if sys.argv[1:] != ['score']:
        raise ValueError('Use register or score')
    for p in (Path(__file__), REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit retention diagnostic before scoring')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = PUBLIC/'retention.json'
        run.once(out, run.stable(dict(stages=readout(cut), registration=run.sha(REG),
            choices_label_aware_only=True, external_data_opened=False, deployment_changed=False)))
        receipt = PUBLIC/'retention_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Retention result differs from first execution')


if __name__ == '__main__':
    main()
