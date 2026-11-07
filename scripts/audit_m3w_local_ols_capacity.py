"""Compare the frozen local OLS4 proposal with same-recipient alternatives."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_local_ols_proposal as learned

g, run, np = learned.geometry, learned.run, learned.np
base = g.model.base
PUBLIC = learned.PUBLIC
PRIVATE = learned.PRIVATE/'capacity'
REG = PUBLIC/'capacity_registration.json'
FILES = (Path(__file__), PUBLIC/'capacity_protocol.md')


def register():
    for k in ('fit', 'score', 'replay'):
        run.checked(run.read(PUBLIC/f'{k}_execution.json')['artifact'])
    parents = (learned.REG, PUBLIC/'analysis.json', PUBLIC/'fit_execution.json',
               PUBLIC/'replay_execution.json', g.REG, g.PUBLIC/'development.json',
               Path(g.model.__file__), Path(base.__file__))
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in (*FILES, *parents)],
        candidate='donor_selected_self_only', choice_reason='earlier_matched_neighbor_comparison_negative',
        fit_or_reselection=False, original_risk_rule=True, external_access=False,
        primary='new_verified_positive_ols4_capacity_vs_previous_both_slate',
        second='direct_paired_full_path_self_vs_uniform', seconds_per_solve=4.)))


def audit(cv, ols, y, valid, offsets, slots, paths, proposal, recording, cut):
    matched = np.full_like(slots[:4], -1)
    new = np.zeros_like(paths[:4])
    used = slots[:4] >= 0
    new[used] = proposal[slots[:4][used]]
    changed = np.zeros_like(used)
    changed[used] = (new[used] != cv[slots[:4][used]]).any(axis=(-1, -2))
    retain = used & np.all(used == changed, axis=-1, keepdims=True)
    matched[retain] = slots[:4][retain]
    ix, pred = np.concatenate((slots, matched)), np.concatenate((paths, new))
    p = base.prepare(cv, ols, y, valid, offsets, ix, pred, cut['easy'])
    take, info = base.solve(p, 'ols4', tuple(range(16)), seconds=4.)
    info.update(feasible=False, positive=False, new_proposals_used=0)
    if take is not None:
        out = cv.copy()
        for i in take:
            a, q = p['arm'][i], p['query'][i]
            use = ix[a, q] >= 0
            out[ix[a, q, use]] = pred[a, q, use]
            info['new_proposals_used'] += int(a >= 12)
        risk = base.policy(out, cv, cv, y, valid, np.full(len(cv), recording),
                           easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][recording]
        mass = base.comparison_masses(ols, out, cv, y, valid, p['weights'])
        lower = mass['known_delta']-mass['unknown_radius']
        np.testing.assert_allclose(info['witness_gain_mass'], lower, rtol=1e-10, atol=1e-7)
        info.update(feasible=risk['supported'], positive=risk['supported'] and lower > 0.)
    return info


def readout(beat):
    _, cut, controller = g.prior.register()
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = learned.h.previous.prior.blocks(d, lab['frames'])
    frozen = run.read(learned.PRIVATE/'freeze.json')['cells']
    cells, contrasts, roster = [], [], {}
    for c in controller['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        ols, kin = (bank[:, run.NAMES.index(k)] for k in (run.parent.OLS, 'self'))
        prop = np.full_like(ols, np.nan)
        for f in frozen:
            m = f['identity']
            if m['seed'] == seed and m['arm'] == 'self':
                with np.load(run.checked(f['forecast']), allow_pickle=False) as z:
                    at = np.flatnonzero(d['sites'] == m['held'])
                    np.testing.assert_array_equal(z['index'], at)
                    prop[at] = ols[at]+f['amplitude']*z['delta']
        if not np.isfinite(prop).all():
            raise ValueError('Incomplete frozen learned paths')
        with np.load(run.checked(c['cache']), allow_pickle=False) as f:
            coeff = dict(zip(f['sources'], f['excluded_source_coefficients']))
        for r, (ix, groups) in records.items():
            mark = PRIVATE/f'{seed}_{r}.json'
            loc = ix[np.concatenate(groups)]
            site = str(d['sites'][loc[0]])
            roster[r] = site
            if not mark.exists():
                offsets = np.r_[0, np.cumsum(list(map(len, groups)))]
                slots = np.full((4, len(groups), 2), -1, dtype=int)
                paths = np.zeros((*slots.shape, 12, 2))
                uniform = ols[loc].copy()
                for q, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
                    z = loc[lo:hi]
                    choices = g.cal.model.decide(d['history'][z], bank[z], cv[z], ols[z], coeff[site])
                    for j, a in enumerate(choices[:4]):
                        i = np.flatnonzero(a >= 0)
                        slots[j, q, :len(i)] = lo+i
                        paths[j, q, :len(i)] = bank[z[i], a[i]]
                    uniform[lo:hi] += ((cv[z]-ols[z])+(kin[z]-ols[z])).mean(0)/3.
                slots, paths = g.model.augment(cv[loc], ols[loc], kin[loc], offsets, slots, paths)
                result = audit(cv[loc], ols[loc], y[loc], valid[loc], offsets, slots, paths, prop[loc], r, cut)
                old = run.read(g.PRIVATE/f'{seed}_{r}.json')
                if old['registration'] != run.sha(g.REG):
                    raise ValueError('Incompatible previous same-recipient score')
                ref = old['result']['outcomes']['both']
                result.update(new_feasible=ref['proved_infeasible'] and result['feasible'],
                    new_positive=result['positive'] and not ref['witness_positive'])
                weight = np.repeat(1./np.diff(offsets), np.diff(offsets))
                mass = base.comparison_masses(uniform, prop[loc], cv[loc], y[loc], valid[loc], weight)
                run.once(mark, run.stable(dict(registration=run.sha(REG), result=result, comparison=mass)))
            saved = run.read(mark)
            if saved['registration'] != run.sha(REG):
                raise ValueError('Incompatible follow-up checkpoint')
            cells.append(dict(saved['result'], source=site))
            contrasts.append(dict(recording=r, seed=seed, source=site, metrics={},
                                  comparisons={'self_vs_uniform': saved['comparison']}))
        beat(seed=seed, completed=len(cells))
    if len(cells) != 102:
        raise ValueError('Require every development view')
    counts = {k: sum(v[k] for v in cells) for k in ('feasible', 'positive', 'new_feasible', 'new_positive')}
    counts['new_positive_sources'] = len({v['source'] for v in cells if v['new_positive']})
    counts['views_using_proposal'] = sum(v['feasible'] and v['new_proposals_used'] > 0 for v in cells)
    counts['unresolved'] = sum(not (v['proved_infeasible'] or (v['feasible'] and v['optimum_proved']))
                               for v in cells)
    comparison = learned.grouped.grouped_readout(contrasts, roster, learned.SEEDS)
    return dict(capacity=counts, comparison=comparison['effects']['self_vs_uniform'],
                deployment=False, external_data_opened=False, label_aware_capacity=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the follow-up protocol before scoring')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        out = PUBLIC/'capacity_analysis.json'
        run.once(out, run.stable(dict(registration=run.sha(REG), **readout(beat))))
        receipt = PUBLIC/'capacity_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Follow-up replay differs')


if __name__ == '__main__':
    main()
