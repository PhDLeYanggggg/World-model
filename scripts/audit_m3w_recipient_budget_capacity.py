"""Separate candidate geometry from recipient retrieval at the original query caps."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_slate_geometry_capacity as geometry
from src.evaluation import m3w_recipient_budget_capacity as model

g, cal, run, np = geometry, geometry.cal, geometry.run, geometry.np
PUBLIC = g.PUBLIC.parent/'native_recipient_budget_capacity_v1'
PRIVATE = g.PRIVATE.parent/'native_recipient_budget_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_recipient_budget_capacity.py')
SLATES = ('same_pairs', 'same_recipients', 'all_recipients')


def register():
    cut, controller = g.register()
    parents = (g.REG, g.PUBLIC/'development.json', g.PUBLIC/'development_execution.json',
               Path(g.model.__file__), Path(g.model.base.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut,
        design=SLATES, source_scope='exposed_moving_A_only', seeds=g.prior.SEEDS,
        cap='actual_fixed_matched_arm_intervention_count_per_query', seconds_per_solve=4.,
        primary='gain_over_ols4_at_original_recording_risk', calibration_or_confirmation_access=False)))
    return cut, controller


def query_design(d, loc, offsets, cv, bank, coeff):
    oi, ki = (run.NAMES.index(k) for k in (cal.OLS, 'self'))
    ols, kin = bank[:, oi], bank[:, ki]
    sizes = np.diff(offsets)
    center = np.add.reduceat((cv-ols)+(kin-ols), offsets[:-1], axis=0)
    shifted = ols + np.repeat(center/(3.*sizes[:, None, None]), sizes, axis=0)
    cap = np.zeros(len(offsets)-1, int)
    legal_pairs = np.zeros((len(ols), bank.shape[1]+1), bool)
    for j, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        q = slice(lo, hi)
        choices = cal.model.decide(d['history'][loc[q]], bank[q], cv[q], ols[q], coeff)[:4]
        cap[j] = (choices[0] >= 0).sum()
        for a in choices:
            row = np.flatnonzero(a >= 0)
            i = row+lo
            if not len(i):
                continue
            legal_pairs[i, a[row]] = True
            for k, path in ((oi, ols), (-1, shifted)):
                if (path[i] != cv[i]).any(axis=(1, 2)).all():
                    legal_pairs[i, k] = True
    paths = np.concatenate((bank, shifted[:, None]), axis=1)
    return paths, cap, legal_pairs


def run_all(cut, controller, beat):
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    by_record = cal.prior.h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    rows = []
    for cell in controller['cells']:
        seed = cell['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
            coeff = dict(zip(f['sources'], f['excluded_source_coefficients']))
        for r, (ix, groups) in by_record.items():
            out = PRIVATE/f'{seed}_{r}.json'
            if not out.exists():
                loc = ix[np.concatenate(groups)]
                site = str(d['sites'][loc[0]])
                offs = np.r_[0, np.cumsum(list(map(len, groups)))]
                paths, cap, same = query_design(d, loc, offs, cv[loc], bank[loc], coeff[site])
                masks = (same, np.broadcast_to(same.any(1)[:, None], same.shape), np.ones_like(same))
                ols = paths[:, run.NAMES.index(cal.OLS)]
                outcomes = {k: model.solve(cv[loc], ols, paths, y[loc], valid[loc], offs, cap, a, r,
                                           cut['easy'], cut['hard'])
                            for k, a in zip(SLATES, masks)}
                previous = run.read(g.PRIVATE/f'{seed}_{r}.json')
                if previous['registration'] != run.sha(g.REG):
                    raise ValueError('Incompatible whole-query prior')
                old = previous['result']['outcomes']['both']
                prior = dict(feasible=old['verified_witness'], positive=old['witness_positive'],
                    proved_infeasible=old['proved_infeasible'], optimum_proved=old['optimum_proved'],
                    upper_mass=old.get('upper_gain_mass'), witness_mass=old.get('witness_gain_mass'))
                for restricted, superset in zip((prior, *outcomes.values()), outcomes.values()):
                    if (restricted['feasible'] and superset['proved_infeasible']) or (
                            restricted['feasible'] and superset['upper_mass'] is not None
                            and restricted['witness_mass'] > superset['upper_mass']+1e-5):
                        raise ValueError('Relaxed capacity must contain the previous feasible set')
                run.once(out, run.stable(dict(registration=run.sha(REG), source=site, outcomes=outcomes,
                    whole=prior, rows=len(loc), previous_eligible_rows=int(same.any(1).sum()),
                    query_count=len(cap), nonzero_cap_queries=int((cap > 0).sum()))))
            saved = run.read(out)
            if saved['registration'] != run.sha(REG):
                raise ValueError('Stale recipient-budget result')
            rows.append(saved)
        beat(seed=seed, completed=len(rows))
    if len(rows) != 102:
        raise ValueError('Incomplete exposed development views')
    return rows


def summarize(rows):
    previous = [r['whole'] for r in rows]
    comparison = {}
    for k in SLATES:
        cur = [r['outcomes'][k] for r in rows]
        inc = [i for i, (p, c) in enumerate(zip(previous, cur)) if c['positive'] and (
               p['proved_infeasible'] or (p['optimum_proved'] and not p['positive']))]
        fit = [100*(c['witness_mass']-p['upper_mass'])/c['reference_mass'] for p, c in zip(previous, cur)
               if p['feasible'] and p['optimum_proved'] and c['feasible'] and c['reference_mass'] > 0.]
        comparison[k] = dict(feasible=sum(c['feasible'] for c in cur), positive=sum(c['positive'] for c in cur),
            proved_infeasible=sum(c['proved_infeasible'] for c in cur),
            infeasible_no_easy_option=sum(c['status'] == 'no_selected_easy_candidate' for c in cur),
            nonpositive_upper_among_feasible=sum(c['feasible'] and c['upper_mass'] is not None
                                                and c['upper_mass'] <= 0. for c in cur),
            unresolved=sum(not (c['proved_infeasible'] or c['optimum_proved']) for c in cur),
            new_feasible_vs_previous=sum(c['feasible'] and p['proved_infeasible'] for p, c in zip(previous, cur)),
            new_positive_vs_previous=len(inc), new_positive_sources=len({rows[i]['source'] for i in inc}),
            comparable_feasible_views=len(fit), median_lower_gain_delta_pp=float(np.median(fit)) if fit else None)
        previous = cur
    return dict(views=len(rows), sources=len({r['source'] for r in rows}),
        queries_across_seeds=sum(r['query_count'] for r in rows),
        positive_cap_queries_across_seeds=sum(r['nonzero_cap_queries'] for r in rows),
        rows_across_seeds=sum(r['rows'] for r in rows),
        whole_slate_eligible_rows_across_seeds=sum(r['previous_eligible_rows'] for r in rows),
        whole_slate=dict(feasible=sum(r['whole']['feasible'] for r in rows),
                         positive=sum(r['whole']['positive'] for r in rows)), slates=comparison)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'score'))
    phase = p.parse_args().phase
    cut, ctrl = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the matched-budget design before scoring')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        out = PUBLIC/'analysis.json'
        run.once(out, run.stable(dict(registration=run.sha(REG), summary=summarize(run_all(cut, ctrl, beat)),
            result_source='fresh_label_aware_fixed_budget_diagnostic', external_data_opened=False,
            independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)))
        receipt = PUBLIC/'execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start, pid=os.getpid()))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Replayed result differs')


if __name__ == '__main__':
    main()
