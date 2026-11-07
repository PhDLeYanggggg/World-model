"""Test geometry changes on the same exposed-development recipients."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_joint_slate_capacity as prior
from src.evaluation import m3w_slate_geometry_capacity as model

cal, run, np = prior.cal, prior.run, prior.np
PUBLIC = prior.PUBLIC.parent/'native_slate_geometry_capacity_v1'
PRIVATE = prior.PRIVATE.parent/'native_slate_geometry_capacity_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_slate_geometry_capacity.py')


def register():
    _, cut, controller = prior.register()
    parents = (prior.REG, prior.PUBLIC/'development.json', prior.PUBLIC/'development_execution.json',
               Path(prior.audit.__file__), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents], cut=cut, slates=model.SLATES,
        scope='exposed_development_only', seeds=prior.SEEDS, solve_time_seconds=4.,
        original_risk_unchanged=True, primary='positive_conservative_OLS4_capacity',
        forecasts='same_recipients_exact_count_CV_fallback', external_access=False)))
    return cut, controller


def packets(cut, controller):
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    by_record = cal.prior.h.previous.prior.blocks(d, labels['frames'])
    for cell in controller['cells']:
        seed = cell['seed']
        missing = [r for r in by_record if not (PRIVATE/f'{seed}_{r}.json').exists()]
        if missing:
            floor, bank = run.candidates(d, refs, seed)
            with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
                coef = dict(zip(f['sources'], f['excluded_source_coefficients']))
        for r, (ix, groups) in by_record.items():
            path = PRIVATE/f'{seed}_{r}.json'
            if not path.exists():
                loc = ix[np.concatenate(groups)]
                site = str(d['sites'][loc[0]])
                cv, paths = floor[loc], bank[loc]
                ols, kin = (paths[:, run.NAMES.index(a)] for a in (cal.OLS, 'self'))
                offsets = np.r_[0, np.cumsum(list(map(len, groups)))]
                slots = np.full((4, len(groups), 2), -1, dtype=int)
                replacements = np.zeros((*slots.shape, 12, 2))
                for q, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
                    z = slice(lo, hi)
                    for j, a in enumerate(cal.model.decide(d['history'][loc[z]], paths[z], cv[z],
                                                            ols[z], coef[site])[:4]):
                        i = np.flatnonzero(a >= 0)
                        slots[j, q, :len(i)] = lo+i
                        replacements[j, q, :len(i)] = paths[lo+i, a[i]]
                y = labels['target_eval'][loc]-d['origin'][loc, None]
                result = model.audit(cv, ols, kin, y, labels['valid'][loc], offsets, slots, replacements,
                                     r, cut['easy'], cut['hard'])
                run.once(path, run.stable(dict(registration=run.sha(REG), result=result, site=site)))
            saved = run.read(path)
            if saved['registration'] != run.sha(REG):
                raise ValueError('Stale geometry result')
            yield saved
        event = dict(pid=os.getpid(), seed=seed, state='completed_development_seed', unix=time.time())
        run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
        print(event, flush=True)


def summarize(rows):
    names = list(model.SLATES)
    res = [r['result'] for r in rows]
    data = {k: [r['outcomes'][k] for r in res] for k in names}
    reference = run.read(prior.PUBLIC/'development.json')['summary']['options']['four_option_vs_ols4']
    out = {}
    for k, cells in data.items():
        b = data['frozen']
        new = [i for i, (a, v) in enumerate(zip(b, cells)) if a['proved_infeasible'] and v['verified_witness']]
        pos = [i for i, (a, v) in enumerate(zip(b, cells)) if
               (a['proved_infeasible'] or (a.get('upper_gain_mass') is not None and a['upper_gain_mass'] <= 0.))
               and v['witness_positive']]
        lower = [100*(v['witness_gain_mass']-a['upper_gain_mass'])/r['reference_mass']
                 for a, v, r in zip(b, cells, res) if a['verified_witness'] and a['optimum_proved']
                 and v['verified_witness'] and r['reference_mass'] > 0.]
        out[k] = dict(verified_risk_feasible=sum(v['verified_witness'] for v in cells),
            proved_infeasible=sum(v['proved_infeasible'] for v in cells),
            positive_witnesses=sum(v['witness_positive'] for v in cells),
            unresolved=sum(not (v['proved_infeasible'] or (v['verified_witness'] and v['optimum_proved']))
                           for v in cells),
            new_feasible_vs_frozen=len(new), new_positive_vs_proved_frozen_no_gain=len(pos),
            sources_with_new_positive=len({rows[i]['site'] for i in pos}),
            previously_feasible_comparable_views=len(lower),
            median_gain_over_frozen_optimum_pp=float(np.median(lower)) if lower else None,
            views_with_proved_strict_improvement=sum(v > 1e-7 for v in lower))
    for key in ('verified_risk_feasible', 'proved_infeasible', 'positive_witnesses'):
        if out['frozen'][key] != reference[key]:
            raise ValueError('Frozen slate did not match previous study')
    return dict(views=len(rows), sources=len({r['site'] for r in rows}), slates=out,
                extra_options=np.sum([r['extra_options'] for r in res], axis=0).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'development'))
    phase = p.parse_args().phase
    cut, controller = register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit geometry study design before readout')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = list(packets(cut, controller))
        if len(rows) != 102:
            raise ValueError('Incomplete development roster')
        out = PUBLIC/'development.json'
        run.once(out, run.stable(dict(registration=run.sha(REG), summary=summarize(rows),
            result_source='fresh_label_aware_same_recipient_geometry_diagnostic',
            deployment=False, independent_confirmation=False, original_rules_unchanged=True,
            external_inputs_opened=False, external_labels_opened=False, stage5c_executed=False, smc_enabled=False)))
        receipt = PUBLIC/'development_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), pid=os.getpid(), seconds=time.monotonic()-start))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Previous receipt binds another result')


if __name__ == '__main__':
    main()
