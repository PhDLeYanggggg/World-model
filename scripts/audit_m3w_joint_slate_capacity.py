"""Measure label-aware capacity without selecting an external operating rule."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal
from src.evaluation import m3w_joint_slate_capacity as audit

run, np = cal.run, cal.np
PUBLIC = cal.roles.HOME/'native_joint_slate_capacity_v1'
PRIVATE = cal.DATA/'native_joint_slate_capacity_v1'
REG = PUBLIC/'registration.json'
SEEDS = cal.SEEDS
FILES = (Path(audit.__file__), Path(__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_joint_slate_capacity.py')


def register():
    role, cut, controller = cal.register()
    parents = (cal.REG, cal.prior.REG, cal.prior.PUBLIC/'analysis.json', cal.prior.PUBLIC/'replay_execution.json',
               Path(audit.moments.__code__.co_filename), Path(audit.policy.__code__.co_filename), *FILES)
    run.once(REG, run.stable(dict(bindings=[run.artifact(p) for p in parents],
        cut=cut, option_families=cal.prior.model.MATCHED, solve_time_seconds=1.,
        role_scope=['exposed_development', 'replayed_calibration'], confirmation_access=False,
        risk_rule_unchanged=True, label_aware_not_policy=True, choices_exported=False)))
    return role, cut, controller


def dev_packets(cut, controller, beat):
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    by_record = cal.prior.h.previous.prior.blocks(d, labels['frames'])
    for cell in controller['cells']:
        seed = cell['seed']
        floor, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
            coef = dict(zip(f['sources'], f['excluded_source_coefficients']))
        for r, (ix, groups) in by_record.items():
            out = PRIVATE/f'dev_{seed}_{r}.json'
            if not out.exists():
                loc = ix[np.concatenate(groups)]
                s = str(d['sites'][loc[0]])
                cv, paths = floor[loc], bank[loc]
                ols = paths[:, run.NAMES.index(cal.OLS)]
                offsets = np.r_[0, np.cumsum(list(map(len, groups)))]
                slots = np.full((4, len(groups), 2), -1, dtype=int)
                replacements = np.zeros((*slots.shape, 12, 2))
                for q, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
                    z = slice(lo, hi)
                    choices = cal.model.decide(d['history'][loc[z]], paths[z], cv[z], ols[z], coef[s])
                    for j, a in enumerate(choices[:4]):
                        i = np.flatnonzero(a >= 0)
                        slots[j, q, :len(i)] = lo+i
                        replacements[j, q, :len(i)] = paths[lo+i, a[i]]
                y = labels['target_eval'][loc]-d['origin'][loc, None]
                result = audit.audit(cv, ols, y, labels['valid'][loc], offsets, slots, replacements,
                                     r, cut['easy'], cut['hard'])
                run.once(out, run.stable(dict(registration=run.sha(REG), result=result)))
            saved = run.read(out)
            if saved['registration'] != run.sha(REG):
                raise ValueError('Stale development diagnostic')
            yield saved['result']
        beat(state='development_seed_done', seed=seed)


def calibration_packets(manifest, cut, beat):
    result, freeze = (cal.PUBLIC/k for k in ('analysis.json', 'forecast_freeze.json'))
    for phase, path in (('score', result), ('replay', result), ('infer', freeze)):
        if run.read(cal.PUBLIC/f'{phase}_execution.json')['artifact'] != run.artifact(path):
            raise ValueError('Require fully replayed calibration before any label access')
    doc = run.read(freeze)
    if [c['sequence'] for c in doc['clips']] != manifest['assignments']['calibration']['recordings']:
        raise ValueError('Unapproved calibration roster')
    inv = run.read(cal.CACHE)
    info = {r['sequence_id']: r for r in inv['recordings']
            if r['sequence_id'] in manifest['assignments']['calibration']['recordings']}
    for cell in doc['clips']:
        r = cell['sequence']
        cal.model.require_calibration(manifest, r)
        paths = [PRIVATE/f'cal_{seed}_{r}.json' for seed in SEEDS]
        if not all(p.exists() for p in paths):
            source = cal.reader(manifest, r, info, inv['provenance'])
            with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
                z = {k: f[k] for k in f.files}
            y, valid = cal.label_arrays(source, z)
            for p, seed in zip(paths, SEEDS):
                if not p.exists():
                    res = audit.audit(z['cv'], z['ols'], y, valid, z['offsets'], z[f'index_{seed}'][:4],
                        z[f'paths_{seed}'][:4], r, cut['easy'], cut['hard'])
                    run.once(p, run.stable(dict(registration=run.sha(REG), calibration=run.sha(result), result=res)))
        for p in paths:
            data = run.read(p)
            if data['registration'] != run.sha(REG) or data['calibration'] != run.sha(result):
                raise ValueError('Stale calibration diagnostic')
            yield data['result']
        beat(state='calibration_recording_done', recording=r)


def summarize(rows):
    out = {}
    for k in rows[0]['outcomes']:
        cases = [r['outcomes'][k] for r in rows]
        positive = [v for v, r in zip(cases, rows) if v.get('witness_positive') and r['reference_mass'] > 0.]
        gain = [100*v['witness_gain_mass']/r['reference_mass'] for v, r in zip(cases, rows)
                if v.get('witness_positive') and r['reference_mass'] > 0.]
        out[k] = dict(verified_risk_feasible=sum(v['verified_witness'] for v in cases),
            positive_witnesses=sum(v.get('witness_positive', False) for v in cases),
            no_easy_option=sum(v['status'] == 'no_easy_option_support' for v in cases),
            proved_infeasible=sum(v['proved_infeasible'] for v in cases),
            proved_optimum_with_witness=sum(v['optimum_proved'] and v['verified_witness'] for v in cases),
            unresolved_no_witness=sum(not v['proved_infeasible'] and not v['verified_witness'] for v in cases),
            nonpositive_upper_bound=sum(v.get('upper_gain_mass') is not None and v['upper_gain_mass'] <= 0.
                                        for v in cases),
            positive_witness_median_gain_pp=float(np.median(gain)) if positive else None)
    return dict(views=len(rows), views_without_full_easy_reference=sum(r['easy_rows'] == 0 for r in rows), options=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'development', 'calibration'))
    phase = p.parse_args().phase
    manifest, cut, controller = register()
    if phase == 'register':
        return
    for p in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT) != p.read_bytes():
            raise ValueError('Commit diagnostic design before readout')
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            print(msg, flush=True)

        rows = list(dev_packets(cut, controller, beat) if phase == 'development'
                    else calibration_packets(manifest, cut, beat))
        if len(rows) != (102 if phase == 'development' else 120):
            raise ValueError('Incomplete role-wide capacity diagnostic')
        out = PUBLIC/f'{phase}.json'
        run.once(out, run.stable(dict(registration=run.sha(REG), role=phase, summary=summarize(rows),
            result_source='fresh_label_aware_solver_diagnostic', deployable_policy=False,
            confirmation_inputs_opened=False, confirmation_labels_opened=False,
            stage5c_executed=False, smc_enabled=False, original_rules_unchanged=True)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), pid=os.getpid(), seconds=time.monotonic()-start))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Previous diagnostic receipt binds a different artifact')


if __name__ == '__main__':
    main()
