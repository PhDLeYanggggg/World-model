"""Diagnose headroom under the same fixed OpenDD action menu and risk rule."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_slate_shared_menu as prior
from src.evaluation import m3w_exact_slot_headroom as headroom

base, study, exp, np = prior.base, prior.study, prior.exp, prior.np
SITES, SEEDS = prior.SITES, prior.SEEDS
OUT = prior.OUT.parent/'opendd_exact_slot_headroom_v2'
LOCAL = prior.LOCAL.parent/OUT.name
SECONDS = 2.
ARMS = ('all_edits', 'two_edits_only')


def register():
    parent_reg, freeze, readout = (prior.OUT/name for name in ('registration.json', 'freeze.json', 'readout.json'))
    reg, f, r = map(study.read, (parent_reg, freeze, readout))
    if (f['registration'] != base.sha(parent_reg) or r['freeze'] != study.artifact(freeze)
            or not reg['development_only'] or not reg['original_2pct'] or not reg['no_threshold_tuning']):
        raise ValueError('Require frozen shared-menu development evidence')
    cohorts = study.read(study.checked(reg['parent']))['cohorts']
    if set(cohorts) != set(SITES) or set(f['forecasts']) != {f'{s}_{k}' for s in SITES for k in SEEDS}:
        raise ValueError('Unregistered sites or seeds')
    files = [Path(__file__), Path(headroom.__file__), Path(prior.shared.__file__),
             ROOT/'src/evaluation/m3w_cv_anchor_budget.py', ROOT/'src/evaluation/m3w_selection_forecast_readout.py']
    target = OUT/'registration.json'
    base.immutable(target, dict(parent=[study.artifact(p) for p in (parent_reg, freeze, readout)],
        code=[study.artifact(p) for p in files], sites=SITES, seeds=SEEDS, arms=ARMS, original_2pct=True,
        objective='paired missing-step lower utility versus exact-budget matched additive',
        feasibility='one edit per slot; distinct agents; dual CV and OLS4 lower gain; original three CV risks',
        two_edits_only='singleton queries exactly match additive; only two-edit assignments may change',
        per_reference_tolerance_pp=headroom.POSITIVE_PP, solver_seconds_per_view=SECONDS,
        use='hindsight development diagnostic, never policy training or independent evidence'))
    return base.sha(target), cohorts, f, r


def assess(cv, ols, bank, additive, joint, y, v, q, cut, *, double_only):
    w = 1./np.bincount(q)[q]
    d, reason = headroom.make_design(cv, ols, bank, additive, joint, y, v, q, cut[0], double_only=double_only)
    if d is None:
        return dict(state=reason, slots=0, joint_lower_pp=None)
    ref = d['reference']
    def pair(a, b):
        return float(100.*(w @ headroom.lower(a, b, y, v))/ref)
    def risk(path):
        return study.policy(path, cv, cv, y, v, np.repeat('record', len(cv)),
                            easy_cut=cut[0], hard_cut=cut[1])['recordings']['record']
    joint_delta = pair(additive, joint)
    if joint_delta > d['upper'][2]+1e-7:
        raise ValueError('Frozen joint choice exceeds its relaxed upper bound')
    joint_risk = risk(joint)
    joint_dual = joint_risk['supported'] and min(pair(cv, joint), pair(ols, joint)) > 0.
    use, log = headroom.solve(d, SECONDS)
    row = dict(state=log['status'], slots=d['slots'], joint_lower_pp=joint_delta,
               joint_risk=joint_risk['supported'], joint_dual=joint_dual, relaxed_upper_pp=d['upper'].tolist(),
               solver=log, verified_direct_pp=None, verified_dual=False)
    if use is not None:
        path = cv.copy()
        path[d['row'][use]] = d['path'][use]
        peak = np.linalg.norm(path-cv, axis=-1).max(-1)
        target_peak = np.linalg.norm(joint-cv, axis=-1).max(-1)
        for i in np.unique(q):
            np.testing.assert_allclose(np.sort(peak[q == i]), np.sort(target_peak[q == i]), atol=1e-9, rtol=1e-9)
            if double_only and np.count_nonzero(target_peak[q == i]) == 1:
                np.testing.assert_allclose(path[q == i], additive[q == i], atol=1e-9, rtol=1e-9)
        delta = np.array([pair(k, path) for k in (cv, ols, additive)])
        np.testing.assert_allclose(delta, d['offset']+d['gain'][use].sum(0), atol=1e-7, rtol=1e-7)
        original = risk(path)
        if original['selected'] != d['slots']:
            raise ValueError('Original risk scorer did not see the exact frozen budget')
        row.update(witness_pp=delta.tolist(), witness_risk=original)
        if original['supported'] and min(delta[:2]) > 0.:
            row.update(verified_direct_pp=float(delta[2]), verified_dual=True)
        else:
            row['state'] = 'numerical_boundary'
    if joint_dual and row['state'] in ('no_easy_support', 'nonpositive_reference_bound', 'infeasible'):
        if min(pair(cv, joint), pair(ols, joint)) > headroom.POSITIVE_PP:
            raise ValueError('Certified negative contradicts the original passing choice')
    return row


def summary(rows):
    comparable = [r for r in rows if r['joint_lower_pp'] is not None]
    return dict(views=len(rows), recordings=len({r['record'] for r in rows}), estimable=len(comparable),
        states=dict(Counter(r['state'] for r in rows)), fixed_edits=sum(r['slots'] for r in rows),
        joint_dual=sum(r.get('joint_dual', False) for r in rows),
        verified_dual=sum(r.get('verified_dual', False) for r in rows),
        verified_direct_positive=sum((r.get('verified_direct_pp') or 0.) > headroom.POSITIVE_PP for r in rows),
        verified_direct_positive_recordings=len({r['record'] for r in rows
                                                 if (r.get('verified_direct_pp') or 0.) > headroom.POSITIVE_PP}),
        relaxed_positive=sum(r['relaxed_upper_pp'][2] > headroom.POSITIVE_PP for r in comparable),
        observed_joint_mean_pp=float(np.mean([r['joint_lower_pp'] for r in comparable])) if comparable else None,
        relaxed_mean_pp=float(np.mean([r['relaxed_upper_pp'][2] for r in comparable])) if comparable else None)


def run(key, cohorts, freeze, old):
    source = exp.loader(cohorts)
    groups, merged, receipts = {}, {k: [] for k in ARMS}, {}
    for site in SITES:
        d = source[site]
        cut = np.quantile(np.concatenate([source[s]['e'] for s in SITES if s != site]), [.25, .75])
        rows = {k: [] for k in ARMS}
        for seed in SEEDS:
            name = f'{site}_{seed}'
            cache = LOCAL/f'{name}.json'
            binding = dict(registration=key, seed=seed, site=site, bank=freeze['forecasts'][name])
            if not cache.exists():
                with np.load(study.checked(binding['bank']), allow_pickle=False) as bank:
                    for k in ('q', 'cv'):
                        np.testing.assert_array_equal(d[k], bank[k])
                    np.testing.assert_array_equal(d['ols'], bank['ols4'])
                    control, joint = bank['matched_independent'], bank['joint']
                records = []
                for r in np.unique(d['rec']):
                    at = d['rec'] == r
                    q = np.unique(d['q'][at], return_inverse=True)[1]
                    record = {k: assess(d['cv'][at], d['ols'][at], d['bank'][at], control[at], joint[at],
                                        d['y'][at], d['v'][at], q, cut, double_only=k == ARMS[1]) for k in ARMS}
                    a, b = (record[k] for k in ARMS)
                    if a['joint_lower_pp'] is not None and np.any(np.asarray(b['relaxed_upper_pp']) > a['relaxed_upper_pp'] + np.ones(3)*1e-7):
                        raise ValueError('Restricted menu exceeds the broader upper bound')
                    if b.get('verified_dual') and a['state'] in ('no_easy_support', 'nonpositive_reference_bound', 'infeasible'):
                        raise ValueError('Restricted witness contradicts broader infeasibility')
                    if b.get('verified_dual') and a.get('solver', {}).get('upper_pp') is not None:
                        if b['verified_direct_pp'] > a['solver']['upper_pp']+1e-7:
                            raise ValueError('Restricted witness exceeds broader solver bound')
                    records.append(dict(record=str(r), arms=record))
                base.immutable(cache, dict(binding=binding, rows=records))
            saved = study.read(cache)
            if saved['binding'] != binding or [r['record'] for r in saved['rows']] != list(map(str, np.unique(d['rec']))):
                raise ValueError('Checkpoint belongs to a different source or slot menu')
            for k in ARMS:
                rows[k].extend(dict(record=r['record'], **r['arms'][k]) for r in saved['rows'])
            receipts[name] = study.artifact(cache)
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(dict(pid=os.getpid(), phase='hindsight', completed=len(receipts),
                                               total=len(SITES)*len(SEEDS), last=name))+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(name, dict(views=len(saved['rows']), completed=len(receipts)), flush=True)
        groups[site] = {}
        for k in ARMS:
            merged[k].extend(dict(r, record=f'{site}:{r["record"]}') for r in rows[k])
            a, b = summary(rows[k]), old['groups'][site]
            groups[site][k] = a
            if (a['views'], a['recordings'], a['estimable'], a['fixed_edits'], a['joint_dual']) != (
                    b['views'], b['recordings'], b['estimable'], b['arms']['joint']['edits'], b['arms']['joint']['dual_pass']):
                raise ValueError('Frozen comparison counts do not replay')
            if b['lower_pp'] is not None:
                np.testing.assert_allclose(a['observed_joint_mean_pp'], b['lower_pp'], atol=1e-9, rtol=1e-9)
    totals = {k: summary(merged[k]) for k in ARMS}
    base.immutable(OUT/'readout.json', dict(result_source='fresh_run', frozen_inputs='cached_verified',
        registration=key, groups=groups, totals=totals, checkpoints=receipts,
        hindsight_only=True, no_new_training=True, original_2pct=True, reserved_outcomes_opened=False,
        independent_confirmation=False, deployment_changed=False))
    print(totals, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prior.torch.set_num_threads(4)
    prior.torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(*register())


if __name__ == '__main__':
    main()
