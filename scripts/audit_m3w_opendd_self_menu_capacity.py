"""Measure extra feasible capacity from the self-history menu, not its realized policy score."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_exact_slot_headroom as exact
from scripts import audit_m3w_opendd_self_menu as menu

base, study, np, torch = menu.base, menu.study, menu.np, menu.torch
OUT = menu.OUT.parent/'opendd_self_menu_capacity_v1'
LOCAL = menu.LOCAL.parent/OUT.name


def register():
    key, cohorts, _, _, _, weights, producer = menu.register()
    old_key, old_cohorts, _, _ = exact.register()
    new_reg, new_freeze, new_result = (menu.OUT/f'{k}.json' for k in ('registration', 'freeze', 'readout'))
    old_result = exact.OUT/'readout.json'
    f, r, before = map(study.read, (new_freeze, new_result, old_result))
    if (f['registration'] != key or f['outcomes_accessed'] or r['freeze'] != study.artifact(new_freeze)
            or before['registration'] != old_key or not before['hindsight_only'] or old_cohorts != cohorts
            or r['reserved_outcomes_opened'] or before['reserved_outcomes_opened']):
        raise ValueError('Expanded and base comparison evidence does not align')
    registration = OUT/'registration.json'
    files = (Path(__file__), Path(menu.__file__), Path(exact.__file__), Path(exact.headroom.__file__))
    base.immutable(registration, dict(bindings=[study.artifact(p) for p in (new_reg, new_freeze, new_result, old_result)],
        implementation=[study.artifact(p) for p in files], fixed_singletons=True, original_2pct=True,
        comparator='original base-menu additive selection in both oracle arms',
        varied='only add the causal, source-held self-history path at each already fixed slot',
        positive_margin_pp=exact.headroom.POSITIVE_PP, sites=menu.SITES, seeds=menu.SEEDS,
        hindsight_only=True, development_only=True, no_new_training=True))
    return base.sha(registration), cohorts, weights, producer, f, before


def positive(row):
    return (row.get('verified_direct_pp') or 0.) > exact.headroom.POSITIVE_PP


def execute(key, cohorts, weights, producer, f, before):
    d = menu.exp.loader(cohorts)
    sites, receipts, overall, winners = {}, {}, Counter(), set()
    for site in menu.SITES:
        src = d[site]
        cut = np.quantile(np.concatenate([v['e'] for s, v in d.items() if s != site]), [.25, .75])
        tally, retained = Counter(), set()
        for seed in menu.SEEDS:
            name = f'{site}_{seed}'
            old_path = study.checked(before['checkpoints'][name])
            prev = study.read(old_path)
            if prev['binding']['registration'] != before['registration']:
                raise ValueError('Base solver checkpoint binding changed')
            old_rows = {r['record']: r['arms']['two_edits_only'] for r in prev['rows']}
            path = LOCAL/f'{name}.json'
            bind = dict(registration=key, fold=name, old=study.artifact(old_path), expanded=f['banks'][name])
            if not path.exists():
                with np.load(study.checked(bind['expanded']), allow_pickle=False) as z:
                    for k in ('q', 'cv'):
                        np.testing.assert_array_equal(src[k], z[k])
                    a, j = z['base_additive'], z['base_joint']
                new = menu.self_forecast(src['h'], site, seed, weights[name], producer)
                bank = np.concatenate((src['bank'], new[:, None]), axis=1)
                rows = {}
                for r in np.unique(src['rec']):
                    ix = src['rec'] == r
                    q = np.unique(src['q'][ix], return_inverse=True)[1]
                    row = exact.assess(src['cv'][ix], src['ols'][ix], bank[ix], a[ix], j[ix],
                                       src['y'][ix], src['v'][ix], q, cut, double_only=True)
                    old = old_rows[str(r)]
                    if row['slots'] != old['slots'] or row['joint_lower_pp'] != old['joint_lower_pp']:
                        raise ValueError('Expanded capacity altered the fixed reference experiment')
                    if old['slots'] and np.any(np.asarray(row['relaxed_upper_pp']) + 1e-7 < old['relaxed_upper_pp']):
                        raise ValueError('Adding an option lowered the relaxed bound')
                    if old.get('verified_dual') and row['state'] in ('no_easy_support', 'infeasible', 'nonpositive_reference_bound'):
                        raise ValueError('Expanded menu infeasible despite a verified old witness')
                    if old.get('verified_dual') and row.get('solver', {}).get('upper_pp') is not None:
                        if row['solver']['upper_pp'] + 1e-7 < old['verified_direct_pp']:
                            raise ValueError('Expanded upper bound excludes a verified old witness')
                    rows[str(r)] = row
                base.immutable(path, dict(binding=bind, rows=rows))
            saved = study.read(path)
            if saved['binding'] != bind or set(saved['rows']) != set(old_rows):
                raise ValueError('Solver cache does not cover the bound recording roster')
            for ident, row in saved['rows'].items():
                old = old_rows[ident]
                tally.update(dict(views=1, estimable=int(row['joint_lower_pp'] is not None),
                    base_dual=int(old.get('verified_dual', False)), expanded_dual=int(row.get('verified_dual', False)),
                    base_positive=int(positive(old)), expanded_positive=int(positive(row)),
                    new_positive=int(positive(row) and not positive(old)),
                    unresolved=int(row['state'] in ('limited', 'unresolved', 'numerical_boundary'))))
                if positive(row):
                    retained.add(ident)
            receipts[name] = study.artifact(path)
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(dict(pid=os.getpid(), completed=len(receipts), total=21, last=name))+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(name, dict(completed=len(receipts)), flush=True)
        sites[site] = dict(tally, expanded_positive_recordings=len(retained))
        winners.update((site, ident) for ident in retained)
        overall.update(tally)
    parent = before['totals']['two_edits_only']
    if overall['base_positive'] != parent['verified_direct_positive'] or overall['base_dual'] != parent['verified_dual']:
        raise ValueError('Base capacity did not replay')
    base.immutable(OUT/'readout.json', dict(result_source='fresh_run', inputs='cached_verified', registration=key,
        totals=dict(overall, expanded_positive_recordings=len(winners)), sites=sites, checkpoints=receipts,
        scientific_role='exposed_development_hindsight_only', reserved_outcomes_opened=False, deployment_changed=False))
    print(dict(overall, expanded_positive_recordings=len(winners)), flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(*register())


if __name__ == '__main__':
    main()
