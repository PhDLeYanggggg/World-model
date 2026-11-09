"""Freeze and evaluate a peer transient against a matched self-history control."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_expanded_interaction_frontier as parent
from src.world_model import m3w_peer_transient as model

base, study, np = parent.base, parent.study, parent.np
original, menu = parent.original, parent.menu
OUT = parent.OUT.parent / 'opendd_peer_transient_v1'
LOCAL = parent.LOCAL.parent / OUT.name
ARMS = ('self_transient', 'peer_transient')


def register():
    _, cohorts, weights, producer, frozen, _, _ = parent.register()
    prior = parent.OUT / 'readout.json'
    before = study.read(prior)
    if (before['registration'] != base.sha(parent.OUT / 'registration.json')
            or not before['original_2pct'] or before['reserved_outcomes_opened']):
        raise ValueError('Require the previously verified expanded-menu experiment')
    reg = OUT / 'registration.json'
    base.immutable(reg, dict(cohorts=cohorts, reference=study.artifact(prior),
        source_frozen_paths=study.artifact(menu.OUT / 'freeze.json'), parent_weights=weights,
        files=[study.artifact(Path(x)) for x in (__file__, model.__file__, original.__file__,
                                                 parent.frontier.__file__, OUT / 'protocol.md')],
        arms=ARMS, sites=original.SITES, seeds=original.SEEDS, history_steps=8, forecast_steps=12,
        min_cos=model.MIN_COS, half_life=model.HALF_LIFE, original_2pct=True,
        no_training=True, development_only=True, reserved_outcomes_opened=False))
    return base.sha(reg), cohorts, weights, producer, frozen, before


def heartbeat(phase, fold, count):
    tmp = LOCAL / 'heartbeat.pending'
    tmp.write_text(study.json.dumps(dict(pid=os.getpid(), phase=phase, fold=fold, completed=count,
                                         total=21)) + '\n')
    tmp.replace(LOCAL / 'heartbeat.json')
    print(phase, fold, count, flush=True)


def freeze(key, cohorts, weights, producer, frozen):
    banks, support = {}, {}
    for site in original.SITES:
        h, q, cv, ols, analytic = original.prior.source_inputs(cohorts[site])
        extra, peers, _ = model.candidates(h, q, cv)
        for seed in original.SEEDS:
            fold = f'{site}_{seed}'
            path = LOCAL / f'{fold}.npz'
            binding = dict(registration=key, fold=fold, paths=frozen['banks'][fold])
            receipt = LOCAL / f'{fold}_freeze.json'
            if not receipt.exists():
                with np.load(study.checked(binding['paths']), allow_pickle=False) as z:
                    for k, a in (('q', q), ('cv', cv), ('ols4', ols)):
                        np.testing.assert_array_equal(z[k], a)
                    control, joint = z['base_additive'], z['base_joint']
                neural = menu.self_forecast(h, site, seed, weights[fold], producer)
                a, aa = model.replace_recipients(cv, control, extra[:, 0], q)
                b, bb = model.replace_recipients(cv, control, extra[:, 1], q)
                np.testing.assert_array_equal(aa, bb)
                p = original.exp.previous.save_bank(path, dict(q=q, cv=cv, ols4=ols, control=control, joint=joint,
                    bank=np.concatenate((analytic, neural[:, None]), axis=1), extra=extra, own=a, peer=b, uses=aa))
                base.immutable(receipt, dict(binding=binding, bank=p, peer_rows=int((peers >= 0).sum()),
                                             matched_replacements=int(aa.sum())))
            saved = study.read(receipt)
            if saved['binding'] != binding or study.checked(saved['bank']) != path:
                raise ValueError('Frozen candidate receipt changed')
            banks[fold] = saved['bank']
            support[fold] = {k: saved[k] for k in ('peer_rows', 'matched_replacements')}
            heartbeat('freeze', fold, len(banks))
    base.immutable(OUT / 'freeze.json', dict(registration=key, banks=banks, support=support,
                                            future_arrays_loaded=False, original_2pct=True))


def extra_count(cv, joint, extra, q):
    caps = np.linalg.norm(extra - cv[:, None], axis=-1).max(axis=-1)
    native = np.linalg.norm(joint - cv, axis=-1).max(axis=-1)
    count = np.zeros(2, dtype=int)
    for k in np.unique(q[native > 0.]):
        at = q == k
        radii = native[at & (native > 0.)]
        if len(radii) == 2:
            peak = caps[at, :, None]
            legal = (peak > 0.) & (peak >= radii - 1e-10)
            np.testing.assert_array_equal(legal[:, 0], legal[:, 1])
            count += legal.sum((0, 2))
    if count[0] != count[1]:
        raise ValueError('New menus offer different numbers of legal slot candidates')
    return int(count[0])


def solve_record(z, y, valid, cut, old):
    cv, ols, control, joint, q, bank, extra = (z[k] for k in ('cv', 'ols4', 'control', 'joint', 'q', 'bank', 'extra'))
    support = extra_count(cv, joint, extra, q)
    updated = {}
    for i, arm in enumerate(ARMS):
        if support:
            # Infeasibility certificates for the old menu do not transfer to the larger menu.
            status = old['state'] if not old['edits'] else 'eligible'
            row = original.assess(cv, ols, np.concatenate((bank, extra[:, i:i+1]), axis=1),
                                  control, joint, y, valid, q, cut, dict(state=status))
            if row['edits'] != old['edits'] or np.any(np.asarray(row['relaxed_pair_native']) + 1e-8
                                                     < old['relaxed_pair_native']):
                raise ValueError('Adding a path removed a base choice or changed the edit budget')
            for objective in original.OBJECTIVES:
                was = parent.support(old, objective)
                got = parent.support(row, objective)
                if was is not None and row['solve'][objective]['status'] == 'optimal' and (got is None or got < was - 1e-8):
                    raise ValueError('New menu lost a verified base witness')
            updated[arm] = row
        else:
            updated[arm] = old

    direct = None
    if old['edits']:
        a, b = z['own'], z['peer']
        ca = parent.actual_contrast(cv, ols, a, control, y, valid, q, cut)
        cb = parent.actual_contrast(cv, ols, b, control, y, valid, q, cut)
        cp = parent.actual_contrast(cv, ols, b, a, y, valid, q, cut)
        if all(x is not None for x in (ca, cb, cp)):
            direct = dict(self_transient=ca, peer_transient=cb, peer_vs_self=cp)
    return dict(new_slot_candidates=support, replacements=int(z['uses'].sum()), menus=updated, direct=direct)


def summarize(rows):
    arms = {a: original.summary([dict(record=x['record'], **x['menus'][a]) for x in rows]) for a in ARMS}
    direct = [x['direct'] for x in rows if x['direct'] is not None]
    realized = {}
    for key in (*ARMS, 'peer_vs_self'):
        v = [d[key] for d in direct]
        realized[key] = dict(estimable=len(v), qualified=sum(r['qualified'] for r in v),
            qualified_interaction=sum(r['qualified'] and r['pair_native'][1] > parent.frontier.INTERACTION_TOL for r in v),
            accuracy_pp=np.mean([r['accuracy_pp'] for r in v], axis=0).tolist() if v else None,
            pair_native=np.mean([r['pair_native'] for r in v], axis=0).tolist() if v else None)
    return dict(menus=arms, realized=realized, new_slot_candidates=sum(x['new_slot_candidates'] for x in rows),
                matched_replacements=sum(x['replacements'] for x in rows),
                views_with_new_slot=sum(x['new_slot_candidates'] > 0 for x in rows))


def score(key, cohorts, before):
    fp = OUT / 'freeze.json'
    frozen = study.read(fp)
    if (frozen['registration'] != key or frozen['future_arrays_loaded'] or len(frozen['banks']) != 21):
        raise ValueError('Require all prefix-only predictions frozen')
    sources = original.exp.loader(cohorts)
    sites, receipts, all_rows = {}, {}, []
    for site, src in sources.items():
        cut = np.quantile(np.concatenate([x['e'] for s, x in sources.items() if s != site]), [.25, .75])
        rows = []
        for seed in original.SEEDS:
            fold = f'{site}_{seed}'
            p = LOCAL / f'{fold}_score.json'
            binding = dict(registration=key, freeze=study.artifact(fp), paths=frozen['banks'][fold],
                           previous=before['checkpoints'][fold])
            if not p.exists():
                old = {r['record']: r for r in study.read(study.checked(binding['previous']))['rows']}
                with np.load(study.checked(binding['paths']), allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['q'], src['q'])
                    np.testing.assert_array_equal(f['cv'], src['cv'])
                    z = dict(f)
                fold_rows = []
                for record in np.unique(src['rec']):
                    at = src['rec'] == record
                    row = solve_record({k: a[at] for k, a in z.items()}, src['y'][at], src['v'][at], cut, old[str(record)])
                    fold_rows.append(dict(record=f'{site}:{record}', **row))
                base.immutable(p, dict(binding=binding, rows=fold_rows))
            saved = study.read(p)
            if saved['binding'] != binding:
                raise ValueError('Scoring checkpoint no longer matches inputs')
            rows.extend(saved['rows'])
            receipts[fold] = study.artifact(p)
            heartbeat('score', fold, len(receipts))
        sites[site] = summarize(rows)
        all_rows.extend(rows)
    totals = summarize(all_rows)
    if totals['matched_replacements'] != sum(x['matched_replacements'] for x in frozen['support'].values()):
        raise ValueError('Frozen replacement counts did not replay')
    def ci(group, index):
        values = [x['realized']['peer_vs_self'][group] for x in sites.values()]
        x = np.asarray([v[index] for v in values if v is not None])
        draws = np.random.default_rng(3183).choice(x, (3000, len(x))).mean(1)
        return dict(sites=len(x), point=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist())
    result = dict(result_source='fresh_run', inputs='cached_verified', registration=key, freeze=study.artifact(fp),
        prior=before['totals'], totals=totals, sites=sites, checkpoints=receipts,
        realized_peer_vs_self=dict(accuracy=ci('accuracy_pp', 2), full_grid_pair=ci('pair_native', 1)),
        original_2pct=True, independent_confirmation=False, reserved_outcomes_opened=False,
        training=False, stage5c=False, smc=False, submission_ready=False)
    base.immutable(OUT / 'readout.json', result)
    print(study.json.dumps(dict(totals=totals, paired=result['realized_peer_vs_self']), indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('freeze', 'score'))
    args = ap.parse_args()
    menu.torch.set_num_threads(4)
    menu.torch.set_num_interop_threads(1)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, weights, producer, frozen, before = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, weights, producer, frozen)
        else:
            score(key, cohorts, before)


if __name__ == '__main__':
    main()
