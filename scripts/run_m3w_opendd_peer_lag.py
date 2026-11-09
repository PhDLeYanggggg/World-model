"""Test prefix-validated lagged peer motion at the frozen OpenDD edit budget."""

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
from scripts import run_m3w_opendd_peer_transient as prior
from src.world_model import m3w_peer_lag as lag

base, study, np = prior.base, prior.study, prior.np
original = prior.original
OUT = prior.OUT.parent / 'opendd_peer_lag_v1'
LOCAL = prior.LOCAL.parent / OUT.name


def register():
    key, cohorts, _, _, _, expanded = prior.register()
    fp = prior.OUT / 'freeze.json'
    rp = prior.OUT / 'readout.json'
    f, r = study.read(fp), study.read(rp)
    if (f['registration'] != key or r['registration'] != key or study.checked(r['freeze']) != fp
            or f['future_arrays_loaded'] or not r['original_2pct'] or r['reserved_outcomes_opened']):
        raise ValueError('Require the frozen common OpenDD menu and completed peer study')
    path = OUT / 'registration.json'
    files = (__file__, lag.__file__, prior.__file__, prior.model.__file__,
             original.__file__, prior.parent.frontier.__file__, OUT / 'protocol.md')
    base.immutable(path, dict(implementation=[study.artifact(Path(x)) for x in files], cohorts=cohorts,
        common_forecasts=study.artifact(fp), transient_result=study.artifact(rp),
        original_reference=study.artifact(prior.parent.OUT / 'readout.json'), sites=original.SITES,
        seeds=original.SEEDS, observed_steps=8, predicted_steps=12, fit_pairs=lag.FIT_PAIRS,
        original_2pct=True, thresholds_fitted=False, development_only=True, reserved_outcomes_opened=False))
    return base.sha(path), cohorts, f, expanded


def heartbeat(phase, name, completed):
    tmp = LOCAL / 'heartbeat.pending'
    tmp.write_text(study.json.dumps(dict(pid=os.getpid(), phase=phase, fold=name, completed=completed,
                                         total=21)) + '\n')
    tmp.replace(LOCAL / 'heartbeat.json')
    print(phase, name, completed, flush=True)


def freeze(key, cohorts, parent):
    banks, support = {}, {}
    for site in original.SITES:
        h, q, cv, ols, _ = original.prior.source_inputs(cohorts[site])
        extra, coverage = lag.candidates(h, q, cv)
        for seed in original.SEEDS:
            fold = f'{site}_{seed}'
            receipt = LOCAL / f'{fold}_freeze.json'
            ref = parent['banks'][fold]
            binding = dict(registration=key, parent=ref)
            if not receipt.exists():
                with np.load(study.checked(ref), allow_pickle=False) as f:
                    for k, a in (('q', q), ('cv', cv), ('ols4', ols)):
                        np.testing.assert_array_equal(f[k], a)
                    fields = {k: f[k] for k in ('q', 'cv', 'ols4', 'control', 'joint', 'bank')}
                a, aa = prior.model.replace_recipients(cv, fields['control'], extra[:, 0], q)
                b, bb = prior.model.replace_recipients(cv, fields['control'], extra[:, 1], q)
                np.testing.assert_array_equal(aa, bb)
                slots = prior.extra_count(cv, fields['joint'], extra, q)
                delta = ~np.isclose(a, b, atol=1e-9, rtol=1e-9).all((1, 2))
                np.testing.assert_array_equal(delta & ~aa, False)
                file = original.exp.previous.save_bank(LOCAL / f'{fold}.npz', dict(
                    **fields, extra=extra, own=a, peer=b, uses=aa))
                base.immutable(receipt, dict(binding=binding, bank=file, support=dict(
                    **coverage, slot_candidates=slots, replacements=int(aa.sum()),
                    differing_replacements=int(delta.sum()))))
            data = study.read(receipt)
            if data['binding'] != binding or study.checked(data['bank']) != LOCAL / f'{fold}.npz':
                raise ValueError('Cannot resume with a changed forecast bank')
            banks[fold], support[fold] = data['bank'], data['support']
            heartbeat('freeze', fold, len(banks))
    base.immutable(OUT / 'freeze.json', dict(registration=key, banks=banks, support=support,
        future_arrays_loaded=False, original_2pct=True))


def paired_interval(sites, field, column):
    x = np.asarray([d['realized']['peer_vs_self'][field][column] for d in sites.values()
                    if d['realized']['peer_vs_self'][field] is not None])
    if not len(x):
        return dict(sites=0, mean=None, ci95=None)
    draws = np.random.default_rng(3183).choice(x, (3000, len(x))).mean(axis=1) if len(x) > 1 else None
    return dict(sites=len(x), mean=float(x.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist() if draws is not None else None)


def score(key, cohorts, expanded):
    fp = OUT / 'freeze.json'
    f = study.read(fp)
    if f['registration'] != key or f['future_arrays_loaded'] or len(f['banks']) != 21:
        raise ValueError('Require the complete pre-score freeze')
    sources = original.exp.loader(cohorts)
    results, receipts, all_rows = {}, {}, []
    for site in original.SITES:
        src = sources[site]
        cut = np.quantile(np.concatenate([x['e'] for s, x in sources.items() if s != site]), [.25, .75])
        rows = []
        for seed in original.SEEDS:
            fold = f'{site}_{seed}'
            p = LOCAL / f'{fold}_score.json'
            binding = dict(registration=key, forecast=f['banks'][fold], common=expanded['checkpoints'][fold])
            if not p.exists():
                old = {r['record']: r for r in study.read(study.checked(binding['common']))['rows']}
                with np.load(study.checked(binding['forecast']), allow_pickle=False) as file:
                    z = {k: file[k] for k in file.files}
                np.testing.assert_array_equal(z['q'], src['q'])
                np.testing.assert_array_equal(z['cv'], src['cv'])
                records = []
                for r in np.unique(src['rec']):
                    at = src['rec'] == r
                    result = prior.solve_record({k: x[at] for k, x in z.items()},
                                                src['y'][at], src['v'][at], cut, old[str(r)])
                    records.append(dict(record=f'{site}:{r}', **result))
                base.immutable(p, dict(binding=binding, rows=records))
            saved = study.read(p)
            if saved['binding'] != binding:
                raise ValueError('Scoring checkpoint does not bind the current forecasts')
            rows.extend(saved['rows'])
            receipts[fold] = study.artifact(p)
            heartbeat('score', fold, len(receipts))
        results[site] = prior.summarize(rows)
        all_rows.extend(rows)
    totals = prior.summarize(all_rows)
    support = sum(map(Counter, f['support'].values()), Counter())
    if (totals['new_slot_candidates'] != support['slot_candidates']
            or totals['matched_replacements'] != support['replacements']):
        raise ValueError('Freeze and score disagree on candidate counts')
    result = dict(result_source='fresh_run', inputs='cached_verified', registration=key, freeze=study.artifact(fp),
        sites=results, totals=totals, support=dict(support), prior=expanded['totals'], checkpoints=receipts,
        realized_peer_vs_self=dict(accuracy=paired_interval(results, 'accuracy_pp', 2),
                                   pair=paired_interval(results, 'pair_native', 1)),
        original_2pct=True, deployment_changed=False, reserved_outcomes_opened=False,
        independent_confirmation=False, stage5c=False, smc=False, submission_ready=False)
    base.immutable(OUT / 'readout.json', result)
    print(study.json.dumps(dict(support=dict(support), totals=totals,
                                realized=result['realized_peer_vs_self']), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    prior.menu.torch.set_num_threads(4)
    prior.menu.torch.set_num_interop_threads(1)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, f, expanded = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, f)
        else:
            score(key, cohorts, expanded)


if __name__ == '__main__':
    main()
