"""Nested location validation of absolute versus keep-relative full-grid scores."""

import argparse
from collections import Counter
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if __name__ == '__main__' and (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import m3w_nested_advantage_data as data
from src.world_model import m3w_nested_advantage as model

prior, run, study, exp = data.prior, data.run, data.study, data.exp
np, torch, roles = data.np, data.torch, data.roles
previous, old = prior.previous, prior.old
OUT = prior.OUT.with_name('opendd_nested_advantage_v1')
PRIVATE = prior.PRIVATE.with_name(OUT.name)
SEEDS, ARMS, STEPS = prior.SEEDS, model.ARMS, 400
POLICIES = (*ARMS, 'inner_selected')
CONTROLS = previous.CONTROLS
RESULT_ARMS = (*CONTROLS, *POLICIES, *(k+'_uniform_mask' for k in POLICIES), 'prior_pair_optional')
FIELDS = (*previous.FIELDS, 'rescued_full_grid', 'destroyed_full_grid')


def register():
    key, context = prior.register()
    paths = [Path(__file__), Path(data.__file__), Path(model.__file__), OUT/'protocol.md',
             prior.OUT/'registration.json', prior.OUT/'producer_freeze.json', prior.OUT/'head_freeze.json',
             prior.OUT/'readout.json', prior.OUT/'generalization_diagnostic.json']
    path = OUT/'registration.json'
    run.once(path, run.stable(dict(inputs=[run.artifact(p) for p in paths], parent_registration=key,
        arms=ARMS, seeds=SEEDS, rank_models=105, count_models=105, inner_models=252, outer_models=84,
        rank_steps=700, count_steps=500, selector_steps=STEPS, batch_slates=32,
        normalization=dict(fit='valid_training_options_only', std_floor=.1, clip=8.),
        inner_selection='equal_validation_location_and_seed_mean_full_grid_gain; nonpositive_keep',
        fixed_tie_order=ARMS, original_2pct=True, outcome_role='exposed_development',
        new_external_outcome_access=False, no_outer_tuning=True, independent_confirmation=False,
        threads=4, interop=1, workers=0)))
    return run.sha(path), context


def producers(reg, context, limit, beat):
    ds = data.NestedData(PRIVATE, reg, context, beat)
    cells = {}
    for omitted in combinations(sorted(ds.sites), 3):
        for seed in SEEDS:
            if limit is not None and len(cells) >= limit:
                return dict(cells=len(cells), updates=ds.updates, complete=False)
            ds.count(omitted, seed)
            key = '_'.join(omitted)+f'_{seed}'
            cells[key] = run.artifact(PRIVATE/'producers'/f'count_{key}.json')
            beat(phase='producer_cell', cell=key, cells=len(cells), updates=ds.updates)
    run.once(OUT/'producer_freeze.json', run.stable(dict(registration=reg, cells=cells,
        rank_models=105, count_models=105, configured_updates=126000)))
    return dict(cells=len(cells), updates=ds.updates, complete=True)


def batcher(table):
    active = [row for row in table if len(row[0])]
    if not active:
        raise ValueError('No training slates')
    def batch(rng):
        counts = np.bincount(rng.integers(len(active), size=32), minlength=len(active))
        picks = [rng.integers(len(row[0]), size=n) for row, n in zip(active, counts, strict=True)]
        return tuple(torch.cat([row[k][ix] for row, ix in zip(active, picks, strict=True)]) for k in range(3))
    return batch


def fit_models(ds, excluded, seed):
    fit = roles.fit_roster(ds.sites, excluded)
    table, dependencies, audits = [], {}, {}
    for donor in fit:
        row, dependencies[donor], audits[donor] = ds.packet(excluded, donor, seed)
        table.append(row)
    norm = model.normalization(table)
    key = '_'.join(sorted(excluded))+f'_{seed}'
    nets, metas, updates = {}, {}, 0
    for arm in ARMS:
        identity = dict(registration=ds.reg, omitted=sorted(excluded), fit=fit, seed=seed, arm=arm,
                        data=dependencies, normalization=norm, target='full_grid')
        net, meta, n = study.fit(identity, str(PRIVATE/'selectors'/f'{key}_{arm}'), seed,
            lambda: model.Selector(arm, norm), STEPS, batcher(table),
            lambda m, x, y, valid: model.objective(m(x), y, valid, relative=arm.endswith('advantage')),
            False, ds.beat)
        updates += n
        nets[arm], metas[arm] = net, meta
    return nets, metas, audits, updates


@torch.inference_mode()
def validation(net, row):
    x, y, valid = row
    if not len(x):
        return dict(slates=0, gain_rate=None, switches=0, rescued=0, destroyed=0)
    scores = net(x)
    choices = model.choose(scores, valid)
    base, chosen = y[:, 0], y[torch.arange(len(y)), choices]
    return dict(slates=len(x), gain_rate=float((chosen.float()-base.float()).mean()),
        switches=int((choices != 0).sum()), rescued=int((chosen & ~base).sum()),
        destroyed=int((base & ~chosen).sum()), anchor_passes=int(base.sum()), chosen_passes=int(chosen.sum()),
        oracle_passes=int((y & valid).any(1).sum()))


def inner(reg, context, limit, beat):
    frozen = run.read(OUT/'producer_freeze.json')
    if frozen['registration'] != reg or len(frozen['cells']) != 105:
        raise ValueError('Freeze all nested producers first')
    ds = data.NestedData(PRIVATE, reg, context, beat)
    cells, updates = {}, 0
    for excluded in combinations(sorted(ds.sites), 2):
        for seed in SEEDS:
            if limit is not None and len(cells) >= limit:
                return dict(cells=len(cells), updates=updates, complete=False)
            key = '_'.join(excluded)+f'_{seed}'
            nets, metas, audits, n = fit_models(ds, excluded, seed)
            updates += n
            val = {}
            for held, valid_site in (excluded, excluded[::-1]):
                row, artifact, audit = ds.packet((held,), valid_site, seed)
                val[held] = dict(site=valid_site, packet=artifact, audit=audit,
                                 arms={k: validation(net, row) for k, net in nets.items()})
            cell = dict(registration=reg, omitted=list(excluded), seed=seed,
                        models=metas, donor_audit=audits, validation=val)
            path = PRIVATE/'inner'/f'{key}.json'
            run.once(path, run.stable(cell))
            cells[key] = run.artifact(path)
            beat(phase='inner_cell', cell=key, cells=len(cells), updates=updates)
    run.once(OUT/'inner_freeze.json', run.stable(dict(registration=reg, cells=cells,
        configured_updates=252*STEPS, all_inner_producers_exclude_validation=True)))
    return dict(cells=len(cells), updates=updates, complete=True)


def select(reg):
    frozen = run.read(OUT/'inner_freeze.json')
    if frozen['registration'] != reg or len(frozen['cells']) != 63:
        raise ValueError('Need all 63 inner cells before model selection')
    sites = sorted(prior.parents.parent.source.SITES)
    records = [run.read(run.checked(p)) for p in frozen['cells'].values()]
    selection = {}
    for held in sites:
        scores, support = {}, {}
        for arm in ARMS:
            values = []
            support[arm] = {}
            for val_site in sites:
                if val_site == held:
                    continue
                cells = [r for r in records if r['omitted'] == sorted((held, val_site))]
                if {r['seed'] for r in cells} != set(SEEDS):
                    raise ValueError('Incomplete inner seed set')
                rates = [r['validation'][held]['arms'][arm]['gain_rate'] for r in cells]
                if any(v is None for v in rates):
                    raise ValueError('Inner validation has no supported slates')
                support[arm][val_site] = rates
                values.append(float(np.mean(rates)))
            scores[arm] = float(np.mean(values))
        selection[held] = dict(selected=model.select_arm(scores), mean_gain=scores, per_validation_site=support)
    run.once(OUT/'selection.json', run.stable(dict(registration=reg, inner=run.artifact(OUT/'inner_freeze.json'),
        choices=selection, outer_outcome_selection=False)))
    return selection


def outer(reg, context, beat):
    selection = select(reg)
    ds = data.NestedData(PRIVATE, reg, context, beat)
    cells, updates = {}, 0
    for args, _ in previous.matched.context_cells():
        held, seed = args[:2]
        nets, metas, audits, n = fit_models(ds, (held,), seed)
        updates += n
        picks = []
        for _, _, _, _, aligned, feat in previous.prior.slate_rows(args):
            row = None
            if aligned is not None:
                x = prior.model.with_incumbent(feat[None])[0]
                valid = torch.ones(x.shape[0], dtype=torch.bool)
                with torch.inference_mode():
                    row = dict(aligned=aligned+1, **{k: int(model.choose(net(x), valid)) for k, net in nets.items()})
                selected = selection[held]['selected']
                row['inner_selected'] = 0 if selected == 'keep' else row[selected]
            picks.append(row)
        key = f'{held}_{seed}'
        path = PRIVATE/'choices'/f'{key}.json'
        run.once(path, run.stable(dict(registration=reg, selected=selection[held]['selected'], picks=picks)))
        cells[key] = dict(models=metas, choices=run.artifact(path), donor_audit=audits)
        beat(phase='outer_cell', cell=key, cells=len(cells), updates=updates)
    if len(cells) != 21:
        raise ValueError('Incomplete outer choice matrix')
    run.once(OUT/'head_freeze.json', run.stable(dict(registration=reg, cells=cells,
        selection=run.artifact(OUT/'selection.json'), configured_updates=84*STEPS, scored=False)))
    return dict(cells=len(cells), updates=updates)


def evaluate(args, picks, old_row):
    d, bank, menu, cut = args[3], args[4], args[5], args[8]
    counts, switches = Counter(), Counter({k: 0 for k in POLICIES})
    outcomes = {k: np.zeros(len(FIELDS)) for k in RESULT_ARMS if k != 'prior_pair_optional'}
    for slate, chosen in zip(previous.prior.slate_rows(args), picks, strict=True):
        ai, r, anchor, pairs, aligned, _ = slate
        if (chosen is None) != (aligned is None) or chosen is not None and chosen['aligned'] != aligned+1:
            raise ValueError('Frozen eligibility changed')
        counts.update(views=1, edits=int(anchor.sum()), eligible=int(chosen is not None))
        cv, y, valid = bank[0][ai], d['y'][ai], d['valid'][ai]
        def outcome(plan):
            p, s = old.measure(d, ai, bank, menu[3], r, plan, cut)
            full = previous.prior.prior.certificate(cv, p, y, valid, cut[0], s)[0]
            return np.asarray((s['dual_positive'], full,
                full and min(s['vs_cv'], s['vs_ols4']) > previous.prior.prior.MATERIAL_PP,
                s['undefined'], not s['undefined'] and not s['risk_pass']), dtype=float)
        base = outcome(anchor)
        if chosen is None:
            selected = dict.fromkeys(outcomes, base)
        else:
            options = np.asarray([base, *(outcome(old.model.exchanged(anchor, *p)) for p in pairs)])
            if set(chosen) != {'aligned', *POLICIES} or any(not 0 <= j < len(options) for j in chosen.values()):
                raise ValueError('Invalid frozen choice')
            selected = dict(incumbent=base, aligned=options[chosen['aligned']], uniform_keep=options.mean(0),
                            mask_oracle=options[options[:, 1].argmax()])
            for k in POLICIES:
                selected[k] = options[chosen[k]]
                selected[k+'_uniform_mask'] = options[1:].mean(0) if chosen[k] else base
                switches[k] += int(chosen[k] != 0)
        for k, value in selected.items():
            outcomes[k] += np.r_[value, value[1] if not base[1] else 0., 1.-value[1] if base[1] else 0.]
    result = dict(counts=dict(counts), switches=dict(switches),
                  arms={k: dict(zip(FIELDS, z.tolist(), strict=True)) for k, z in outcomes.items()})
    result['arms']['prior_pair_optional'] = old_row['arms']['pair_optional']
    if result['counts'] != old_row['counts']:
        raise ValueError('Outer roster or forecast budget changed')
    for k in CONTROLS:
        if any(abs(result['arms'][k][f]-old_row['arms'][k].get(f, 0)) > 1e-9 for f in previous.FIELDS):
            raise ValueError('Frozen control failed to reproduce')
    return result


def summary(groups):
    rows = [r for cells in groups.values() for r in cells]
    counts = roles.total_fields([r['counts'] for r in rows], ('views', 'edits', 'eligible'))
    arms = {k: roles.total_fields([r['arms'][k] for r in rows], FIELDS) for k in RESULT_ARMS}
    switches = roles.total_fields([r['switches'] for r in rows], POLICIES)
    draw = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    pairs = [('inner_selected', k) for k in ('incumbent', 'inner_selected_uniform_mask', 'prior_pair_optional')]
    pairs += [(k, 'incumbent') for k in ARMS]
    pairs += [('linear_advantage', 'linear_absolute'), ('mlp_advantage', 'mlp_absolute'),
              ('linear_absolute', 'mlp_absolute'), ('linear_advantage', 'mlp_advantage')]
    for a, b in pairs:
        v = np.asarray([np.mean([100*(r['arms'][a]['full_grid']-r['arms'][b]['full_grid'])/r['counts']['views']
                                 for r in cells]) for cells in groups.values()])
        contrasts[f'{a}_minus_{b}'] = dict(point_pp=float(v.mean()),
            ci95_pp=np.quantile(v[draw].mean(1), [.025, .975]).tolist(), by_location_pp=v.tolist())
    return dict(counts=counts, arms=arms, switches=switches, contrasts=contrasts,
        locations=list(groups), bootstrap_draws=3000,
        primary_supported=all(contrasts[f'inner_selected_minus_{k}']['ci95_pp'][0] > 0
            for k in ('incumbent', 'inner_selected_uniform_mask', 'prior_pair_optional')))


def score(reg, beat):
    frozen = run.read(OUT/'head_freeze.json')
    if frozen['registration'] != reg or len(frozen['cells']) != 21:
        raise ValueError('All outer choices must be frozen before scoring')
    run.checked(frozen['selection'])
    reference = run.read(prior.OUT/'readout.json')
    result = {}
    for args, _ in previous.matched.context_cells():
        held, seed = args[:2]
        meta = frozen['cells'][f'{held}_{seed}']
        for m in meta['models'].values():
            run.checked(m['artifact'])
            roles.verify_role(m['identity'], reference['by_location'], (held,))
        chosen = run.read(run.checked(meta['choices']))
        if chosen['registration'] != reg:
            raise ValueError('Choice registration changed')
        old_row = next(r for r in reference['by_location'][held] if r['seed'] == seed)
        result.setdefault(held, []).append(dict(seed=seed, **evaluate(args, chosen['picks'], old_row)))
        beat(phase='score_cell', held=held, seed=seed)
    if len(result) != 7 or any({r['seed'] for r in cells} != set(SEEDS) for cells in result.values()):
        raise ValueError('Incomplete scoring matrix')
    report = dict(registration=reg, freeze=run.artifact(OUT/'head_freeze.json'), summary=summary(result),
        by_location=result, result_source='fresh_run', evidence_role='exposed_development',
        deployment_changed=False, original_2pct=True, new_external_outcome_access=False)
    run.once(OUT/'readout.json', run.stable(report))
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'producers', 'inner', 'outer', 'score'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and (args.phase not in ('producers', 'inner') or args.limit < 1):
        raise ValueError('Limit is only for a producer/inner pilot')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for sub in ('producers', 'teachers', 'packets', 'selectors', 'inner', 'choices'):
        (PRIVATE/sub).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = PRIVATE/f'events_{time.time_ns()}_{os.getpid()}.jsonl'
        def beat(**event):
            seconds = time.monotonic()-started
            row = dict(pid=os.getpid(), seconds=round(seconds, 3),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **event)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            with log.open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row, allow_nan=False), flush=True)
            if seconds > 12*3600:
                raise TimeoutError('12-hour checkpoint boundary; resume the same command')
            if shutil.disk_usage(PRIVATE).free < 10*2**30:
                raise OSError('Below 10 GiB disk reserve')
        reg, context = register()
        if args.phase == 'register':
            result = dict(registration=reg, updates=0)
        elif args.phase == 'producers':
            result = producers(reg, context, args.limit, beat)
        elif args.phase == 'inner':
            result = inner(reg, context, args.limit, beat)
        elif args.phase == 'outer':
            result = outer(reg, context, beat)
        else:
            result = score(reg, beat)
        beat(phase='complete', command=args.phase, result=result)


if __name__ == '__main__':
    main()
