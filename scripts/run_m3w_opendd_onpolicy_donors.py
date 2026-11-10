"""Nested policy-matched donor training, frozen choices, and separate scoring."""

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
from scripts import run_m3w_opendd_incumbent_slate as previous
from src.world_model import m3w_onpolicy_donors as roles

run, exp, study, np, torch = previous.run, previous.exp, previous.study, previous.np, previous.torch
old, parents, model = previous.old, previous.parents, previous.model
OUT = previous.OUT.with_name('opendd_onpolicy_donors_v1')
PRIVATE = previous.PRIVATE.with_name(OUT.name)
SEEDS = (17, 29, 43)


def register():
    context = exp.register()
    paths = [Path(__file__), Path(roles.__file__), Path(model.__file__), Path(study.__file__),
             Path(study.prior.__file__), Path(parents.base.model.__file__), OUT/'protocol.md',
             previous.OUT/'registration.json', previous.OUT/'runtime_amendment.json',
             previous.OUT/'head_freeze.json', previous.OUT/'readout.json', exp.OUT/'head_freeze.json']
    path = OUT/'registration.json'
    run.once(path, run.stable(dict(inputs=[run.artifact(p) for p in paths],
        context_registration=context[0], seeds=SEEDS, original_2pct=True,
        producer_ranks=105, producer_counts=63, selector_heads=42,
        rank_steps=700, count_steps=500, selector_steps=400, batch_slates=32,
        threads=4, interop=1, workers=0, outcome_access='exposed_development_only',
        independent_confirmation=False, new_external_outcome_access=False,
        unresolved_donor_score_policy='omit_and_report', no_held_tuning=True)))
    return run.sha(path), context


def load_data():
    sites = {s: parents.parent.prior.prior.old.load_site(s) for s in parents.parent.source.SITES}
    for s, d in sites.items():
        with np.load(parents.parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz', allow_pickle=False) as z:
            d.update(recordings=z['recording'], query_rec=z['recording'], starts=z['start'])
    return sites, {s: parents.parent.prior.analytic(d) for s, d in sites.items()}


def cuts(sites, fit):
    return np.quantile(np.concatenate([sites[s]['cut_samples'] for s in fit]), [.25, .75])


def cache(file, identity, build):
    receipt = file.with_suffix('.json')
    if not receipt.exists():
        arrays, audit = build()
        file.parent.mkdir(parents=True, exist_ok=True)
        pending = file.with_suffix('.pending')
        with pending.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        pending.replace(file)
        run.once(receipt, run.stable(dict(identity=identity, artifact=run.artifact(file), audit=audit)))
    meta = run.read(receipt)
    if meta['identity'] != run.stable(identity):
        raise ValueError('Cache role changed')
    run.checked(meta['artifact'])
    with np.load(file, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, meta, run.artifact(receipt)


def rank_producer(reg, omitted, seed, sites, banks, beat):
    omitted = sorted(omitted)
    fit = roles.fit_roster(sites, omitted)
    name = PRIVATE/'producers'/('rank_'+'_'.join(omitted)+f'_{seed}')
    identity = dict(registration=reg, omitted=omitted, fit=fit, seed=seed, kind='nominee_ranker')
    builder = lambda: study.prior.model.SwapValue(False)
    if name.with_suffix('.json').exists():
        meta = run.read(name.with_suffix('.json'))
        if meta['identity'] != identity:
            raise ValueError('Rank producer role changed')
        roles.verify_role(meta['identity'], sites, omitted)
        return study.load(meta, builder, study.RANK_STEPS), meta, 0
    data = [study.prior.prepare(sites[s], banks[s], cuts(sites, fit)[0]) for s in fit]
    net, meta, updates = study.fit(identity, str(name), seed, builder, study.RANK_STEPS,
        lambda rng: study.prior.batch(data, rng),
        lambda n, x, y, pair, size: study.prior.model.objective(n(x, pair, size), y, True), False, beat)
    run.once(name.with_suffix('.json'), run.stable(meta))
    return net, meta, updates


def count_producer(reg, omitted, seed, sites, banks, beat):
    omitted = sorted(omitted)
    fit = roles.fit_roster(sites, omitted)
    name = PRIVATE/'producers'/('count_'+'_'.join(omitted)+f'_{seed}')
    role = dict(registration=reg, omitted=omitted, fit=fit, seed=seed, kind='recording_count')
    if name.with_suffix('.json').exists():
        saved = run.read(name.with_suffix('.json'))
        if saved['role'] != role:
            raise ValueError('Count producer role changed')
        roles.verify_role(saved['model']['identity'], sites, omitted)
        for dependency in saved['dependencies'].values():
            run.checked(dependency)
        net = study.load(saved['model'], lambda: study.model.CountHead(saved['width']), study.HEAD_STEPS)
        return net, saved, 0
    train, dependencies, audit, updates = [], {}, {}, 0
    for target in fit:
        triple = sorted((*omitted, target))
        rank, rm, n = rank_producer(reg, triple, seed, sites, banks, beat)
        updates += n
        roles.verify_role(rm['identity'], sites, triple)
        inner_fit = roles.fit_roster(sites, triple)
        file = PRIVATE/'teachers'/('_'.join(omitted)+f'_{seed}_{target}.npz')
        identity = dict(registration=reg, omitted=omitted, label_site=target, rank=rm['artifact'],
                        cutoff_fit=inner_fit, seed=seed)
        def build():
            menu = study.menus(sites[target], banks[target], rank)
            labels, tally = study.labels(sites[target], banks[target], menu, cuts(sites, inner_fit))
            return dict(**menu, local=labels[0], recording=labels[1]), tally
        arrays, meta, dependencies[target] = cache(file, identity, build)
        menu = {k: arrays[k] for k in ('x', 'legal', 'row', 'arm')}
        train.append((menu, (arrays['local'], arrays['recording'])))
        audit[target] = meta['audit']
        beat(phase='teacher', omitted=omitted, target=target, seed=seed, audit=meta['audit'])
    width = train[0][0]['x'].shape[-1]
    identity = dict(**role, teachers=dependencies)
    net, meta, n = study.fit(identity, str(name), seed, lambda: study.model.CountHead(width),
        study.HEAD_STEPS, study.batches(train, 1),
        lambda m, x, legal, y: study.model.objective(m(x, legal), y), False, beat)
    saved = dict(role=role, width=width, dependencies=dependencies, teacher_audit=audit, model=meta)
    run.once(name.with_suffix('.json'), run.stable(saved))
    return net, saved, updates+n


def producers(reg, sites, banks, limit, beat):
    cells, updates = {}, 0
    for omitted in combinations(sorted(sites), 2):
        for seed in SEEDS:
            if limit is not None and len(cells) >= limit:
                return dict(cells=len(cells), updates=updates, complete=False)
            _, saved, n = count_producer(reg, omitted, seed, sites, banks, beat)
            updates += n
            name = '_'.join(omitted)+f'_{seed}'
            cells[name] = run.artifact(PRIVATE/'producers'/f'count_{name}.json')
            beat(phase='producer_cell', cell=name, cells=len(cells), updates=updates)
    run.once(OUT/'producer_freeze.json', run.stable(dict(registration=reg, cells=cells,
        rank_models=105, count_models=63, configured_updates=105*700+63*500, independent_confirmation=False)))
    return dict(cells=len(cells), updates=updates, complete=True)


def donor(reg, held, site, seed, sites, banks, maps, heads, beat):
    omitted = sorted((held, site))
    count, cm, n = count_producer(reg, omitted, seed, sites, banks, beat)
    if n:
        raise ValueError('Freeze all producers before selector training')
    rm = heads['rank_heads']['rank_'+'_'.join(omitted)+f'_{seed}']
    roles.verify_role(rm['identity'], sites, omitted)
    if rm['identity']['seed'] != seed or rm['identity']['registration'] != heads['registration']:
        raise ValueError('Unexpected archived nominee producer')
    rank = study.load(rm, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
    d, bank = sites[site], banks[site]
    plain = parents.prefix.menu_for(d, bank, rank)
    width = plain[0].shape[-1]
    if cm['width'] != width:
        raise ValueError('Count and recipient feature schemas differ')
    menu, _ = exp.augment(d, bank, plain, maps[site])
    cut = cuts(sites, roles.fit_roster(sites, omitted))
    identity = dict(registration=reg, held=held, donor=site, seed=seed,
        count=cm['model']['artifact'], rank=rm['artifact'], cutoff_fit=roles.fit_roster(sites, omitted))
    file = PRIVATE/'donors'/f'{held}_{site}_{seed}.npz'
    def build():
        x, legal, rows, path, _ = menu
        with torch.inference_mode():
            logits = count(torch.from_numpy(plain[0]), torch.from_numpy(legal)).numpy()
        move = parents.base.model.movement_prefix(bank[0], path, rows)
        rng = np.random.default_rng(seed+913)
        features, labels, anchors, lengths, records = [], [], [], [], []
        tally = Counter()
        for rec in sorted(set(d['rec'])):
            ai, qi, r = old.indices(d, rec, rows)
            anchor, cap, status = roles.score_anchor(logits[qi], legal[qi], move[qi],
                                                     d['recordings'][qi], d['starts'][qi])
            tally.update(recordings=1, **{status: 1})
            if anchor is None:
                continue
            pairs = old.model.exchange_menu(anchor, legal[qi], move[qi], cap)
            if not len(pairs):
                tally.update(no_exchange=1)
                continue
            tally.update(legal_pairs=len(pairs))
            if len(pairs) > old.SAMPLE:
                pairs = pairs[np.sort(rng.choice(len(pairs), size=old.SAMPLE, replace=False))]
            feat = old.model.features(x[qi], legal[qi], move[qi], anchor, cap, pairs)[0]
            # Outcomes are consulted only after logits, plans, pairs and features are fixed.
            if np.linalg.norm(bank[0][ai]-d['y'][ai], axis=-1)[d['valid'][ai]].sum() <= 0.:
                tally.update(zero_reference_mass=1)
                continue
            s0 = old.measure(d, ai, bank, path, r, anchor, cut)[1]
            sy = [bool(old.measure(d, ai, bank, path, r, old.model.exchanged(anchor, *p), cut)[1]['dual_positive'])
                  for p in pairs]
            features.append(feat)
            labels.extend(sy)
            anchors.append(bool(s0['dual_positive']))
            lengths.append(len(pairs))
            records.append(str(rec))
        feature_width = 4*x.shape[1]+9
        flat = np.concatenate(features) if features else np.empty((0, feature_width), np.float32)
        y = torch.tensor(labels, dtype=torch.bool)
        px, py, valid = previous.prior.padded(torch.from_numpy(flat), y, y, lengths)
        a = np.asarray(anchors, bool)
        arrays = dict(x=px.numpy(), y=py[..., 0].numpy(), valid=valid.numpy(), anchor=a,
                      recordings=np.asarray(records, dtype='U128'))
        if not np.isfinite(arrays['x']).all():
            raise ValueError('Nonfinite donor features')
        tally.update(slates=len(a), options=len(labels), anchor_positive=int(a.sum()), option_positive=int(y.sum()))
        return arrays, dict(tally)
    arrays, meta, receipt = cache(file, identity, build)
    return tuple(torch.from_numpy(arrays[k]) for k in ('x', 'y', 'valid', 'anchor')), receipt, meta['audit']


def fit(reg, context, sites, banks, beat):
    producers_path = OUT/'producer_freeze.json'
    frozen = run.read(producers_path)
    if frozen['registration'] != reg or len(frozen['cells']) != 63:
        raise ValueError('All role-safe producers must be frozen first')
    for artifact in frozen['cells'].values():
        run.checked(artifact)
    heads, assets = context[6], context[10]
    maps = {s: exp.road.raster_for(s, assets[s])[0] for s in sites}
    cells, updates = {}, 0
    for args, _ in previous.matched.context_cells():
        held, seed, width = args[0], args[1], args[6]
        key, table, data, audits = f'{held}_{seed}', [], {}, {}
        for site in sites:
            if site != held:
                row, data[site], audits[site] = donor(reg, held, site, seed, sites, banks, maps, heads, beat)
                table.append(row)
        nets, artifacts = {}, {}
        for arm in previous.MODELS:
            identity = dict(registration=reg, held=held, train=roles.fit_roster(sites, (held,)),
                            seed=seed, arm=arm, data=data)
            def loss(net, x, y, valid, anchor):
                masked = exp.design(x.flatten(0, 1), width, 'aligned').reshape_as(x)
                return model.objective(net(model.with_incumbent(masked)), y, valid, anchor, pairwise=arm == 'pair')
            nets[arm], artifacts[arm], n = study.fit(identity, str(PRIVATE/f'{key}_{arm}'), seed,
                lambda: model.ListRanker(table[0][0].shape[-1]+1), old.STEPS, previous.batcher(table), loss, False, beat)
            updates += n
        path = PRIVATE/f'{key}_choices.json'
        run.once(path, run.stable(dict(registration=reg, picks=previous.choose(args, nets))))
        cells[key] = dict(models=artifacts, choices=run.artifact(path), donor_audit=audits,
            donor_slates=sum(len(d[0]) for d in table), donor_options=sum(int(d[2].sum()) for d in table),
            incumbent_positives=sum(int(d[3].sum()) for d in table))
        beat(phase='selector_cell', cell=key, cells=len(cells), updates=updates)
    if len(cells) != 21:
        raise ValueError('Incomplete frozen selector matrix')
    run.once(OUT/'head_freeze.json', run.stable(dict(registration=reg, producers=run.artifact(producers_path),
        cells=cells, cells_count=21, configured_updates=16800, outcome_scoring_this_phase=False)))
    return dict(cells=21, updates=updates)


def summarize(groups, reference):
    flat = [r for rows in groups.values() for r in rows]
    fields = (*previous.FIELDS, 'rescued_full_grid', 'destroyed_full_grid')
    arms = {k: roles.total_fields([r['arms'][k] for r in flat], fields) for k in previous.ARMS}
    counts = roles.total_fields([r['counts'] for r in flat], ('views', 'edits', 'eligible'))
    switches = roles.total_fields([r['switches'] for r in flat], previous.POLICIES)
    if counts != reference['summary']['counts']:
        raise ValueError('The frozen roster or budget changed')
    for k in previous.CONTROLS:
        for field in previous.FIELDS:
            if abs(arms[k][field]-reference['summary']['arms'][k].get(field, 0)) > 1e-9:
                raise ValueError('Archived control did not reproduce')
    draw = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    comparisons = {}
    for comparator in ('incumbent', 'pair_uniform_mask', 'old_pair_optional', 'point_optional', 'pair_forced'):
        values = []
        for site, rows in groups.items():
            prior_rows = {r['seed']: r for r in reference['by_location'][site]}
            delta = []
            for row in rows:
                b = (prior_rows[row['seed']]['arms']['pair_optional'] if comparator == 'old_pair_optional'
                     else row['arms'][comparator])
                delta.append(100.*(row['arms']['pair_optional']['full_grid']-b['full_grid'])/row['counts']['views'])
            values.append(float(np.mean(delta)))
        v = np.asarray(values)
        comparisons[comparator] = dict(point_pp=float(v.mean()),
            ci95_pp=np.quantile(v[draw].mean(1), [.025, .975]).tolist(), by_location_pp=values)
    return dict(arms=arms, counts=counts, switches=switches, contrasts=comparisons,
        primary_supported=all(comparisons[k]['ci95_pp'][0] > 0
            for k in ('incumbent', 'pair_uniform_mask', 'old_pair_optional')), bootstrap_draws=3000)


def score(reg, beat):
    path = OUT/'head_freeze.json'
    frozen = run.read(path)
    if frozen['registration'] != reg or frozen['cells_count'] != 21:
        raise ValueError('Complete choice freeze required')
    result = {}
    for args, _ in previous.matched.context_cells():
        held, seed = args[:2]
        meta = frozen['cells'][f'{held}_{seed}']
        for m in meta['models'].values():
            run.checked(m['artifact'])
            if m['identity']['held'] != held or held in m['identity']['train']:
                raise ValueError('Selector used held labels')
        chosen = run.read(run.checked(meta['choices']))
        if chosen['registration'] != reg:
            raise ValueError('Unbound choice artifact')
        result.setdefault(held, []).append(dict(seed=seed, **previous.evaluate(args, chosen['picks'])))
        beat(phase='score_cell', held=held, seed=seed)
    if len(result) != 7 or any({r['seed'] for r in rows} != set(SEEDS) for rows in result.values()):
        raise ValueError('Incomplete score matrix')
    summary = summarize(result, run.read(previous.OUT/'readout.json'))
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, freeze=run.artifact(path),
        summary=summary, by_location=result, result_source='fresh_run', evidence_role='exposed_development',
        new_external_outcome_access=False, deployment_changed=False, original_2pct=True)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'producers', 'fit', 'score'))
    parser.add_argument('--limit-producers', type=int)
    args = parser.parse_args()
    if args.limit_producers is not None and (args.phase != 'producers' or not 1 <= args.limit_producers <= 63):
        raise ValueError('Limit applies only to a training pilot')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for sub in ('producers', 'teachers', 'donors'):
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
                raise OSError('Below the registered 10 GiB disk reserve')
        reg, context = register()
        if args.phase == 'register':
            result = dict(registration=reg, updates=0)
        elif args.phase == 'score':
            result = score(reg, beat)
        else:
            sites, banks = load_data()
            result = (producers(reg, sites, banks, args.limit_producers, beat) if args.phase == 'producers'
                      else fit(reg, context, sites, banks, beat))
        beat(phase='complete', command=args.phase, result=result)


if __name__ == '__main__':
    main()
