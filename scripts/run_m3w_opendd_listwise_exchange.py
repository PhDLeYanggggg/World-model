"""Fit the same-mask listwise exchange controls on donor locations only."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__' and (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_full_grid_exchange as prior
from scripts import run_m3w_opendd_exchange_risk_heads as parents
from src.world_model import m3w_exchange_listwise as ranker

run, old, np, torch = parents.run, parents.old, parents.np, parents.torch
exp, study, matched = parents.exp, parents.study, parents.matched
OUT = prior.OUT.with_name('opendd_listwise_exchange_v1')
PRIVATE = prior.PRIVATE.with_name(OUT.name)
ARMS = ('point_original', 'list_original', 'list_full_grid')
CONTROLS = ('incumbent', 'aligned', 'uniform')
FIELDS = ('original', 'full_grid', 'material_full_grid')
BATCH = 32


def targets(d, bank, menu, cut, seed, saved):
    x, legal, rows, path, _ = menu
    cv = bank[0]
    move = parents.base.model.movement_prefix(cv, path, rows)
    rng = np.random.default_rng(seed+913)
    labels, lengths = [], []
    for rec in sorted(set(d['rec'])):
        ai, q, r = old.indices(d, rec, rows)
        c, y, v = cv[ai], d['y'][ai], d['valid'][ai]
        if np.linalg.norm(c-y, axis=-1)[v].sum() <= 0.:
            continue
        l, m = legal[q], move[q]
        schedules = (np.tile([0., 1., -1.], (len(l), 1)),
                     np.column_stack((np.zeros(len(l)), rng.standard_normal((len(l), 2)))))
        seen = set()
        for scores in schedules:
            a = study.model.streaming_count(scores, l, d['recordings'][q], d['starts'][q], study.RATE)
            if tuple(a) in seen:
                continue
            seen.add(tuple(a))
            cap = float(m[np.arange(len(a)), a].sum())
            pairs = old.model.exchange_menu(a, l, m, cap)
            if not len(pairs):
                continue
            if len(pairs) > old.SAMPLE:
                pairs = pairs[np.sort(rng.choice(len(pairs), size=old.SAMPLE, replace=False))]
            feat = old.model.features(x[q], l, m, a, cap, pairs)[0]
            if not np.array_equal(feat, saved[0][len(labels):len(labels)+len(pairs)].numpy()):
                raise ValueError('Donor slate differs from the archived causal design')
            lengths.append(len(pairs))
            for pair in pairs:
                p, s = old.measure(d, ai, bank, path, r, old.model.exchanged(a, *pair), cut)
                if bool(s['dual_positive']) != bool(saved[3][len(labels)] == 3):
                    raise ValueError('Original donor label changed')
                labels.append(prior.certificate(c, p, y, v, cut[0], s)[0])
    if len(labels) != len(saved[0]):
        raise ValueError('Donor option population changed')
    return dict(full_grid=labels, lengths=lengths)


def padded(x, original, full_grid, lengths):
    if sum(lengths) != len(x) or len(x) != len(original) or len(x) != len(full_grid):
        raise ValueError('Misaligned causal lists and donor labels')
    pos = torch.zeros((len(lengths), old.SAMPLE, 2), dtype=torch.bool)
    mask = torch.zeros_like(pos[..., 0])
    feat = x.new_zeros((len(lengths), old.SAMPLE, x.shape[-1]))
    idx = 0
    for j, n in enumerate(lengths):
        if not 1 <= n <= old.SAMPLE:
            raise ValueError('Unexpected donor slate length')
        feat[j, :n], mask[j, :n] = x[idx:idx+n], True
        pos[j, :n, 0], pos[j, :n, 1] = original[idx:idx+n], full_grid[idx:idx+n]
        idx += n
    if (pos[..., 1] & ~pos[..., 0]).any():
        raise ValueError('Full-grid training positives must also pass the original test')
    return feat, pos, mask


def donor(held, site, seed, sites, banks, maps, heads, reg, risk_reg, context_reg):
    saved, upstream = parents.data_for(held, site, seed, sites, banks, maps, heads, risk_reg, context_reg)
    omitted = sorted((held, site))
    donors = [s for s in sites if s not in omitted]
    meta = heads['rank_heads']['rank_'+'_'.join(omitted)+f'_{seed}']
    if meta['identity']['omitted'] != omitted or meta['identity']['fit'] != donors:
        raise ValueError('Donor cutoffs or forecast bank used held outcomes')
    path = PRIVATE/'donors'/f'{held}_{site}_{seed}.json'
    identity = dict(registration=reg, archived_donor=upstream, producer=meta['artifact']['sha256'], fit=donors)
    if not path.exists():
        net = study.load(meta, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
        d, bank = sites[site], banks[site]
        menu, _ = exp.augment(d, bank, parents.prefix.menu_for(d, bank, net), maps[site])
        cut = np.quantile(np.concatenate([sites[t]['cut_samples'] for t in donors]), [.25, .75])
        run.once(path, run.stable(dict(identity=identity, **targets(d, bank, menu, cut, seed, saved))))
    readout = run.read(path)
    if readout['identity'] != identity:
        raise ValueError('Unbound donor label cache')
    return padded(saved[0], saved[3] == 3, torch.tensor(readout['full_grid'], dtype=torch.bool), readout['lengths']), run.sha(path)


def batcher(table):
    active = [d for d in table if len(d[0])]
    if not active:
        raise ValueError('No donor slates')

    def batch(rng):
        counts = np.bincount(rng.integers(len(active), size=BATCH), minlength=len(active))
        pick = [rng.integers(len(d[0]), size=n) for d, n in zip(active, counts, strict=True)]
        return tuple(torch.cat([d[k][ix] for d, ix in zip(active, pick, strict=True)]) for k in range(3))

    return batch


def slate_rows(args):
    site, seed, motion, d, bank, menu, width, nets, _ = args
    x, legal, rows, path, _ = menu
    move = parents.base.model.movement_prefix(bank[0], path, rows)
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(parents.base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Frozen edit count, cap or roster changed')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        aligned, feat = None, None
        if len(pairs):
            z = old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
            feat = exp.design(torch.from_numpy(z), width, 'aligned')
            with torch.inference_mode():
                aligned = old.model.rank(nets['aligned'](feat))
        yield ai, r, a, pairs, aligned, feat


def choose(args, nets):
    picks = []
    for _, _, _, _, aligned, feat in slate_rows(args):
        row = None
        if aligned is not None:
            with torch.inference_mode():
                row = dict(aligned=int(aligned), **{k: int(nets[k](feat).argmax()) for k in ARMS})
        picks.append(row)
    return picks


def evaluate(args, picks):
    d, bank, menu, cut = args[3], args[4], args[5], args[8]
    rows, tally = {k: np.zeros(3) for k in (*CONTROLS, *ARMS)}, Counter()
    for slate, p in zip(slate_rows(args), picks, strict=True):
        ai, r, a, pairs, aligned, _ = slate
        if (p is None) != (aligned is None) or (p is not None and p['aligned'] != aligned):
            raise ValueError('The registered held mask did not reproduce')
        tally.update(views=1, edits=int(a.sum()), switched=int(p is not None))
        cv, y, v = bank[0][ai], d['y'][ai], d['valid'][ai]

        def score(c):
            pred, s = old.measure(d, ai, bank, menu[3], r, c, cut)
            full = prior.certificate(cv, pred, y, v, cut[0], s)[0]
            return np.array((s['dual_positive'], full, full and min(s['vs_cv'], s['vs_ols4']) > prior.MATERIAL_PP))

        anchor = score(a)
        rows['incumbent'] += anchor
        if p is None:
            for k in ('aligned', 'uniform', *ARMS):
                rows[k] += anchor
        else:
            options = np.asarray([score(old.model.exchanged(a, *pair)) for pair in pairs])
            rows['uniform'] += options.mean(0)
            for k, j in p.items():
                rows[k] += options[j]
    return dict(counts=dict(tally), arms={k: dict(zip(FIELDS, z.tolist(), strict=True)) for k, z in rows.items()})


def summarize(groups, reference):
    cells = [r for g in groups.values() for r in g]
    counts = sum((Counter(r['counts']) for r in cells), Counter())
    arms = {k: dict(sum((Counter(r['arms'][k]) for r in cells), Counter())) for k in (*CONTROLS, *ARMS)}
    for key, value in counts.items():
        if value != reference['counts'][key]:
            raise ValueError('The registered evaluation roster or budget changed')
    for k in CONTROLS:
        if any(abs(arms[k][f]-reference['arms'][k][f]) > 1e-10 for f in FIELDS):
            raise ValueError('The frozen control did not reproduce')
    draws = np.random.default_rng(817).integers(7, size=(3000, 7))
    comparisons = {}
    for a, b in (('list_original', 'point_original'), ('list_full_grid', 'list_original'),
                 ('list_full_grid', 'uniform'), ('list_full_grid', 'incumbent'), ('list_full_grid', 'aligned')):
        delta = np.asarray([np.mean([100*(r['arms'][a]['full_grid']-r['arms'][b]['full_grid'])/r['counts']['views']
                                    for r in g]) for g in groups.values()])
        comparisons[f'{a}_minus_{b}'] = dict(point_pp=float(delta.mean()),
            ci95_pp=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
    return dict(counts=dict(counts), arms=arms, full_grid_contrasts=comparisons, bootstrap_draws=len(draws))


def execute(phase, beat):
    context_reg, _, _, _, _, _, heads, _, _, _, assets = exp.register()
    p = parents.OUT/'registration.json'
    risk_reg = run.sha(p)
    upstream = parents.OUT/'readout.json'
    saved = run.read(upstream)
    if saved['registration'] != risk_reg or saved['external_outcome_access'] or not saved['original_2pct']:
        raise ValueError('Require the frozen donor lineage')
    regpath = OUT/'registration.json'
    sources = [Path(__file__), Path(ranker.__file__), Path(prior.__file__), OUT/'protocol.md',
               p, upstream, prior.OUT/'readout.json', exp.OUT/'head_freeze.json']
    run.once(regpath, run.stable(dict(inputs=[run.artifact(f) for f in sources], threads=4, workers=0,
        arms=ARMS, seeds=study.prior.SEEDS, steps=old.STEPS, batch_slates=BATCH, original_2pct=True,
        no_held_tuning=True, new_external_outcome_access=False)))
    reg = run.sha(regpath)
    path = OUT/'head_freeze.json'
    groups, cells, updates = {}, {}, 0
    if phase == 'fit':
        sites = {s: parents.parent.prior.prior.old.load_site(s) for s in parents.parent.source.SITES}
        for s, d in sites.items():
            with np.load(parents.parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz', allow_pickle=False) as z:
                d.update(recordings=z['recording'], starts=z['start'])
        banks = {s: parents.parent.prior.analytic(d) for s, d in sites.items()}
        maps = {s: exp.road.raster_for(s, assets[s])[0] for s in sites}
    else:
        receipt = run.read(path)
        if receipt['registration'] != reg:
            raise ValueError('Held choices are not frozen for this exact design')
    for args, ref in matched.context_cells():
        held, seed, width = args[0], args[1], args[6]
        key = f'{held}_{seed}'
        file = PRIVATE/f'{key}_choices.json'
        if phase == 'fit':
            donors = [s for s in sites if s != held]
            pairs = {s: donor(held, s, seed, sites, banks, maps, heads, reg, risk_reg, context_reg) for s in donors}
            table, ids = [a for a, _ in pairs.values()], {s: sha for s, (_, sha) in pairs.items()}
            batch, nets, artifacts = batcher(table), {}, {}
            for k in ARMS:
                identity = dict(registration=reg, held=held, train=donors, seed=seed, arm=k, data=ids)

                def loss(net, x, y, v):
                    z = exp.design(x.flatten(0, 1), width, 'aligned').reshape_as(x)
                    return ranker.loss(net(z), y[..., int(k == 'list_full_grid')], v, listwise=k != 'point_original')

                nets[k], artifacts[k], n = study.fit(identity, str(PRIVATE/f'{key}_{k}'), seed,
                    lambda: ranker.ListRanker(4*args[5][0].shape[1]+9), old.STEPS, batch, loss, False, beat)
                updates += n
            picks = choose(args, nets)
            if sum(p is not None for p in picks) != ref['transitions']['aligned']['exchanged']:
                raise ValueError('Fixed switch mask changed')
            run.once(file, run.stable(dict(registration=reg, picks=picks)))
            cells[key] = dict(choices=run.artifact(file), models=artifacts, donor_slates=sum(len(a[0]) for a in table),
                              donor_options=sum(int(a[2].sum()) for a in table),
                              donor_original=sum(int(a[1][..., 0].sum()) for a in table),
                              donor_full_grid=sum(int(a[1][..., 1].sum()) for a in table))
        else:
            cell = receipt['cells'][key]
            run.checked(cell['choices'])
            for k in ARMS:
                meta = cell['models'][k]
                run.checked(meta['artifact'])
                if meta['identity']['held'] != held or held in meta['identity']['train']:
                    raise ValueError('A frozen model used held data')
            row = evaluate(args, run.read(file)['picks'])
            if row['arms']['aligned']['original'] != ref['scores']['aligned']['dual_positive']:
                raise ValueError('The archived model result did not replay')
            groups.setdefault(held, []).append(dict(seed=seed, **row))
        beat(phase='cell', held=held, seed=seed, updates=updates)
    if phase == 'fit':
        if len(cells) != 21:
            raise ValueError('Incomplete checkpoint freeze')
        run.once(path, run.stable(dict(registration=reg, cells=cells, configured_updates=21*len(ARMS)*old.STEPS,
            outcome_scoring_this_phase=False)))
    else:
        if len(groups) != 7 or any({r['seed'] for r in g} != {17, 29, 43} for g in groups.values()):
            raise ValueError('Incomplete held roster')
        summary = summarize(groups, run.read(prior.OUT/'readout.json')['summary'])
        run.once(OUT/'readout.json', run.stable(dict(registration=reg, freeze_sha256=run.sha(path), summary=summary,
            by_location=groups, result_source='fresh_matched_opendd_development_training', no_held_tuning=True,
            original_2pct=True, new_external_outcome_access=False, independent_confirmation=False, deployment_changed=False)))
        print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('fit', 'score'))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    (PRIVATE/'donors').mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**event):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if event['phase'] == 'cell':
                print(row, flush=True)

        execute(phase, beat)


if __name__ == '__main__':
    main()
