"""Test an explicit unchanged allocation within the frozen OpenDD exchange slate."""

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
from scripts import run_m3w_opendd_listwise_exchange as prior
from src.world_model import m3w_incumbent_slate as model

run, parents, old, np, torch = prior.run, prior.parents, prior.old, prior.np, prior.torch
exp, study, matched = prior.exp, prior.study, prior.matched
OUT = prior.OUT.with_name('opendd_incumbent_slate_v1')
PRIVATE = prior.PRIVATE.with_name(OUT.name)
MODELS = ('point', 'pair')
POLICIES = tuple(f'{k}_{mode}' for k in MODELS for mode in ('forced', 'optional'))
CONTROLS = ('incumbent', 'aligned', 'uniform_keep', 'mask_oracle')
ARMS = (*CONTROLS, *POLICIES, 'point_uniform_mask', 'pair_uniform_mask')
FIELDS = (*prior.FIELDS, 'risk_undefined', 'risk_violation')


def donor_table(held, seed, sites, banks, maps, heads, context_reg):
    risk_reg = run.sha(parents.OUT/'registration.json')
    slate_reg = run.sha(prior.OUT/'registration.json')
    table, hashes = [], {}
    for site in sites:
        if site == held:
            continue
        saved, source = parents.data_for(held, site, seed, sites, banks, maps, heads, risk_reg, context_reg)
        padded, label_sha = prior.donor(held, site, seed, sites, banks, maps, heads,
                                        slate_reg, risk_reg, context_reg)
        x, y, valid = padded
        lengths = valid.sum(1).tolist()
        anchor = model.incumbent_targets(saved[3] == 3, saved[1], lengths)
        if not torch.equal(y[..., 0][valid], saved[3] == 3):
            raise ValueError('The original donor labels changed')
        table.append((x, y[..., 0], valid, anchor))
        hashes[site] = dict(features_and_transition=source, slate_labels=label_sha)
    return table, hashes


def batcher(table):
    active = [d for d in table if len(d[0])]
    if not active:
        raise ValueError('No donor slates')
    def batch(rng):
        counts = np.bincount(rng.integers(len(active), size=prior.BATCH), minlength=len(active))
        picks = [rng.integers(len(d[0]), size=n) for d, n in zip(active, counts, strict=True)]
        return tuple(torch.cat([d[k][ix] for d, ix in zip(active, picks, strict=True)]) for k in range(4))
    return batch


def choose(args, nets):
    picks = []
    for _, _, _, _, aligned, feat in prior.slate_rows(args):
        row = None
        if aligned is not None:
            with torch.inference_mode():
                z = model.with_incumbent(feat[None])[0]
                row = dict(aligned=aligned+1)
                for k, net in nets.items():
                    row.update({f'{k}_{a}': i for a, i in model.choices(net(z)).items()})
        picks.append(row)
    return picks


def evaluate(args, picks):
    d, bank, menu, cut = args[3], args[4], args[5], args[8]
    counts, switches = Counter(), Counter({k: 0 for k in POLICIES})
    outcomes = {k: np.zeros(len(FIELDS)+2) for k in ARMS}
    for slate, chosen in zip(prior.slate_rows(args), picks, strict=True):
        ai, r, anchor, pairs, aligned, _ = slate
        if ((chosen is None) != (aligned is None)
                or chosen is not None and chosen['aligned'] != aligned+1):
            raise ValueError('The frozen eligibility mask changed')
        counts.update(views=1, edits=int(anchor.sum()), eligible=int(chosen is not None))
        cv, y, valid = bank[0][ai], d['y'][ai], d['valid'][ai]
        def score(plan):
            p, s = old.measure(d, ai, bank, menu[3], r, plan, cut)
            full = prior.prior.certificate(cv, p, y, valid, cut[0], s)[0]
            return np.asarray((s['dual_positive'], full,
                full and min(s['vs_cv'], s['vs_ols4']) > prior.prior.MATERIAL_PP,
                s['undefined'], not s['undefined'] and not s['risk_pass']), dtype=float)
        base = score(anchor)
        if chosen is None:
            options = base[None]
            selected = dict.fromkeys(ARMS, base)
        else:
            options = np.asarray([base, *(score(old.model.exchanged(anchor, *p)) for p in pairs)])
            if any(not 0 <= j < len(options) for j in chosen.values()):
                raise ValueError('Frozen choice is outside its legal slate')
            selected = dict(incumbent=base, aligned=options[chosen['aligned']],
                            uniform_keep=options.mean(0), mask_oracle=options[options[:, 1].argmax()])
            for k in POLICIES:
                selected[k] = options[chosen[k]]
                switches[k] += int(chosen[k] != 0)
            for k in MODELS:
                selected[f'{k}_uniform_mask'] = options[1:].mean(0) if chosen[f'{k}_optional'] else base
        for k, value in selected.items():
            rescued = value[1] if not base[1] else 0.
            destroyed = 1.-value[1] if base[1] else 0.
            outcomes[k] += np.r_[value, rescued, destroyed]
    return dict(counts=dict(counts), switches=dict(switches),
        arms={k: dict(zip((*FIELDS, 'rescued_full_grid', 'destroyed_full_grid'), z.tolist(), strict=True))
              for k, z in outcomes.items()})


def summarize(groups):
    cells = [r for g in groups.values() for r in g]
    counts = dict(sum((Counter(r['counts']) for r in cells), Counter()))
    switches = dict(sum((Counter(r['switches']) for r in cells), Counter()))
    arms = {k: dict(sum((Counter(r['arms'][k]) for r in cells), Counter())) for k in ARMS}
    ref = run.read(prior.OUT/'readout.json')['summary']
    if counts != dict(views=ref['counts']['views'], edits=ref['counts']['edits'], eligible=ref['counts']['switched']):
        raise ValueError('Registered roster or forecast spend changed')
    for k in ('incumbent', 'aligned'):
        if any(arms[k][f] != ref['arms'][k][f] for f in prior.FIELDS):
            raise ValueError('Archived control did not reproduce')
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    contrasts = {}
    pairs = [('pair_optional', k) for k in ('incumbent', 'pair_uniform_mask', 'point_optional', 'pair_forced')]
    pairs += [('point_optional', 'point_forced'), ('point_optional', 'incumbent')]
    for a, b in pairs:
        v = np.asarray([np.mean([100.*(r['arms'][a]['full_grid']-r['arms'][b]['full_grid'])/r['counts']['views']
                                 for r in group]) for group in groups.values()])
        contrasts[f'{a}_minus_{b}'] = dict(point_pp=float(v.mean()),
            ci95_pp=np.quantile(v[draws].mean(1), [.025, .975]).tolist(), by_location_pp=v.tolist())
    return dict(counts=counts, switches=switches, arms=arms, full_grid_contrasts=contrasts,
        locations=list(groups), bootstrap_draws=len(draws),
        primary_supported=all(contrasts[f'pair_optional_minus_{k}']['ci95_pp'][0] > 0.
                              for k in ('incumbent', 'pair_uniform_mask')))


def execute(phase, limit, beat):
    context_reg, _, _, _, _, _, heads, _, _, _, assets = exp.register()
    sources = [Path(__file__), Path(model.__file__), OUT/'protocol.md',
               prior.OUT/'registration.json', prior.OUT/'head_freeze.json', prior.OUT/'readout.json']
    path = OUT/'registration.json'
    specification = dict(inputs=[run.artifact(p) for p in sources], threads=4, workers=0,
        steps=old.STEPS, batch_slates=prior.BATCH, arms=ARMS, models=MODELS, seeds=study.prior.SEEDS,
        forecast_steps=[8, 12], original_2pct=True, no_held_tuning=True,
        new_external_outcome_access=False, independent_confirmation=False)
    amendment = OUT/'runtime_amendment.json'
    if path.exists() and run.read(path) != run.stable(specification):
        original = run.read(path)
        # Preserve valid checkpoints after the audited empty-donor handling repair.
        expected = ('656f97d60655bb750839e9a736e00618cb32554181f969ead3b30d2cd55a0bda',
                    '260e51c68b3f89434a08db3e6418cb95498f1cc790d8eba4283e8280b42229dc')
        if (tuple(v['sha256'] for v in original['inputs'][:2]) != expected
                or original['inputs'][2:] != specification['inputs'][2:]
                or {k: v for k, v in original.items() if k != 'inputs'} !=
                   run.stable({k: v for k, v in specification.items() if k != 'inputs'})):
            raise ValueError('Only the documented empty-donor runtime repair is allowed')
        if not amendment.exists() and (OUT/'head_freeze.json').exists():
            raise ValueError('Cannot amend runtime after choice freeze')
        run.once(amendment, run.stable(dict(original_registration=run.sha(path),
            replacement_code=specification['inputs'][:2],
            changes=['empty_donor_returns_empty_boolean_target', 'compare_normalized_registration_objects'],
            completed_cells_before_repair=4, outcomes_scored_before_repair=False,
            preserved_checkpoints=True, training_and_evaluation_rules_unchanged=True)))
    else:
        run.once(path, run.stable(specification))
    reg = run.sha(path)
    freeze = OUT/'head_freeze.json'
    result, cells, updates = {}, {}, 0
    if phase == 'fit':
        sites = {s: parents.parent.prior.prior.old.load_site(s) for s in parents.parent.source.SITES}
        for s, d in sites.items():
            with np.load(parents.parent.prior.prior.old.source.prior.PRIVATE/f'{s}_cohort.npz', allow_pickle=False) as z:
                d.update(recordings=z['recording'], starts=z['start'])
        banks = {s: parents.parent.prior.analytic(d) for s, d in sites.items()}
        maps = {s: exp.road.raster_for(s, assets[s])[0] for s in sites}
    else:
        saved = run.read(freeze)
        if (saved['registration'] != reg or saved['cells_count'] != 21
                or saved.get('runtime_amendment') != (run.sha(amendment) if amendment.exists() else None)):
            raise ValueError('Complete frozen choices required before scoring')
    for index, (args, ref) in enumerate(matched.context_cells()):
        if limit and index >= limit:
            break
        held, seed, width = args[0], args[1], args[6]
        key = f'{held}_{seed}'
        choices_path = PRIVATE/f'{key}_choices.json'
        if phase == 'fit':
            table, data = donor_table(held, seed, sites, banks, maps, heads, context_reg)
            batch, nets, artifacts = batcher(table), {}, {}
            for arm in MODELS:
                identity = dict(registration=reg, held=held, train=[s for s in sites if s != held],
                                seed=seed, arm=arm, data=data)
                def loss(net, x, y, valid, anchor):
                    masked = exp.design(x.flatten(0, 1), width, 'aligned').reshape_as(x)
                    return model.objective(net(model.with_incumbent(masked)), y, valid, anchor,
                                           pairwise=arm == 'pair')
                nets[arm], artifacts[arm], n = study.fit(identity, str(PRIVATE/f'{key}_{arm}'), seed,
                    lambda: model.ListRanker(table[0][0].shape[-1]+1), old.STEPS, batch, loss, False, beat)
                updates += n
            picks = choose(args, nets)
            run.once(choices_path, run.stable(dict(registration=reg, picks=picks)))
            cells[key] = dict(models=artifacts, choices=run.artifact(choices_path),
                donor_slates=sum(len(d[0]) for d in table), donor_options=sum(int(d[2].sum()) for d in table),
                incumbent_positives=sum(int(d[3].sum()) for d in table))
            beat(phase='cell', held=held, seed=seed, updates=updates,
                 train_loss={k: v['trace'] for k, v in artifacts.items()})
        else:
            meta = saved['cells'][key]
            run.checked(meta['choices'])
            for m in meta['models'].values():
                run.checked(m['artifact'])
                if m['identity']['held'] != held or held in m['identity']['train']:
                    raise ValueError('A model used held training labels')
            picks = run.read(choices_path)
            if picks['registration'] != reg:
                raise ValueError('Choices do not belong to this experiment')
            row = evaluate(args, picks['picks'])
            if row['counts']['eligible'] != ref['transitions']['aligned']['exchanged']:
                raise ValueError('Changed outer eligibility')
            result.setdefault(held, []).append(dict(seed=seed, **row))
            beat(phase='cell', held=held, seed=seed, updates=0)
    if phase == 'fit':
        if len(cells) == 21:
            run.once(freeze, run.stable(dict(registration=reg, cells=cells, cells_count=21,
                configured_updates=21*len(MODELS)*old.STEPS, outcome_scoring_this_phase=False,
                runtime_amendment=run.sha(amendment) if amendment.exists() else None)))
        beat(phase='complete' if len(cells) == 21 else 'pilot_complete', cells=len(cells), updates=updates)
    else:
        if len(result) != 7 or any({r['seed'] for r in g} != {17, 29, 43} for g in result.values()):
            raise ValueError('Incomplete location/seed matrix')
        summary = summarize(result)
        run.once(OUT/'readout.json', run.stable(dict(registration=reg, freeze_sha256=run.sha(freeze),
            summary=summary, by_location=result, result_source='fresh_exposed_development_training_and_scoring',
            independent_confirmation=False, new_external_outcome_access=False, new_release=False,
            original_2pct=True, no_held_tuning=True)))
        print(summary, flush=True)
        beat(phase='complete', updates=0, primary_supported=summary['primary_supported'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'score'))
    parser.add_argument('--limit-cells', type=int)
    args = parser.parse_args()
    if args.limit_cells is not None and (args.phase != 'fit' or not 1 <= args.limit_cells <= 21):
        raise ValueError('A pilot can limit training only, never scoring')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**event):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **event)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if event['phase'] != 'fit':
                print(row, flush=True)
        execute(args.phase, args.limit_cells, beat)


if __name__ == '__main__':
    main()
