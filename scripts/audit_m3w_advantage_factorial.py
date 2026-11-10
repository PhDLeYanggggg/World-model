"""Check nested lineage, training-only scaling, and exact frozen choice replay."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_advantage_factorial as trial

run, torch, np, roles = trial.run, trial.torch, trial.np, trial.roles


def state(artifact, sites, omitted, seed, steps, cache):
    path = run.checked(artifact)
    key = artifact['sha256']
    if key not in cache:
        cache[key] = torch.load(path, weights_only=True, map_location='cpu')
    z = cache[key]
    roles.verify_role(z['identity'], sites, omitted)
    assert z['identity']['seed'] == seed and z['step'] == steps
    assert np.isfinite(z['trace']).all() and len(z['trace']) == steps//100
    return z


def producer_audit(reg, sites, cache):
    frozen = run.read(trial.OUT/'producer_freeze.json')
    assert frozen['registration'] == reg and len(frozen['cells']) == 105
    ranks, teachers, counts, tally = set(), set(), set(), Counter()
    for item in frozen['cells'].values():
        meta = run.read(run.checked(item))
        omitted, seed = meta['role']['omitted'], meta['role']['seed']
        assert len(omitted) == 3 and meta['role']['registration'] == reg
        z = state(meta['model']['artifact'], sites, omitted, seed, 500, cache)
        assert z['identity'] == meta['model']['identity']
        assert set(meta['dependencies']) == set(roles.fit_roster(sites, omitted))
        counts.add(meta['model']['artifact']['path'])
        for target, artifact in meta['dependencies'].items():
            teacher = run.read(run.checked(artifact))
            ident = teacher['identity']
            assert ident['omitted'] == omitted and ident['label_site'] == target
            excluded = sorted((*omitted, target))
            assert ident['cutoff_fit'] == roles.fit_roster(sites, excluded)
            rank = state(ident['rank'], sites, excluded, seed, 700, cache)
            assert rank['identity']['registration'] == reg
            run.checked(teacher['artifact'])
            ranks.add(ident['rank']['path'])
            teachers.add(teacher['artifact']['path'])
            tally.update(teacher['audit'])
    assert (len(ranks), len(counts), len(teachers)) == (105, 105, 420)
    return dict(new_rank_models=len(ranks), new_count_models=len(counts), teacher_caches=len(teachers),
                teacher_status=dict(tally))


def packet(artifact, reg, sites, excluded, donor, seed, cache):
    meta = run.read(run.checked(artifact))
    ident = meta['identity']
    omitted = sorted((*excluded, donor))
    assert ident['registration'] == reg and ident['excluded'] == sorted(excluded)
    assert ident['donor'] == donor and donor not in excluded and ident['seed'] == seed
    assert ident['cutoff_fit'] == roles.fit_roster(sites, omitted) and ident['label'] == 'full_grid'
    for k, steps in (('rank', 700), ('count', 500)):
        state(ident[k], sites, omitted, seed, steps, cache)
    with np.load(run.checked(meta['artifact']), allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('x', 'full', 'original', 'valid')}
    assert arrays['valid'][:, 0].all()
    assert not (arrays['full'] & ~arrays['original']).any()
    assert np.isfinite(arrays['x']).all()
    return tuple(torch.from_numpy(arrays[k]) for k in ('x', 'full', 'valid')), meta['audit']


def selector_cell(cell, reg, sites, excluded, seed, cache):
    metas = cell['models']
    assert set(metas) == set(trial.ARMS)
    fit = roles.fit_roster(sites, excluded)
    table, tallies = [], Counter()
    for donor, artifact in metas[trial.ARMS[0]]['identity']['data'].items():
        row, audit = packet(artifact, reg, sites, excluded, donor, seed, cache)
        table.append(row)
        tallies.update(audit)
    assert len(table) == len(fit)
    norm = trial.model.normalization(table)
    nets = {}
    for arm, meta in metas.items():
        ident = meta['identity']
        assert ident['registration'] == reg and ident['arm'] == arm and ident['target'] == 'full_grid'
        assert set(ident['data']) == set(fit) and ident['normalization'] == norm
        assert ident['data'] == metas[trial.ARMS[0]]['identity']['data']
        z = state(meta['artifact'], sites, excluded, seed, 400, cache)
        assert z['identity'] == ident
        nets[arm] = trial.study.load(meta, lambda: trial.model.Selector(arm, norm), 400)
    return nets, dict(tallies)


def audit():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _ = trial.register()
    sites = tuple(trial.prior.parents.parent.source.SITES)
    cache = {}
    producers = producer_audit(reg, sites, cache)
    inner = run.read(trial.OUT/'inner_freeze.json')
    assert inner['registration'] == reg and len(inner['cells']) == 63
    inner_tally = Counter()
    for item in inner['cells'].values():
        cell = run.read(run.checked(item))
        excluded, seed = cell['omitted'], cell['seed']
        nets, tally = selector_cell(cell, reg, sites, excluded, seed, cache)
        inner_tally.update(tally)
        for held, valid_site in (excluded, excluded[::-1]):
            v = cell['validation'][held]
            assert v['site'] == valid_site
            row, _ = packet(v['packet'], reg, sites, (held,), valid_site, seed, cache)
            for arm, net in nets.items():
                assert trial.validation(net, row) == v['arms'][arm]
    selection = trial.select(reg)
    outer = run.read(trial.OUT/'head_freeze.json')
    assert outer['registration'] == reg and len(outer['cells']) == 21
    run.checked(outer['selection'])
    outer_tally = Counter()
    for args, _ in trial.previous.matched.context_cells():
        held, seed = args[:2]
        meta = outer['cells'][f'{held}_{seed}']
        nets, tally = selector_cell(meta, reg, sites, (held,), seed, cache)
        outer_tally.update(tally)
        chosen = run.read(run.checked(meta['choices']))
        assert chosen['registration'] == reg and chosen['selected'] == selection[held]['selected']
        replay = []
        for _, _, _, _, aligned, feat in trial.previous.prior.slate_rows(args):
            row = None
            if aligned is not None:
                x = trial.prior.model.with_incumbent(feat[None])[0]
                valid = torch.ones(len(x), dtype=torch.bool)
                with torch.inference_mode():
                    row = dict(aligned=aligned+1, **{k: int(trial.model.choose(net(x), valid))
                                                   for k, net in nets.items()})
                arm = selection[held]['selected']
                row['inner_selected'] = 0 if arm == 'keep' else row[arm]
            replay.append(row)
        assert replay == chosen['picks']
    return dict(registration=reg, producers=producers, inner_models=252, outer_models=84,
        inner_validation_cells_replayed=126, outer_choice_cells_replayed=21,
        all_recorded_losses_finite=True, normalization_recomputed_from_training_only=True,
        role_exclusions_pass=True, inner_donor_repeated_counts=dict(inner_tally),
        outer_donor_repeated_counts=dict(outer_tally),
        selected_families={k: v['selected'] for k, v in selection.items()},
        new_training_updates=0, new_external_outcome_access=False, deployment_changed=False,
        code=run.artifact(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-score', action='store_true')
    args = parser.parse_args()
    result = audit()
    path = trial.OUT/('pre_score_audit.json' if args.pre_score else 'audit.json')
    if not args.pre_score:
        scored = run.read(trial.OUT/'readout.json')
        assert scored['registration'] == result['registration']
        assert trial.summary(scored['by_location']) == scored['summary']
        result['readout'] = run.artifact(trial.OUT/'readout.json')
        result['primary_supported'] = scored['summary']['primary_supported']
    run.once(path, run.stable(result))
    print(run.stable(result))
