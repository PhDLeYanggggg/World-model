"""Stress-test pair-relative evidence by masking complete OpenDD queries."""

import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_backbone_relative as backbone
from src.evaluation import m3w_relative_label_stress as stress

social, eq, prior, np, torch = backbone.social, backbone.eq, backbone.prior, backbone.np, backbone.torch
OUT = backbone.OUT.with_name('opendd_pair_label_stress_v1')
LOCAL = backbone.LOCAL.with_name(OUT.name)
CONTRASTS = {k: backbone.PAIRS[k] for k in ('eq_neighbors_minus_ego',
             'social_neighbors_minus_ego', 'completion_minus_cv')}
MASK_SEED = 711


def fit_free_cell(keys, site, seed, data, incumbent, freezes, reference, archived):
    a_key, b_key = keys
    tag = f'{site}_{seed}'
    d = data
    cv, y, mask, weights = (d[k] for k in ('cv', 'future', 'valid', 'weight'))
    p = dict(cv=cv, completion=eq.predict_incumbent(site, seed, d, incumbent))
    for a in social.model.ARMS:
        p[a] = social.predict(a_key, d, site, seed, a, freezes['social']['checkpoints'][f'{tag}_{a}'])
    for a in eq.ARMS:
        p[a] = eq.predict_eq(eq.LOCAL/f'{tag}_{a}.pt', freezes['eq']['checkpoints'][f'{tag}_{a}'],
                             eq.identity(b_key, site, seed, a), d)
    mass = float(weights @ np.where(mask, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    for a, pred in p.items():
        check = [float(100.*weights @ x.sum(1)/mass) for x in social.vehicle.paired(cv, pred, y, mask)]
        for raw in reference.values():
            old = raw['by_site_seed'][tag]['metrics']
            if a in old:
                np.testing.assert_allclose(check, old[a]['cv_gain_known_lower_pp'], atol=1e-9, rtol=0.)
    g, distance, support = stress.pair_step_terms(p, y, mask, d['query_counts'], CONTRASTS)
    if support['queries'] != archived['by_site_seed'][tag]['support']['complete_pair_queries']:
        raise ValueError('Complete pair-query roster disagrees with prior readout')
    return dict(support=support, contrasts=stress.evaluate(g, distance, MASK_SEED+social.SITES.index(site)))


def summarize(cells):
    sites, seeds = social.SITES, social.SEEDS
    for s in sites:
        if any(cells[f'{s}_{r}']['support'] != cells[f'{s}_{seeds[0]}']['support'] for r in seeds):
            raise ValueError('The originally complete label roster changed across seeds')
    support = {k: sum(cells[f'{s}_{seeds[0]}']['support'][k] for s in sites) for k in ('queries', 'pair_steps')}
    if support['queries'] != 3711:
        raise ValueError('Expected every originally complete query')
    draws = np.random.default_rng(3183).integers(len(sites), size=(3000, len(sites)))

    def stats(values):
        return dict(point=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist(),
                    locations=values.tolist())

    def aggregate(contrast, getter):
        return np.asarray([np.mean([getter(cells[f'{s}_{r}']['contrasts'][contrast]) for r in seeds], axis=0)
                           for s in sites])

    results = {}
    for k in CONTRASTS:
        full = stats(aggregate(k, lambda r: r['full']))
        rates = {}
        for hidden in stress.HIDDEN_STEPS:
            rate = str(hidden)
            fields = ('observed', 'bias', 'lower', 'upper')
            values = {a: {m: aggregate(k, lambda r: r['masked'][rate][a][m]) for m in fields}
                      for a in ('first', 'last', 'uniform')}
            arms = {a: {m: stats(v if a != 'uniform' else v.mean(1)) for m, v in d.items()}
                    for a, d in values.items()}
            uniform_bias = values['uniform']['bias'].mean(0)
            arms['uniform']['mask_bias_95'] = np.quantile(uniform_bias, [.025, .975]).tolist()
            rates[rate] = dict(arms=arms, retained_pair_steps=support['pair_steps']*(12-hidden)//12,
                              first_minus_last=stats(values['first']['observed']-values['last']['observed']))
        results[k] = dict(full=full, masked=rates)
    return support, results


def run():
    k_social, source, incumbent, geometry = social.register()
    k_eq, _, _, _ = eq.register()
    refs = {a: p/'freeze.json' for a, p in (('eq', eq.OUT), ('social', social.OUT))}
    scores = {a: p.with_name('readout.json') for a, p in refs.items()}
    f = {a: social.read(p) for a, p in refs.items()}
    r = {a: social.read(p) for a, p in scores.items()}
    for a, key in (('social', k_social), ('eq', k_eq)):
        if (f[a]['registration'] != key or f[a]['external_outcomes_opened']
                or r[a]['binding']['freeze'] != prior.sha(refs[a]) or r[a]['external_outcomes_opened']):
            raise ValueError('The frozen weights do not bind the prior score')
    src = backbone.OUT/'readout.json'
    previous = social.read(src)
    if (previous['registration'] != prior.sha(backbone.OUT/'registration.json')
            or previous['new_external_outcomes_opened'] or not previous['original_absolute_scores_reproduced']):
        raise ValueError('Require the verified exposed-development relative readout')
    p = OUT/'registration.json'
    files = (src, *refs.values(), *scores.values(), Path(__file__), Path(stress.__file__), OUT/'protocol.md')
    prior.write_json(p, dict(inputs={str(x.relative_to(prior.ROOT)): prior.sha(x) for x in files},
        mask_seed=MASK_SEED, hidden_steps=stress.HIDDEN_STEPS, uniform_masks=stress.RANDOM_REPLICATES,
        no_training=True, original_2pct=True, no_external_outcome_access=True), immutable=True)
    reg = prior.sha(p)
    cells = {}
    for s, d in eq.load(source, geometry).items():
        for seed in social.SEEDS:
            tag = f'{s}_{seed}'
            p = LOCAL/f'{tag}.json'
            if not p.exists():
                value = fit_free_cell((k_social, k_eq), s, seed, d, incumbent, f, r, previous)
                prior.write_json(p, dict(registration=reg, readout=value), immutable=True)
            obj = social.read(p)
            if obj['registration'] != reg:
                raise ValueError('Checkpoint belongs to a different masking protocol')
            cells[tag] = obj['readout']
            status = dict(pid=os.getpid(), location=s, seed=seed, completed=len(cells), total=21)
            prior.write_json(LOCAL/'heartbeat.json', status)
            print(status, flush=True)
    support, results = summarize(cells)
    prior.write_json(OUT/'readout.json', dict(registration=reg, locations=list(social.SITES),
        seeds=list(social.SEEDS), cells=len(cells), support=support, results=results, all_bounds_cover_known_gain=True,
        result_source='fresh_masking_sensitivity_on_scored_opendd_development', training=False,
        new_external_targets_opened=False, independent_confirmation=False, original_2pct=True,
        deployment_changed=False), immutable=True)
    print(support, {a: {'full': r['full']['point'], 'first_minus_last': r['masked']['3']['first_minus_last']}
                    for a, r in results.items()}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
