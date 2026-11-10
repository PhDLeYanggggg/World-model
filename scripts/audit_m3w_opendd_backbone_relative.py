"""Replay frozen OpenDD backbones and measure their vehicle-pair trajectory error."""

import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_stgcnn as social
from src.evaluation import m3w_relative_forecast_readout as relative

eq, prior, np, torch = social.match, social.prior, social.np, social.torch
OUT = social.OUT.with_name('opendd_backbone_relative_v1')
LOCAL = social.LOCAL.with_name(OUT.name)
ARMS = ('cv', 'ols4', 'completion', 'comoving_ego', 'comoving_peers', 'social_ego', 'social_peers')
PAIRS = {f'{a}_minus_cv': (a, 'cv') for a in ARMS[1:]} | {
    'eq_neighbors_minus_ego': ('comoving_peers', 'comoving_ego'),
    'social_neighbors_minus_ego': ('social_peers', 'social_ego'),
    'completion_minus_eq_neighbors': ('completion', 'comoving_peers'),
    'completion_minus_social_neighbors': ('completion', 'social_peers'),
}


def cell(keys, site, seed, d, incumbent, freezes, scores):
    k_social, k_eq = keys
    tag = f'{site}_{seed}'
    cv, ols, y, valid, w = (d[k] for k in ('cv', 'ols4', 'future', 'valid', 'weight'))
    out = dict(cv=cv, ols4=ols, completion=eq.predict_incumbent(site, seed, d, incumbent))
    for a in social.model.ARMS:
        out[a] = social.predict(k_social, d, site, seed, a, freezes['social']['checkpoints'][f'{tag}_{a}'])
    for a in eq.ARMS:
        out[a] = eq.predict_eq(eq.LOCAL/f'{tag}_{a}.pt', freezes['eq']['checkpoints'][f'{tag}_{a}'],
                               eq.identity(k_eq, site, seed, a), d)
    if set(out) != set(ARMS):
        raise ValueError('Expected both frozen variants of each backbone')
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    for a, p in out.items():
        value = [float(100.*w @ x.sum(1)/mass) for x in social.vehicle.paired(cv, p, y, valid)]
        for k, v in scores.items():
            archived = v['by_site_seed'][tag]['metrics']
            if a in archived:
                np.testing.assert_allclose(value, archived[a]['cv_gain_known_lower_pp'], atol=1e-9, rtol=0.,
                                           err_msg=f'Original absolute comparison changed: {site}/{seed}/{k}/{a}')
    return relative.evaluate(out, y, valid, d['query_counts'], PAIRS)


def summarize(results):
    sites = social.SITES
    seeds = social.SEEDS
    counts = [results[f'{s}_{seeds[0]}']['support'] for s in sites]
    for s, support in zip(sites, counts, strict=True):
        if any(results[f'{s}_{seed}']['support'] != support for seed in seeds):
            raise ValueError('Pair and mask support must be identical across seeds')
    support = {k: sum(c[k] for c in counts) for k in counts[0]}
    if (support['rows'], support['queries']) != (141783, 25596):
        raise ValueError('Unexpected OpenDD cohort size')
    draws = np.random.default_rng(3183).integers(len(sites), size=(3000, len(sites)))

    def statistic(block, arm, name):
        x = np.asarray([np.mean([results[f'{s}_{seed}'][block][arm][name] for seed in seeds]) for s in sites])
        if not np.isfinite(x).all():
            raise ValueError('An original location does not support this readout')
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(), sites=x.tolist())

    absolute = {a: {k: statistic('absolute', a, k) for k in ('relative_ade', 'relative_fde', 'query_relative_ade_p95')}
                for a in ARMS}
    paired = {a: {k: statistic('contrasts', a, k) for k in ('observed', 'observed_contribution', 'lower', 'upper')}
              for a in PAIRS}
    return dict(support=support, absolute=absolute, paired=paired)


def run():
    k_social, source, incumbent, geometry = social.register()
    k_eq, _, _, _ = eq.register()
    paths = {k: root/'readout.json' for k, root in (('eq', eq.OUT), ('social', social.OUT))}
    refs = {k: p.with_name('freeze.json') for k, p in paths.items()}
    scores = {k: social.read(p) for k, p in paths.items()}
    freezes = {k: social.read(p) for k, p in refs.items()}
    for k, key in (('social', k_social), ('eq', k_eq)):
        if (freezes[k]['registration'] != key or freezes[k]['external_outcomes_opened']
                or scores[k]['binding']['freeze'] != prior.sha(refs[k]) or scores[k]['external_outcomes_opened']):
            raise ValueError('Frozen checkpoints and archived scores do not agree')
    files = [*paths.values(), *refs.values(), Path(__file__), Path(relative.__file__), OUT/'protocol.md']
    registration = OUT/'registration.json'
    prior.write_json(registration, dict(inputs={str(p.relative_to(prior.ROOT)): prior.sha(p) for p in files},
        original_2pct_unchanged=True, no_refit=True, no_new_external_outcomes=True), immutable=True)
    binding = prior.sha(registration)
    results = {}
    for site, d in eq.load(source, geometry).items():
        for seed in social.SEEDS:
            tag = f'{site}_{seed}'
            p = LOCAL/f'{tag}.json'
            if not p.exists():
                value = cell((k_social, k_eq), site, seed, d, incumbent, freezes, scores)
                prior.write_json(p, dict(registration=binding, readout=value), immutable=True)
            obj = social.read(p)
            if obj['registration'] != binding:
                raise ValueError('Cached relative comparison does not bind the exact run')
            results[tag] = obj['readout']
            status = dict(pid=os.getpid(), site=site, seed=seed, completed=len(results), total=21)
            prior.write_json(LOCAL/'heartbeat.json', status)
            print(status, flush=True)
    out = summarize(results)
    prior.write_json(OUT/'readout.json', dict(registration=binding, sites=list(social.SITES), seeds=list(social.SEEDS),
        cells=len(results), **out, by_site_seed=results, original_absolute_scores_reproduced=True,
        result_source='fresh_relative_metrics_on_previously_scored_opendd', training=False,
        new_external_outcomes_opened=False, independent_confirmation=False,
        original_2pct_unchanged=True, deployment_changed=False), immutable=True)
    print(out['support'], {a: out['paired'][a]['observed'] for a in PAIRS}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
