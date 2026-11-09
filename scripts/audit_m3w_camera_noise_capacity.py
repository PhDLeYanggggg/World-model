"""Separate candidate accuracy and radius limits on the exposed pilot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_camera_noise_control as experiment

study, base, np, torch = experiment.study, experiment.base, experiment.np, experiment.torch
pilot = experiment.pilot
camera = pilot.camera


def analyze():
    key, _, streams = experiment.register()
    f = study.read(experiment.OUT/'freeze.json')
    if f['registration'] != key or not f['no_camera_outcomes_read']:
        raise ValueError('Require frozen donor forecasts')
    reg, cells, _ = study.register()
    nets = study.models(reg, cells)
    source = streams['wepdtof']
    h, q = source['history'], source['q']
    old = study.predict(h, q, nets)
    y, v, m = experiment.targets('wepdtof', source)
    if m.sum() != 57 or v.all(1).sum() != 57:
        raise ValueError('Development camera cohort changed')
    _, q = np.unique(q, return_inverse=True)
    cuts = study.read(study.OUT/'pilot_freeze.json')['cut_px']
    results = {}
    for seed in study.SEEDS:
        z = old[seed]
        cv, ols = z['cv'], z['ols4']
        prev = pilot.fixed.method.bank(cv, ols, z['spline'], q)
        x, scale, legal, _ = study.costs.features(h, cv, ols, prev)
        with torch.inference_mode():
            value = nets[seed][0](torch.from_numpy(x)).numpy()*scale[:, None, None]
        path = pilot.fixed.fixed(value, legal, q, h, cv, prev)['fixed_joint']
        counts, radius_sum = pilot.counts_of(path, cv, q, int(q.max())+1)
        eligible = legal.all(1)
        radius = np.linalg.norm(prev[:, 0]-cv, axis=-1).max(1)
        with np.load(study.checked(f['forecasts']['wepdtof'][str(seed)]), allow_pickle=False) as a:
            np.testing.assert_array_equal(a['cv'], cv)
            np.testing.assert_array_equal(a['ols4'], ols)
            np.testing.assert_array_equal(a['q'], source['q'])
            np.testing.assert_array_equal(a['ids'], source['ids'])
            analytic = np.stack([a['ols4'], a['median4']], axis=1)
            augmented = np.concatenate((analytic, np.stack([a['clean'], a['noise']], axis=1)), axis=1)
        seed_result = {}
        for name, raw in (('analytic', analytic), ('with_splines', augmented)):
            clipped, _ = camera.matched_menu(cv, raw, eligible, radius)
            unclipped = np.where(eligible[:, None, None, None], raw, cv[:, None])
            peak = np.linalg.norm(unclipped-cv[:, None], axis=-1).max((1, 2))
            limit = np.zeros(len(counts))
            np.maximum.at(limit, q, peak)
            limits = dict(clipped=(clipped, radius_sum), full=(unclipped, counts*limit))
            seed_result[name] = {label:pilot.finite(cv, ols, paths, y, v, q, cuts, counts, budget)
                                 for label, (paths, budget) in limits.items()}
        results[str(seed)] = dict(query_slots=int(counts.sum()), per_menu=seed_result)
        print(dict(seed=seed, cells={m:{k:v['solver']['status'] for k,v in r.items()}
                                     for m,r in seed_result.items()}), flush=True)
    base.immutable(experiment.OUT/'development_capacity.json', dict(
        frozen_predictions=study.artifact(experiment.OUT/'freeze.json'),
        camera_registration=key, code=study.artifact(Path(__file__)), seeds=results,
        original_2pct=True, source='wepdtof_convenience_store', labels_hindsight_only=True,
        matched_count_caps=True, unchanged_eligible_rows=True, external_test_reopened=False,
        independent_confirmation=False, deployment_changed=False))


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    analyze()
