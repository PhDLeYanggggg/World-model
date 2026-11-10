"""Compare two published backbone adaptations on the same targets and masks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_stgcnn as study

match, prior, np, torch = study.match, study.prior, study.np, study.torch
OUT = study.OUT
PAIRS = dict(social_ego_minus_eqmotion_ego=('social_ego', 'comoving_ego'),
             social_peers_minus_eqmotion_peers=('social_peers', 'comoving_peers'))


def run():
    key, source, _, geometry = study.register()
    old_key, _, _, _ = match.register()
    reports = {a: p/'readout.json' for a, p in dict(social=study.OUT, eq=match.OUT).items()}
    freezes = {a: p.parent/'freeze.json' for a, p in reports.items()}
    old, new = (study.read(reports[a]) for a in ('eq', 'social'))
    f, g = (study.read(freezes[a]) for a in ('eq', 'social'))
    if (f['registration'] != old_key or g['registration'] != key
            or any(x['binding']['freeze'] != prior.sha(freezes[a]) or x['external_outcomes_opened']
                   for a, x in (('eq', old), ('social', new)))):
        raise ValueError('Both frozen readouts must bind their checkpoints')
    binding = {a: prior.sha(p) for a, p in reports.items()}
    binding['driver'] = prior.sha(Path(__file__))
    data = match.load(source, geometry)
    results = {}
    for s, d in data.items():
        cv, y, mask, w = (d[k] for k in ('cv', 'future', 'valid', 'weight'))
        mass = float(w @ np.where(mask, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
        for seed in study.SEEDS:
            tag = f'{s}_{seed}'
            path = study.LOCAL/f'{tag}_backbone.json'
            if not path.exists():
                row = {}
                for k, (a, b) in PAIRS.items():
                    q = study.predict(key, d, s, seed, a, g['checkpoints'][f'{tag}_{a}'])
                    r = match.predict_eq(match.LOCAL/f'{tag}_{b}.pt', f['checkpoints'][f'{tag}_{b}'],
                                         match.identity(old_key, s, seed, b), d)
                    values = []
                    for pred, ref, arm in ((q, new, a), (r, old, b)):
                        zz = [float(100.*w @ x.sum(1)/mass) for x in study.vehicle.paired(cv, pred, y, mask)]
                        expected = ref['by_site_seed'][tag]['metrics'][arm]['cv_gain_known_lower_pp']
                        np.testing.assert_allclose(zz, expected, atol=1e-9, rtol=0.)
                        values.append(zz)
                    v = [float(100.*w @ x.sum(1)/mass) for x in study.vehicle.paired(r, q, y, mask)]
                    row[k] = dict(observed=v[0], lower=v[1], cv_relative_controls=values)
                prior.write_json(path, dict(binding=binding, pairs=row), immutable=True)
            v = study.read(path)
            if v['binding'] != binding:
                raise ValueError('Comparison does not bind the same two readouts')
            results[tag] = v['pairs']
            study.beat('backbone_pair', site=s, seed=seed, completed=len(results))
    draws = np.random.default_rng(3183).integers(7, size=(3000, 7))
    def ci(a, metric):
        x = np.array([np.mean([results[f'{s}_{seed}'][a][metric] for seed in study.SEEDS]) for s in study.SITES])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(), sites=x.tolist())
    paired = {a: {m: ci(a, m) for m in ('observed', 'lower')} for a in PAIRS}
    prior.write_json(OUT/'backbone_pairs.json', dict(binding=binding, paired=paired, cells=len(results), by_site_seed=results,
        original_reference_metrics_reproduced=True, original_2pct=True, external_outcomes_opened=False,
        independent_confirmation=False, deployment_changed=False), immutable=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run()
