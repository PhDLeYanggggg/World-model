"""Re-score frozen OpenDD comparisons and attribute their observed/missing differences."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_geometry as geo
from src.evaluation import m3w_opendd_pair_disagreement as diagnostic

learn, pair, base, old, np = geo.learn, geo.pair, geo.base, geo.old, geo.np
sha, write = base.prior.sha, base.prior.write_json
OUT = geo.OUT.parent / 'opendd_pair_disagreement_v1'


def summarize(rows, draws_seed):
    sites = base.SITES
    draws = np.random.default_rng(draws_seed).integers(len(sites), size=(3000, len(sites)))

    def interval(get):
        x = np.array([np.mean([get(r) for r in rows if r['site'] == s]) for s in sites])
        return dict(point=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    by_site=dict(zip(sites, x.tolist(), strict=True)))

    first = rows[0]
    counts = {k: sum(r['choice'][k] for r in rows) for k in first['choice']}
    categories = {k: dict(recordings=sum(r['categories'][k]['recordings'] for r in rows),
                          known_pp=interval(lambda r: r['categories'][k]['known_pp']),
                          missing_debit_pp=interval(lambda r: r['categories'][k]['missing_debit_pp']))
                  for k in first['categories']}
    transitions = {a: {b: sum(r['transitions'][a][b] for r in rows) for b in v}
                   for a, v in first['transitions'].items()}
    return dict(choices=counts, support=categories, control_to_treatment_gates=transitions,
                record_views=sum(counts.values()), site_seed_directions=len(rows),
                paired={k: interval(lambda r: r[k])
                        for k in ('known_pp', 'missing_debit_pp', 'lower_pp', 'upper_pp')})


def audit(study, key, data, comparisons, draws_seed):
    p = study.OUT / 'freeze.json'
    ref = study.OUT / 'readout.json'
    freeze, prior = old.read(p), old.read(ref)
    if (freeze['registration'] != key or prior['freeze'] != sha(p)
            or not freeze['held_choices_frozen'] or prior['external_outcomes_read']
            or not prior['original_2pct_unchanged'] or len(freeze['cells']) != 63):
        raise ValueError('Expected the existing registered OpenDD development readout')
    old_views = {(v['site'], v['seed'], v['calibration_site']): v for v in prior['views']}
    out = {f'{a}_minus_{b}': [] for a, b in comparisons}
    directions = set()
    recordings = set()
    for name, c in freeze['cells'].items():
        for site, packet in c['packets'].items():
            cal = c['rules'][site]['calibration_site']
            direction = (site, c['seed'], cal)
            path = study.LOCAL / packet['file']
            if (sha(path) != packet['sha256'] or direction in directions
                    or set(c['fit_sites']) != set(base.SITES)-{site, cal} or site == cal):
                raise ValueError('Frozen choices or their fit/held/calibration roles changed')
            directions.add(direction)
            d = data[site]
            with np.load(path, allow_pickle=False) as z:
                use_arms = {a for ab in comparisons for a in ab}
                preds = {a: pair.prediction(z, a) for a in use_arms}
                gates = {a: pair.evaluate(preds[a], z, d, c['cut']) for a in use_arms}
                cv = z['cv']
                names = z['records']
                recordings.update(names.tolist())
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
                for a in use_arms:
                    rows = z[f'{a}_rows']
                    bounds = z['bounds']
                    if (np.any(rows < bounds[:, :1]) or np.any(rows >= bounds[:, 1:])
                            or not (d['recording'][rows] == names[:, None]).all()):
                        raise ValueError('An edited actor is outside the registered query')
                    for field, suffix in (('dual', 'dual'), ('risk', 'risk'), ('missing', 'missing')):
                        if int(gates[a][field].sum()) != old_views[direction]['arms'][a][f'matched_{suffix}']:
                            raise ValueError('Original risk counts do not replay')
                for a, b in comparisons:
                    r = diagnostic.decompose(preds[a], preds[b], d['valid'], d['future'], d['weight'], mass,
                                             z[f'{a}_rows'], z[f'{b}_rows'])
                    old_score = old_views[direction]['contrasts'][b]
                    np.testing.assert_allclose([r['known_pp'], r['lower_pp']],
                        [old_score['ade_available_pp'], old_score['ade_lower_pp']], atol=1e-10, rtol=0.)
                    out[f'{a}_minus_{b}'].append(dict(site=site, **r, transitions=diagnostic.gate_transitions(gates[a], gates[b])))
        print(study.OUT.name, name, len(directions), flush=True)
    if directions != set(old_views) or len(directions) != 126 or len(recordings) != 493:
        raise ValueError('Expected all repeated views of the original 493 recordings')
    return dict(freeze_sha256=sha(p), original_readout_sha256=sha(ref), unique_recordings=len(recordings),
                comparisons={k: summarize(v, draws_seed) for k, v in out.items()})


def run():
    learned_key, source, *_ = learn.register()
    geometry_key, geometry_source, *_ = geo.register()
    if source != geometry_source:
        raise ValueError('Expected the same registered vehicle cohorts')
    data = base.parent.load(source)
    result = dict(
        learned=audit(learn, learned_key, data, [('joint', 'additive'), ('joint', 'same_family')], 2917),
        geometry=audit(geo, geometry_key, data, [('geometry', 'motion')], 1739),
        result_source='fresh_run', prediction_packets='cached_verified',
        implementation_sha256=sha(Path(__file__)), diagnostic_sha256=sha(Path(diagnostic.__file__)),
        original_2pct_unchanged=True, support_slices_use_outcomes_for_attribution_only=True,
        external_outcomes_read=False, model_selection_changed=False, new_training=False,
        independent_confirmation=False, deployment_changed=False)
    OUT.mkdir(exist_ok=True)
    write(OUT / 'diagnostic.json', result, immutable=True)


if __name__ == '__main__':
    geo.torch.set_num_threads(4)
    run()
