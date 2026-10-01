"""Standalone aggregate evidence verifier; requires Python and NumPy only."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


def equal_number(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, 'Undefined quantities cannot become zero'
    else:
        assert math.isfinite(actual) and math.isfinite(expected)
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)


def ratio(harm, reference):
    return harm / reference if reference > 0 else None


def verify(document):
    assert document['schema'] == 'selected_pool_source_aggregate_v1'
    assert document['risk_budget'] == .02
    assert document['scope'] == 'source_oof_joint_only'
    assert document['independent_confirmation'] is False
    rows = document['rows']
    assert len(rows) == len({r['view'] for r in rows}) == 72
    assert len({r['site'] for r in rows}) == 12
    assert {r['head_seed'] for r in rows} == {17, 29, 43}
    shifts, cases, point_checks = {}, [], 0
    for row in rows:
        pools = row['pools']
        assert set(pools) == {'raw', 'kept', 'removed'}
        for p in pools.values():
            for key in ('rows', 'known', 'unknown'):
                assert type(p[key]) is int and p[key] >= 0
            assert p['rows'] == p['known'] + p['unknown']
            for key in ('envelope', 'unknown_envelope'):
                assert math.isfinite(p[key]) and p[key] >= 0
            for key in ('truth', 'original', 'adjusted'):
                assert len(p[key]) == 5
                assert all(math.isfinite(v) and v >= 0 for v in p[key])
            if p['known'] == 0:
                assert p['envelope'] == 0
                assert all(v == 0 for key in ('truth', 'original', 'adjusted') for v in p[key])
            if p['unknown'] == 0:
                assert p['unknown_envelope'] == 0
        raw, kept, removed = [pools[k] for k in ('raw', 'kept', 'removed')]
        for key in ('rows', 'known', 'unknown'):
            assert raw[key] == kept[key] + removed[key]
        for key in ('envelope', 'unknown_envelope'):
            equal_number(raw[key], kept[key] + removed[key])
        for key in ('truth', 'original', 'adjusted'):
            for j in range(5):
                equal_number(raw[key][j], kept[key][j] + removed[key][j])
        risks = {}
        for name, p in pools.items():
            y, q = p['truth'], p['adjusted']
            risks[name] = dict(observed=ratio(y[4], y[3]), predicted=ratio(q[4], q[3]),
                bias=ratio(y[4] - q[4] + .02 * (q[3] - y[3]), p['envelope']))
        r, k = risks['raw'], risks['kept']
        delta = None if r['observed'] is None or k['observed'] is None else k['observed'] - r['observed']
        identity = ratio(kept['truth'][4] * removed['truth'][3] - removed['truth'][4] * kept['truth'][3],
                         kept['truth'][3] * raw['truth'][3])
        equal_number(delta, identity)
        shift = None if r['bias'] is None or k['bias'] is None else k['bias'] - r['bias']
        new = bool(delta is not None and r['observed'] <= .02 < k['observed'])
        point = dict(raw_risk=r['observed'], kept_risk=k['observed'], predicted_kept_risk=k['predicted'],
            signed_bias_shift=shift, harm_retained=ratio(kept['truth'][4], raw['truth'][4]),
            reference_retained=ratio(kept['truth'][3], raw['truth'][3]),
            benefit_retained=ratio(kept['truth'][0], raw['truth'][0]))
        assert set(row['expected']) == set(point) | {'new_known_violation', 'kept_known', 'kept_unknown'}
        for name, value in point.items():
            equal_number(value, row['expected'][name]); point_checks += 1
        assert new is row['expected']['new_known_violation']
        assert kept['known'] == row['expected']['kept_known']
        assert kept['unknown'] == row['expected']['kept_unknown']
        shifts.setdefault(row['site'], []).append(shift)
        if new:
            cases.append(dict(view=row['view'], site=row['site'], head_seed=row['head_seed'],
                source_screen=row['source_screen']))
    defined = [(site, v) for site, values in shifts.items() for v in values if v is not None]
    by = {site: float(np.mean([v for s, v in defined if s == site])) for site in sorted({s for s, _ in defined})}
    values = np.array(list(by.values()))
    cfg = document['bootstrap']
    assert cfg == {'draws': 3000, 'seed': 20261001, 'unit': 'locality'}
    rng = np.random.default_rng(cfg['seed'])
    samples = values[rng.integers(0, len(values), (cfg['draws'], len(values)))].mean(1)
    ci = np.quantile(samples, [.025, .975]).tolist()
    expected = document['expected_interval']['defined_only_descriptive']
    assert set(by) == set(expected['by_locality'])
    for site, value in by.items(): equal_number(value, expected['by_locality'][site])
    equal_number(float(values.mean()), expected['mean'])
    for a, b in zip(ci, expected['CI95']): equal_number(a, b)
    missing_sites = [s for s, v in shifts.items() if any(x is None for x in v)]
    assert document['expected_interval']['all_views']['mean'] is None
    assert set(missing_sites) == set(document['expected_interval']['all_views']['undefined_localities'])
    assert len(defined) == document['expected_interval']['defined_views']
    assert len(rows) - len(defined) == document['expected_interval']['undefined_views']
    assert len(cases) == 2 and len({r['site'] for r in cases}) == 1
    assert all(not r['source_screen'] for r in cases)
    return dict(status='aggregate_arithmetic_and_locality_bootstrap_reproduced', views=len(rows),
        defined=len(defined), undefined=len(rows)-len(defined), contributing_localities=len(by),
        all_view_mean=None, defined_only_mean=float(values.mean()), CI95=ci,
        counterexamples=cases, point_value_checks=point_checks, bootstrap_draws=cfg['draws'],
        raw_data_or_checkpoint_replay=False, independent_confirmation=False,
        torch_imported='torch' in sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdin', action='store_true')
    parser.add_argument('evidence', nargs='?', default='evidence.json')
    args = parser.parse_args()
    data = json.load(sys.stdin) if args.stdin else json.loads(Path(args.evidence).read_text())
    print(json.dumps(verify(data), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
