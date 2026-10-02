"""Read-only aggregate checks; no Torch, fitting, data-role access or refitting."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_recording_stability_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return sum(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        assert len(a) == len(b)
        return sum(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)
    else:
        assert a == b, (a, b)
    return 1


def check_groups(groups, summary, cfg):
    checks = 0
    additive = ['selected_count', 'selected_unknown', 'selected_known_benefit_mass',
        'selected_known_harm_mass', 'selected_known_reference_mass', 'selected_known_easy_reference_mass',
        'selected_known_easy_harm_mass', 'selected_unknown_envelope_mass', 'selected_net_gain_lower_mass']
    for g in groups:
        for arm, r in g['arms'].items():
            for key in additive:
                checks += close(r['parent'][key], r['consensus'][key]+r['removed'][key])
            checks += close(r['removed']['selected_count'],
                            r['removed_for_unestimable_deletion']+r['removed_despite_estimable_deletions'])
            fs = r['folds']
            assert len({f['held_recording'] for f in fs}) == g['source_recordings'] == len(fs)
            checks += close(r['parent']['selected_count'], sum(f['base_selected'] for f in fs))
            checks += close(r['consensus']['selected_count'], sum(f['retained'] for f in fs))
            for f in fs:
                assert f['retained'] <= f['base_selected']
                if not f['estimable']:
                    assert f['retained'] == 0
            assert r['matched']['ratio_or_safety_guarantee'] is False
    for arm, shown in summary['arms'].items():
        rr = [g['arms'][arm] for g in groups]
        for kind in ('parent', 'consensus', 'removed'):
            v = [r[kind] for r in rr]
            risks = [r['easy_selected_risk_upper'] for r in v]
            expected = dict(complete_support_pass=sum(r['finite_completion_supported'] for r in v),
                selected_occurrences=sum(r['selected_count'] for r in v),
                unknown_occurrences=sum(r['selected_unknown'] for r in v),
                defined=sum(x is not None for x in risks), undefined=sum(x is None for x in risks),
                upper_violations=sum(x is not None and x > .02+1e-12 for x in risks),
                worst_completion_upper=max((x for x in risks if x is not None), default=None))
            checks += close(expected, shown[kind])
        for field, comparator in [('utility_change_vs_parent', 'parent'), ('utility_change_vs_matched_expectation', 'matched')]:
            pairs = []
            for g in groups:
                r = g['arms'][arm]
                ref = r['parent']['selected_net_gain_lower_mass'] if comparator == 'parent' else r['matched']['conservative_utility_expectation']
                mass = g['full_known_reference']
                pairs.append((g['source'], 100*(r['consensus']['selected_net_gain_lower_mass']-ref)/mass if mass > 0 else None))
            sites = sorted({s for s, v in pairs if v is not None})
            means = np.array([math.fsum(v for s, v in pairs if s == site and v is not None)/sum(s == site and v is not None for s, v in pairs) for site in sites])
            n = sum(v is not None for _, v in pairs)
            expected = dict(defined_views=n, undefined_views=len(groups)-n, localities=len(sites), mean=None, CI95=None)
            if sites:
                rng = np.random.default_rng(cfg['bootstrap_seed'])
                boot = means[rng.integers(0, len(sites), (cfg['bootstrap_resamples'], len(sites)))].mean(1)
                expected.update(mean=float(means.mean()), CI95=np.quantile(boot, [.025, .975]).tolist(),
                    strict_all_views_mean=float(means.mean()) if n == len(groups) else None,
                    nominal_exposed_development_only=True)
            checks += close(expected, shown[field])
    return checks


def main():
    reg = json.loads((PUBLIC/'registration.json').read_text())
    for name, sha in reg['bindings'].items():
        assert digest(ROOT/name) == sha
    complete = json.loads((PUBLIC/'complete.json').read_text())
    assert complete['registration_sha256'] == digest(PUBLIC/'registration.json')
    assert complete['exact_nested_replay'] is True and complete['new_neural_updates'] == 0
    assert len(complete['groups']) == 72
    groups = []
    prior = json.loads((PUBLIC.parent/'european_selected_set_calibration_v1/joint_oof_details.json').read_text())
    prior = {r['group']: r for r in prior}
    anchors = 0
    total = 0
    for r in complete['groups']:
        path = ROOT/r['path']
        assert path.parent == PUBLIC/'groups' and digest(path) == r['sha256']
        assert path.stat().st_size == r['bytes']
        total += r['bytes']
        g = json.loads(path.read_text()); groups.append(g)
        for arm, key in [('raw_pool', 'parent'), ('selected_set', 'selected_set')]:
            anchors += close(g['arms'][arm]['parent'], prior[path.stem][key])
    assert digest(PUBLIC/'summary.json') == complete['summary_sha256']
    summary = json.loads((PUBLIC/'summary.json').read_text())
    cfg = json.loads((ROOT/'configs/m3w_european_recording_stability_v1.json').read_text())
    checks = check_groups(groups, summary, cfg)
    receipt = dict(groups=len(groups), aggregate_checks=checks, prior_anchor_fields=anchors,
        group_bytes=total, summary_sha256=complete['summary_sha256'],
        complete_sha256=digest(PUBLIC/'complete.json'),
        verification_code_sha256=digest(__file__),
        independent_aggregate_reduction=True, independent_nested_fitting=False,
        numerical_arrays_loaded=False, new_neural_updates=0,
        tolerance=dict(relative=1e-10, absolute=1e-10))
    raw = json.dumps(receipt, indent=2, allow_nan=False)+'\n'
    target = PUBLIC/'verification.json'
    if target.exists():
        assert target.read_text() == raw
    else:
        with target.open('x') as f:
            f.write(raw)
    print(raw)


if __name__ == '__main__':
    main()
