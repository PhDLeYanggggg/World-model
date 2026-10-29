import json
import pytest

from scripts import report_m3w_forecast_amplitude as report


def fixture(tmp_path, monkeypatch, supported):
    public, private = tmp_path/'public', tmp_path/'private'
    (public/'groups').mkdir(parents=True); private.mkdir()
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report.run, 'PUBLIC', public)
    monkeypatch.setattr(report.run, 'PRIVATE', private)
    cfg = dict(families=['forest', 'neural'], amplitudes=[1.])
    monkeypatch.setattr(report.run, 'load', lambda: (cfg, (None, dict(
        seeds=[17, 29, 43], bootstrap_seed=12, bootstrap_resamples=16))))
    names = (*report.run.api.ARMS, 'factorized')
    stats = dict(safety={n: dict(supported_record_views=444 if supported else 443,
                                required_record_views=444) for n in names},
                 floor_contrasts={n: dict(point=1., CI95=[.5, 1.5]) for n in names})
    summary = dict(families={f: {'1.0': stats} for f in cfg['families']})
    (public/'summary.json').write_text(json.dumps(summary))
    verification = dict(exact_replay=True, groups=144, amplitude_views=720,
                        summary_sha256=report.run.sha(public/'summary.json'))
    (public/'verification.json').write_text(json.dumps(verification))
    refs = []
    for family in cfg['families']:
        for i in range(72):
            records = {}
            for j in range(7 if i < 12 else 6):
                ok = supported or (i, j) != (0, 0)
                p = dict(selected_count=2, selected_unknown=0 if ok else 1,
                    easy_selected_risk_upper=.01 if ok else .03,
                    known_positive_risk=.005, known_easy_positive_risk=.005,
                    budget_status=dict(supported=ok, conditions={
                        'all_selected_risk_upper': ok, 'easy_selected_risk_upper': ok,
                        'easy_degradation_upper': True}))
                records[str(j)] = {n: p for bank in names for n in (bank, bank+'_same_set')}
            view = dict(context='context'+str(i//12), source='source'+str(i % 12),
                seed=[17, 29, 43][(i//12) % 3], family=family,
                numerical_convexity={str(j): dict(max_numerical_excess=0., known_rows=2, unknown_rows=1) for j in range(5)},
                amplitudes={'1.0': dict(recordings=records, comparisons={'physical_neural_minus_physical_only': {
                    mode: dict(known_pp=1., lower_pp=.5, upper_pp=1.5) for mode in ('full', 'matched')}})})
            path = public/'groups'/f'{family}_{i}.json'; path.write_text(json.dumps(view))
            refs.append(dict(path=str(path.relative_to(tmp_path)), sha256=report.run.sha(path)))
    complete = dict(groups=refs, pid=5, seconds=10., peak_RSS_bytes=1000)
    (public/'complete.json').write_text(json.dumps(complete))
    (private/'verification_runtime.json').write_text(json.dumps(dict(pid=6, seconds=10.)))
    return public


@pytest.mark.parametrize('supported', [False, True])
def test_even_positive_fitting_controls_cannot_promote_deployment(tmp_path, monkeypatch, supported):
    fixture(tmp_path, monkeypatch, supported)
    value = report.build(); gates = value['gates']
    assert gates['at_least_one_fitting_full_risk_and_completion_gain'] == supported
    assert not gates['deployment_promoted'] and not gates['submission_ready']
    assert not gates['held_scored'] and not gates['independent_confirmation']
    assert not gates['learned_costs_are_calibrated']
    assert value['numerical_convexity']['checks'] == 720
    count = value['descriptive_risk_failure_counts']['forest']['1.0']['physical_neural']
    assert count['required_record_views'] == 444
    assert count['unsupported_with_selected_unknown_no_known_violation'] == (0 if supported else 1)


def test_changed_group_or_missing_roster_is_rejected(tmp_path, monkeypatch):
    public = fixture(tmp_path, monkeypatch, False)
    path = public/'complete.json'; complete = json.loads(path.read_text())
    complete['groups'] = complete['groups'][:-1]
    path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match='All completed group'):
        report.build()
    public = fixture(tmp_path/'second', monkeypatch, False)
    (public/'groups'/'forest_0.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen group changed'):
        report.build()


def test_undefined_intervals_are_not_zero():
    assert report.number(None) == 'not_estimable'
    assert report.interval(dict(point=None, CI95=None)) == 'not_estimable'
