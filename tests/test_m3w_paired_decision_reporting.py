import copy
import pytest
from scripts import report_m3w_moving_paired_decision as report


def test_roster_requires_all_unique_complete_real_fits():
    jobs = [dict(key=str(j)) for j in range(3)]
    rows = [dict(job=j['key'], arm=a, step=2000, parameters=64260,
        both_neural_encoders_updated=True) for j in jobs for a in report.api.ARMS]
    report.verify_roster(rows, jobs)
    for bad in (rows[:-1], rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError):
            report.verify_roster(bad, jobs)
    bad = copy.deepcopy(rows); bad[0]['step'] = 1999
    with pytest.raises(ValueError):
        report.verify_roster(bad, jobs)


def test_scientific_figures_deterministic_and_svg_valid(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    monkeypatch.setattr(report, 'PUBLIC', tmp_path)
    monkeypatch.setattr(report.run, 'PRIVATE', tmp_path)
    rows = [dict(job='unit_critic'+str(j), arm=a, trace=[dict(step=1, risk_MSE=1), dict(step=2, risk_MSE=.5)],
        joint_attribution=dict(total=dict(false_safe_cost=.1, false_reject_cost=.2, total_joint_cost=.3)))
        for j in range(3) for a in report.api.ARMS]
    report.figure(rows)
    p = tmp_path/'figures/paired_fitting.svg'; saved = p.read_bytes()
    assert ET.fromstring(saved).tag.endswith('svg')
    report.figure(rows)
    assert p.read_bytes() == saved and (tmp_path/'paired_fitting.png').stat().st_size > 1000
