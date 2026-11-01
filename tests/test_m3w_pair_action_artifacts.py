import json
from pathlib import Path
import pytest

from scripts import report_m3w_pair_action_allocation as report
from scripts import run_m3w_pair_action_allocation as run


def test_report_rejects_incomplete_experiment(tmp_path,monkeypatch):
    reg=tmp_path/'registration.json'; reg.write_text(json.dumps(dict(bindings={})))
    monkeypatch.setattr(run,'REG',reg); monkeypatch.setattr(run,'PUBLIC',tmp_path)
    with pytest.raises(FileNotFoundError): report.build()
    assert not (tmp_path/'readout.md').exists()


def test_signed_intervals_and_missing_not_zero():
    assert report.fmt(dict(point=None,CI95=None))=='not_estimable'
    assert report.fmt(dict(point=-1.,CI95=[-2.,.5]))=='-1.000000 [-2.000000, +0.500000]'


def test_frozen_registration_and_original_population():
    reg=json.loads(run.REG.read_text())
    for rel,digest in reg['bindings'].items(): assert run.sha(run.ROOT/rel)==digest
    cfg=json.loads(run.CONFIG.read_text())
    assert cfg['pair_weight']==1. and cfg['past_nearest_neighbors']==4
    assert cfg['risk_budget']==.02 and cfg['new_training_updates']==0
    assert cfg['seeds']==[17,29,43]
    assert reg['role']=='exposed_SOURCE_fitting_resubstitution'
    assert not reg['inference_labels_used'] and not reg['independent_roles_opened']


def test_current_final_artifacts_if_generated():
    if not (run.PUBLIC/'evidence.json').exists():
        pytest.skip('Final complete-replay artifacts not generated;not a scientific pass')
    evidence=report.build(); s=evidence['summary']
    assert s['groups']==72 and s['unique_recordings']==74 and s['unique_target_rows']==149510
    assert not s['independent_confirmation'] and not s['new_training_updates'] and not s['deployment_changed']
    assert evidence['replay']['all72groups_exact'] and evidence['replay']['all72_old_controls_exact']
    for name in ('readout.md','failure_analysis.md','model_data_card.md','gates.md','reproduction_zh.md',
            'paper_results_component.md','locality_effects.csv','next_steps.md','figures/policy_readout.svg'):
        assert (run.PUBLIC/name).stat().st_size>100
