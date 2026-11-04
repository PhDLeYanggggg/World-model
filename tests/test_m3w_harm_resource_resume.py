from pathlib import Path
import platform
import runpy
import sys

import pytest

from scripts import resume_m3w_harm_refinement_local as wrapper


def test_rosetta_is_rejected_before_training_imports(monkeypatch):
    monkeypatch.setattr(platform,'system',lambda:'Darwin')
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    with pytest.raises(RuntimeError,match='native arm64'):
        runpy.run_path(str(Path(wrapper.__file__)),run_name='__native_guard_test__')


def test_resource_wrapper_changes_only_disk_setting_and_forces_resume(monkeypatch):
    calls=[];beats=[]
    config=dict(disk_reserve_bytes=10200547328,positive_harm_budget=.02,
        easy_net_degradation_budget=.02,workers=0,torch_threads=4,
        private_artifact_cap_bytes=104857600,hard_runtime_limit_seconds=43200)
    def runtime(c,phase):
        calls.append((c,phase))
        return 0,lambda:None,lambda **kw:beats.append(kw)
    monkeypatch.setattr(wrapper.run.source,'runtime',runtime)
    monkeypatch.setattr(wrapper.run,'main',lambda:wrapper.run.source.runtime(config,'unit_only_no_training'))
    monkeypatch.setattr(sys,'argv',['resource_wrapper','train'])
    wrapper.main()
    assert calls==[(dict(config,disk_reserve_bytes=9663676416),'unit_only_no_training')]
    assert config['disk_reserve_bytes']==10200547328
    assert '--resume'in sys.argv
    assert beats[0]['risk_budget_changed']is False and beats[0]['training_budget_changed']is False


def test_wrapper_cannot_register_a_new_study(monkeypatch):
    monkeypatch.setattr(sys,'argv',['resource_wrapper','register'])
    with pytest.raises(ValueError,match='continuation'):wrapper.main()
