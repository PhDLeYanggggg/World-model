import sys
import pytest
from scripts import resume_m3w_harm_inference_local as wrapper


def test_inference_wrapper_cannot_train(monkeypatch):
    monkeypatch.setattr(sys,'argv',['inference','train'])
    with pytest.raises(ValueError,match='Inference only'):wrapper.main()


def test_only_inference_reserve_is_changed(monkeypatch):
    calls=[]
    original=dict(disk_reserve_bytes=10200547328,private_artifact_cap_bytes=104857600,
        positive_harm_budget=.02,easy_net_degradation_budget=.02,hard_runtime_limit_seconds=43200)
    def runtime(config,phase):calls.append(config);return 0,lambda:None,lambda **_:None
    monkeypatch.setattr(wrapper.run.source,'runtime',runtime)
    monkeypatch.setattr(wrapper.run,'main',lambda:wrapper.run.source.runtime(original,'verify'))
    monkeypatch.setattr(sys,'argv',['inference','verify'])
    wrapper.main()
    assert calls==[dict(original,disk_reserve_bytes=5368709120)]
    assert original['positive_harm_budget']==original['easy_net_degradation_budget']==.02
