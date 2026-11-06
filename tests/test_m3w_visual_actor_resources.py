import pytest

from scripts.resume_m3w_moving_visual_actor_resources import runtime_settings


def test_resource_amendment_only_changes_disk_reserve():
    cfg = dict(disk_reserve_bytes=5*1024**3, artifact_cap_bytes=1024**3,
        steps_per_model=4000, original_risk_budget=.02, identity_certification=False,
        hard_runtime_limit_seconds=43200)
    original = dict(cfg)
    amendment = dict(old_disk_reserve_bytes=5*1024**3,
        new_disk_reserve_bytes=2*1024**3, unchanged_artifact_cap_bytes=1024**3)
    revised = runtime_settings(cfg, amendment)
    assert cfg == original
    assert {k for k in revised if revised[k] != cfg[k]} == {'disk_reserve_bytes'}
    assert revised['disk_reserve_bytes'] == 2*1024**3


def test_resource_amendment_rejects_changed_artifact_limit():
    with pytest.raises(ValueError, match='cap'):
        runtime_settings(dict(disk_reserve_bytes=5, artifact_cap_bytes=3),
            dict(old_disk_reserve_bytes=5, new_disk_reserve_bytes=2,
                unchanged_artifact_cap_bytes=1))
