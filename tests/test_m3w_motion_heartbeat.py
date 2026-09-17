import os
from scripts.run_m3w_observed_motion_v2 import heartbeat_payload


def test_training_pid_does_not_collide_with_heartbeat_metadata():
    data = heartbeat_payload({'pid': os.getpid(), 'state': 'training', 'step': 100})
    assert data['pid'] == os.getpid()
    assert data['state'] == 'training'
    assert data['step'] == 100
    assert isinstance(data['time_unix'], float)
