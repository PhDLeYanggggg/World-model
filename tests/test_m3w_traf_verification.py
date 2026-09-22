from copy import deepcopy

import pytest

from scripts.verify_m3w_traf_intake import verify_recording
from src.evaluation.m3w_traf_intake import audit_recording


def test_separate_recount_accepts_gaps_and_numeric_ids(tmp_path):
    p = tmp_path / 'TRAF1_gt.txt'
    p.write_text('\n'.join(f'{i},1,20,30,5,6,7' for i in list(range(245)) + list(range(260, 300))))
    report, _ = audit_recording(p)
    assert verify_recording(p, report)['tracks'] == 1
    altered = deepcopy(report)
    altered['by_class_before_recording_quarantine']['untyped_id']['stride12']['history_and_12step_future']['8'] += 1
    with pytest.raises(AssertionError):
        verify_recording(p, altered)
    p.write_text(p.read_text() + '\n300,0')
    with pytest.raises(ValueError, match='hash changed'):
        verify_recording(p, report)


def test_separate_recount_duplicate_frame_identity(tmp_path):
    p = tmp_path / 'TRAF2_gt.txt'
    p.write_text('0,1,0,0,3,4,ped0\n1,2,0,0,3,4,ped0,0,0,3,4,ped0\n2,1,0,0,3,4,ped0')
    report, _ = audit_recording(p)
    assert verify_recording(p, report)['rejected_frames'] == 1
