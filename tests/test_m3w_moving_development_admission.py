import pytest

from src.evaluation.m3w_moving_development_admission import eligible_train_nodes,group_folds,require_fit_record
from src.evaluation.m3w_recording_lineage import sha256


def test_reserved_source_link_excludes_entire_parent_group_without_reassignment():
    kept,excluded=eligible_train_nodes(['moving_a','moving_b','moving_c'],
        [['moving_a','moving_b','dut_intersection_01'],['moving_c'],['drone_1']])
    assert kept=={'moving_c':1} and set(excluded)=={'moving_a','moving_b'}
    with pytest.raises(ValueError):eligible_train_nodes(['moving_missing'],[['moving_c']])
    with pytest.raises(ValueError):eligible_train_nodes(['moving_a'],[['moving_a'],['moving_a']])


def test_folds_are_group_whole_deterministic_and_use_structural_support_only():
    records={str(i):dict(exclusion_group='g'+str(i//2),complete_native8_12_windows=i+1) for i in range(12)}
    a,total=group_folds(records,'fixed')
    assert a==group_folds(records,'fixed')[0] and set(a.values())=={0,1,2}
    assert sum(total)==sum(r['complete_native8_12_windows'] for r in records.values())


def test_fit_guard_rejects_wrong_role_held_group_unknown_record_and_changed_raw(tmp_path):
    raw=tmp_path/'source.csv';raw.write_text('annotation')
    admission=dict(recordings=dict(moving_a=dict(exclusion_group='g0',source_annotation_sha256=sha256(raw))))
    require_fit_record(admission,'moving_a','fit',raw)
    with pytest.raises(ValueError,match='fit-only'):require_fit_record(admission,'moving_a','confirmation',raw)
    with pytest.raises(ValueError,match='Held'):require_fit_record(admission,'moving_a','fit',raw,held_groups=['g0'])
    with pytest.raises(ValueError,match='not admitted'):require_fit_record(admission,'moving_unknown','fit',raw)
    raw.write_text('changed')
    with pytest.raises(ValueError,match='Changed'):require_fit_record(admission,'moving_a','fit',raw)
