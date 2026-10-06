import json
import sys

import pytest

from scripts import run_m3w_easy_harm_readout as run


def setup(monkeypatch,tmp_path,phase):
    monkeypatch.setattr(run,'PUBLIC',tmp_path/'public')
    run.PUBLIC.mkdir();(run.PUBLIC/'registration.json').write_text('{}')
    monkeypatch.setattr(run,'registration',lambda:({},{}))
    monkeypatch.setattr(run.parent.base.inter,'committed',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['readout',phase])
    def must_not_load(): raise AssertionError('Source rows must remain closed')
    monkeypatch.setattr(run.parent.inner.old,'load',must_not_load)


def test_partial_training_cannot_open_source_rows(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,'run')
    def missing(): raise ValueError('Only36fits complete')
    monkeypatch.setattr(run.heads,'local_admission',missing)
    with pytest.raises(ValueError,match='Only36fits'): run.main()


def test_completed_readout_cannot_be_silently_replaced(monkeypatch,tmp_path):
    setup(monkeypatch,tmp_path,'run')
    monkeypatch.setattr(run.heads,'local_admission',lambda:({},{}))
    (run.PUBLIC/'complete.json').write_text('{}')
    with pytest.raises(RuntimeError,match='immutable'): run.main()


def test_preflight_does_not_predict_or_load_scientific_arrays(monkeypatch,tmp_path,capsys):
    setup(monkeypatch,tmp_path,'preflight')
    monkeypatch.setattr(run.heads,'local_admission',lambda:({i:{} for i in range(144)},{}))
    run.main()
    out=json.loads(capsys.readouterr().out)
    assert out['actual_readout']=='not_run' and out['complete_heads']==144
    assert out['independent_roles_read'] is False
