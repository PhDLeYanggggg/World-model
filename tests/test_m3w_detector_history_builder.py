import pytest

from scripts import build_m3w_detector_history_store as run


def test_pilot_enters_runtime_before_raw_archive_or_feature_access(monkeypatch,tmp_path):
    class EnteredRuntime(Exception):
        pass
    def stop(_):
        raise EnteredRuntime('runtime function entered')
    monkeypatch.setattr(run,'PUBLIC',tmp_path)
    monkeypatch.setattr(run,'admission',lambda: ({},{},None,None))
    monkeypatch.setattr(run,'runtime',stop)
    with pytest.raises(EnteredRuntime,match='runtime function entered'):
        run.execute('pilot')
