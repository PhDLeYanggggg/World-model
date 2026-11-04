import hashlib
from pathlib import Path
import shutil
from zipfile import ZipFile

from scripts import compress_verified_m3w_reports as api


def test_no_saving_keeps_original_bytes_and_immutable_archive_proof(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'ROOT',tmp_path)
    path=tmp_path/'reports/groups/a.json';path.parent.mkdir(parents=True)
    text=b'{"metric": "'+b'x'*8000+b'"}\n';path.write_bytes(text)
    archive=tmp_path/'reports/metric_groups.zip'
    with ZipFile(archive,'w')as zipped:zipped.writestr('groups/a.json',text)
    proof=dict(path='reports/metric_groups.zip',sha256=api.sha(archive),member='groups/a.json')
    def copy(command,**kwargs):shutil.copy2(command[-2],command[-1])
    monkeypatch.setattr(api.subprocess,'run',copy)
    row=api.compress(path,proof)
    assert row['status']=='no_physical_saving_original_unchanged'
    assert row['archive']==proof and row['saved_bytes']==0
    assert path.read_bytes()==text and row['sha256']==hashlib.sha256(text).hexdigest()
    assert not path.with_name(path.name+'.m3w-fscompress-tmp').exists()
