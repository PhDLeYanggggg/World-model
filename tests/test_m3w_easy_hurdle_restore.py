import hashlib
import io
import tarfile
import pytest
from scripts.restore_m3w_easy_hurdle_heads import install_bundle


def bundle(name='heads/g/marginal/checkpoint.pt.gz',raw=b'checkpoint',kind=None):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:') as t:
        e=tarfile.TarInfo(name);e.size=len(raw)
        if kind:e.type=kind;e.size=0;e.linkname='/not_allowed'
        t.addfile(e,io.BytesIO(raw))
    return buf.getvalue(),[dict(path=name,sha256=hashlib.sha256(raw).hexdigest())]


def test_restore_exact_bytes_and_idempotent_resume(tmp_path):
    raw,refs=bundle();calls=[]
    assert install_bundle(raw,refs,tmp_path,calls.append)==10
    assert install_bundle(raw,refs,tmp_path,lambda _:pytest.fail('Existing bytes should be rechecked'))==10
    assert calls==[10] and (tmp_path/refs[0]['path']).read_bytes()==b'checkpoint'


@pytest.mark.parametrize('name',['../x','/x','other/g/marginal/checkpoint.pt.gz'])
def test_restore_rejects_path_escape_and_unrelated_files(tmp_path,name):
    raw,refs=bundle(name)
    with pytest.raises(ValueError):install_bundle(raw,refs,tmp_path,lambda _:None)
    assert not list(tmp_path.iterdir())


def test_restore_rejects_links_bad_hash_and_existing_mutation(tmp_path):
    raw,refs=bundle(kind=tarfile.SYMTYPE)
    with pytest.raises(ValueError):install_bundle(raw,refs,tmp_path,lambda _:None)
    raw,refs=bundle();refs[0]['sha256']='0'*64
    with pytest.raises(ValueError):install_bundle(raw,refs,tmp_path,lambda _:None)
    raw,refs=bundle();p=tmp_path/refs[0]['path'];p.parent.mkdir(parents=True);p.write_bytes(b'old')
    with pytest.raises(ValueError,match='differs'):install_bundle(raw,refs,tmp_path,lambda _:None)
    assert p.read_bytes()==b'old'


def test_restore_obeys_reserve_without_creating_checkpoint(tmp_path):
    raw,refs=bundle()
    def refuse(_):raise OSError('reserve')
    with pytest.raises(OSError):install_bundle(raw,refs,tmp_path,refuse)
    assert not list(tmp_path.iterdir())
