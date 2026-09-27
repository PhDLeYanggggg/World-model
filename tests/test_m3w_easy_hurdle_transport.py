import subprocess
import hashlib
import json
import sys
import pytest
from scripts import export_m3w_easy_hurdle_create as transport
from scripts.m3w_packet_stream import RECEIVER


def test_idempotent_remote_retry_preserves_payload(monkeypatch):
    calls=[]; waits=[]
    def call(cmd, **kw):
        calls.append((cmd,kw))
        if len(calls)==1:
            return subprocess.CompletedProcess(cmd,255,b'',b'Connection reset by peer')
        return subprocess.CompletedProcess(cmd,0,b'{"verified":true}',b'')
    monkeypatch.setattr(transport.subprocess,'run',call)
    monkeypatch.setattr(transport.time,'sleep',waits.append)
    assert transport.remote(['ssh'],'code',['name'],b'bound')=={'verified':True}
    assert waits==[30] and calls[0]==calls[1]


def test_authentication_error_not_blindly_retried(monkeypatch):
    def call(cmd, **kw):
        return subprocess.CompletedProcess(cmd,255,b'',b'Permission denied (publickey)')
    monkeypatch.setattr(transport.subprocess,'run',call)
    monkeypatch.setattr(transport.time,'sleep',lambda _:pytest.fail('Auth must not retry'))
    with pytest.raises(RuntimeError,match='Permission denied'):
        transport.remote(['ssh'],'code')


def test_transport_retries_are_bounded(monkeypatch):
    calls=[]; waits=[]
    def call(cmd, **kw):
        calls.append(1)
        return subprocess.CompletedProcess(cmd,255,b'',b'Connection closed')
    monkeypatch.setattr(transport.subprocess,'run',call)
    monkeypatch.setattr(transport.time,'sleep',waits.append)
    with pytest.raises(RuntimeError,match='Connection closed'):
        transport.remote(['ssh'],'code')
    assert len(calls)==3 and waits==[30,90]


def test_stream_receiver_preserves_and_rechecks_multiple_packets(tmp_path):
    (tmp_path/'.owner.json').write_text(json.dumps({'experiment':'european_easy_hurdle_v1'}))
    content=b''
    for name,data in [('g1',b'one'),('g2',b'two'),('g1',b'one')]:
        header=dict(group=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        content+=(json.dumps(header)+'\n').encode()+data
    r=subprocess.run([sys.executable,'-c',RECEIVER,str(tmp_path)],input=content,capture_output=True,timeout=10)
    assert r.returncode==0,r.stderr
    assert len(r.stdout.splitlines())==4
    assert (tmp_path/'inputs/g1.npz').read_bytes()==b'one'
    assert (tmp_path/'inputs/g2.npz').read_bytes()==b'two'


@pytest.mark.parametrize('name,sha',[('../bad',hashlib.sha256(b'x').hexdigest()),('good','0'*64)])
def test_stream_rejects_path_escape_or_corruption(tmp_path,name,sha):
    (tmp_path/'.owner.json').write_text(json.dumps({'experiment':'european_easy_hurdle_v1'}))
    raw=(json.dumps(dict(group=name,bytes=1,sha256=sha))+'\n').encode()+b'x'
    r=subprocess.run([sys.executable,'-c',RECEIVER,str(tmp_path)],input=raw,capture_output=True,timeout=10)
    assert r.returncode!=0 and not (tmp_path/'inputs').exists()
