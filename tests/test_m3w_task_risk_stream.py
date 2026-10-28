import ast
import base64
import hashlib
import io
import json
import tarfile

from scripts import submit_m3w_task_risk_stream as api


def test_only_installation_transport_changes():
    code = api.installer(); ast.parse(code)
    assert len(code.encode()) < 8192
    assert 'sys.argv[3]' not in code and 'sys.argv[4]' not in code
    assert "assert seen==reg['bindings']" in code
    assert "exact(root/'registration.json',regraw.encode())" in code


def test_streamed_registration_and_sources_preserve_exact_bytes():
    reg = json.loads(api.original.REG.read_text())
    content = json.loads(api.payload(reg))
    assert content['registration'].encode() == api.original.REG.read_bytes()
    assert content['config'].encode() == api.original.CONFIG.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(content['archive'])),mode='r:gz') as tar:
        seen = {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
    assert seen == reg['bindings']
