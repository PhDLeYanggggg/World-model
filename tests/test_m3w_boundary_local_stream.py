import hashlib
import io
import json
import numpy as np
import pytest
from tests.test_m3w_boundary_diagnostic import packet
from scripts.run_m3w_boundary_local_stream import read_exact, evaluate_packet


def test_read_exact_handles_short_reads_and_truncation():
    class Short(io.BytesIO):
        def read(self,n=-1):return super().read(min(n,2))
    assert read_exact(Short(b'abcdef'),6)==b'abcdef'
    with pytest.raises(EOFError):read_exact(Short(b'abc'),4)
    with pytest.raises(ValueError):read_exact(Short(b''),129*2**20)


def example():
    a,m,c=packet();m['view']='test_group';a['meta_json']=np.array(json.dumps(m))
    out=io.BytesIO();np.savez_compressed(out,**a);raw=out.getvalue()
    return raw,dict(group='test_group',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),c


def test_frozen_packet_replay_and_accounting():
    raw,entry,c=example();r,n,t=evaluate_packet(raw,entry,c)
    assert r['unknown_rows']==1 and n>100 and len(t)==2 and all(x>0 for x in t)


def test_wrong_hash_or_alignment_rejected():
    raw,entry,c=example()
    with pytest.raises(AssertionError):evaluate_packet(raw+b'x',entry,c)
    entry['group']='wrong_group'
    with pytest.raises(AssertionError):evaluate_packet(raw,entry,c)
