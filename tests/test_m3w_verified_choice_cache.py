import hashlib
import json

import numpy as np
import pytest
from src.evaluation.m3w_verified_choice_cache import binding,verified_choices,POLICIES


def test_cached_ledger_sha_and_input_binding_are_required(tmp_path):
    c=dict(allowed=np.ones((3,7),bool),groups=[np.arange(3)])
    packet=dict(ids=np.arange(3),frames=np.ones(3,int),recordings=np.asarray(['r']*3))
    meta=dict(completed_queries=1,bindings=dict(prediction='old_prediction_not_freshly_reinferred',
        seconds=.25,version=1,allowed=binding(c['allowed']),**{k:binding(v)for k,v in packet.items()}))
    path=tmp_path/'choices.npz'
    np.savez_compressed(path,metadata_json=np.asarray(json.dumps(meta)),**{k:np.array([-1,0,1])for k in POLICIES})
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    choices,receipt=verified_choices(path,sha,c,packet,.25)
    assert len(choices)==4 and not receipt['old_cost_prediction_freshly_reinferred']
    with pytest.raises(ValueError):verified_choices(path,'0'*64,c,packet,.25)
    with pytest.raises(ValueError):verified_choices(path,sha,c,dict(packet,frames=packet['frames']+1),.25)
    with pytest.raises(ValueError):verified_choices(path,sha,c,packet,.5)
