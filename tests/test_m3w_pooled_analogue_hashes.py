import copy
import numpy as np
import pytest
from scripts import run_m3w_pooled_analogue as run


def test_packet_contract_not_analogue_output_contract():
    args=tuple(np.array([1.,2.],dtype=np.float32) for _ in range(7))+({'training_site':'s'},)
    names=('x','envelope','primary','temporal','sites','recordings','frames')
    correct=run.previous.previous.train.api.fingerprint
    assert correct(args[0])!=run.previous.api.fingerprint(args[0])
    state=dict(anchor_state={'source_preprocess':{'s':args[-1]}},
               input_hashes={'s':{k:correct(a) for k,a in zip(names,args[:-1])}})
    run.verify_source_input(state,'s',args)
    bad=copy.deepcopy(state);bad['input_hashes']['s']['x']=run.previous.api.fingerprint(args[0])
    with pytest.raises(ValueError,match='Input changed'):run.verify_source_input(bad,'s',args)


def test_query_contract_uses_original_selection_format():
    q=np.array([True,False]);correct=run.previous.previous.train.api.fingerprint
    run.verify_query(q,{'parent_cohort_sha256':correct(q)})
    with pytest.raises(ValueError,match='selection'):
        run.verify_query(q,{'parent_cohort_sha256':run.previous.api.fingerprint(q)})
