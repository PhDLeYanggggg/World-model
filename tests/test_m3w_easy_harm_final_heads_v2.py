import hashlib
import json

import pytest

from scripts import read_m3w_easy_harm_final_heads_v2 as reader
from tests.test_m3w_easy_harm_final_heads import bundle as old_bundle


def bundle():
    freeze,files,expected,cfg,sha,_=old_bundle()
    amendment=dict(previous_amendment_sha256='v1',references={},preserved_fit_refs=[])
    freeze.update(historical_quadratic_controls_exact=54,historical_nonexact_replay_verified=18,
        control_execution_amendment_sha256='v2',previous_amendment_sha256='v1',
        preserved_accepted_fits=110,adopted_diagnostic_quadratic_fits=17,new_candidate_fits=17)
    for i,ref in enumerate(freeze['fits']):
        row=json.loads(files[ref['path']]);row.update(independent_roles_read=False,new_forecaster=False)
        if i%2==0 and i//2%4==0:
            identity=row['identity'];name=ref['path'].split('/')[-1][:-5]
            cp=dict(path='proof/'+name,sha256='d'*64)
            amendment['references'][name]=dict(identity=identity,reference_checkpoint=cp,preserve_v1=i==0)
            row.update(parent_control_exact=False,historical_control_exact=False,
                original_implementation_control_exact=True,floating_tolerance_relaxed=False,
                reference_checkpoint=cp,amendment_sha256='v1' if i==0 else 'v2')
        raw=json.dumps(row).encode();files[ref['path']]=raw;ref['sha256']=hashlib.sha256(raw).hexdigest()
    amendment['preserved_fit_refs']=[r.copy() for i,r in enumerate(freeze['fits']) if i<2 or i//2%4!=0]
    return freeze,files,expected,cfg,sha,amendment,'v2'


def test_complete54_plus18_grid_is_admitted():
    assert len(reader.validate(*bundle()))==144


@pytest.mark.parametrize('change',['partial','duplicate','extra','hash','scored','historical_claim','proof','preservation','tolerance','step'])
def test_no_incomplete_or_changed_readout(change):
    freeze,files,expected,cfg,sha,amend,amendsha=bundle()
    if change=='partial':freeze['fits']=freeze['fits'][:110]
    elif change=='duplicate':freeze['fits'][-1]=freeze['fits'][0]
    elif change=='extra':files['extra']=b'{}'
    elif change=='hash':freeze['fits'][0]['sha256']='f'*64
    elif change=='scored':freeze['independent_roles_read']=True
    elif change=='historical_claim':freeze['historical_bitwise_reproduction_complete']=True
    elif change=='proof':next(iter(amend['references'].values()))['reference_checkpoint']['sha256']='f'*64
    elif change=='preservation':amend['preserved_fit_refs'][0]['sha256']='f'*64
    else:
        ref=freeze['fits'][0];row=json.loads(files[ref['path']])
        row['floating_tolerance_relaxed' if change=='tolerance' else 'step']=True if change=='tolerance' else 1999
        files[ref['path']]=json.dumps(row).encode();ref['sha256']=hashlib.sha256(files[ref['path']]).hexdigest()
    with pytest.raises(ValueError):reader.validate(freeze,files,expected,cfg,sha,amend,amendsha)


def test_stream_transport_does_not_change_registered_submit():
    from scripts import submit_m3w_easy_harm_recovery_stream as transport
    reg=json.loads(transport.recovery.REG.read_text());code,raw=transport.payload(reg)
    import ast
    ast.parse(code)
    assert len(code.encode())<32768
    header,archive=raw.split(b'\n',1)
    assert json.loads(header)[2]==transport.recovery.REG.read_text()
    assert archive[:2]==b'\x1f\x8b'
    assert repr(transport.recovery.SUBMIT) in code
