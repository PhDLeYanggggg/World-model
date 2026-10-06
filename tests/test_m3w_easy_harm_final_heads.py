import copy
import hashlib
import json

import pytest

from scripts import read_m3w_easy_harm_final_heads as reader


def bundle():
    expected=[dict(group=f'c{c}_s{s}',source=f's{s}',seed=seed)
              for s in range(12) for c in range(2) for seed in (17,29,43)]
    cfg=dict(arms=['quadratic','easy_deviance'])
    amendment=dict(exception_identity=expected[0],reference_checkpoint={'path':'proof','sha256':'d'*64})
    freeze=dict(neural_fits=144,source_heads=72,registration_sha256='reg',
        original_implementation_controls_exact=72,historical_quadratic_controls_exact=71,
        historical_nonexact_replay_verified=1,historical_bitwise_reproduction_complete=False,
        validation_scored=False,independent_roles_read=False,new_forecasters=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False,
        control_execution_amendment_sha256='amend',fits=[])
    files={}
    for identity in expected:
        for arm in cfg['arms']:
            name=identity['group']+'_head'+str(identity['seed'])+'_'+arm
            path=reader.HOME+'/fits/'+name+'.json'
            row=dict(identity=identity,arm=arm,step=2000,validation_scored=False,
                parent_control_exact=arm=='quadratic',checkpoint=dict(path=reader.HEADS+'/'+name+'/checkpoint.pt.gz',bytes=1000,sha256='a'*64))
            if identity==expected[0] and arm=='quadratic':
                row.update(parent_control_exact=False,historical_control_exact=False,
                    original_implementation_control_exact=True,floating_tolerance_relaxed=False,
                    reference_checkpoint=amendment['reference_checkpoint'],amendment_sha256='amend')
            raw=json.dumps(row).encode();files[path]=raw
            freeze['fits'].append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest()))
    return freeze,files,expected,cfg,'reg',amendment


def test_complete_grid_preserves_historical_failure_identity():
    args=bundle();out=reader.validate(*args)
    assert len(out)==144
    assert sum(not row['parent_control_exact'] for row in out.values() if row['arm']=='quadratic')==1


@pytest.mark.parametrize('mutation',['partial','duplicate','extra','bad_hash','scored','fake_historical','wrong_exception','bad_reference','partial_step'])
def test_incomplete_or_changed_grid_never_opens_readout(mutation):
    freeze,files,expected,cfg,sha,amend=bundle()
    if mutation=='partial': freeze['fits']=freeze['fits'][:36]
    elif mutation=='duplicate': freeze['fits'][-1]=freeze['fits'][0]
    elif mutation=='extra': files['extra']=b'{}'
    elif mutation=='bad_hash': freeze['fits'][0]['sha256']='b'*64
    elif mutation=='scored': freeze['validation_scored']=True
    elif mutation=='fake_historical': freeze['historical_bitwise_reproduction_complete']=True
    elif mutation=='wrong_exception': amend['exception_identity']=expected[1]
    elif mutation=='bad_reference': amend['reference_checkpoint']={'sha256':'bad'}
    else:
        path=freeze['fits'][0]['path'];row=json.loads(files[path]);row['step']=100
        raw=json.dumps(row).encode();files[path]=raw;freeze['fits'][0]['sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): reader.validate(freeze,files,expected,cfg,sha,amend)
