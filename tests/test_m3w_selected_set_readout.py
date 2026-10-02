import base64
import copy
import json
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import verify_m3w_selected_set_readout as audit
from scripts import run_m3w_selected_set_calibration as run


def fixture():
    n = 9
    p = np.tile([.4,.002,2.,2.,.002],(n,1))
    y = np.tile([0.,.015,1.,1.,.015],(n,1))
    z = dict(p=p,y=y,env=np.ones(n),moving=np.ones(n,bool),support=np.ones(n,bool),
             recordings=np.repeat(['a','b','c'],3))
    old, act = run.api.parent.calibrate(**z)
    old['identity'] = dict(source='source',target_hash=audit.array_hash(y),prediction_hash=audit.array_hash(p))
    old['oof_action_hashes'] = {k:audit.array_hash(v) for k,v in act.items()}
    meta = dict(source='source',identity=dict(source='source'),
        array_hashes={k:audit.array_hash(v) for k,v in z.items()},
        parent_source=old['source'],parent_final=old['final'],parent_oof_action_hashes=old['oof_action_hashes'])
    z['meta_json'] = np.array(json.dumps(meta))
    cfg = dict(round_cap=8,empirical_quantile=.9,bootstrap_seed=20261002,bootstrap_resamples=3000)
    group = run.compute(z,cfg)
    return z,group,old,cfg


def test_scalar_check_and_independent_inference():
    z,g,old,_ = fixture()
    got = audit.verify_group(z,g,old)
    assert got['scalar_checks'] == 276 and got['action_hash_checks'] == 12
    assert len(got['rows']) == 6


def test_metric_denominator_tampering_rejected():
    z,g,old,_ = fixture()
    g['result']['rows'][0]['selected_set']['selected_known_reference_mass'] += 1
    with pytest.raises(AssertionError): audit.verify_group(z,g,old)


def test_held_recording_leak_rejected():
    z,g,old,_ = fixture()
    fold = g['result']['rows'][0]['folds'][0]
    fold['calibration']['recordings'].append(fold['held_recording'])
    with pytest.raises(AssertionError): audit.verify_group(z,g,old)


def test_summary_bootstrap_reproduced_and_wrong_pass_rejected():
    _,g,_,cfg = fixture()
    groups = [copy.deepcopy(g) for _ in range(72)]
    for i,x in enumerate(groups): x['source'] = 'site'+str(i%12)
    summary = run.summary(groups,cfg)
    assert audit.verify_summary(summary,groups,cfg) > 100
    summary['comparisons']['joint_source_oof']['selected_set']['complete_support_pass'] += 1
    with pytest.raises(AssertionError): audit.verify_summary(summary,groups,cfg)


def test_unknown_and_zero_denominator_never_certified():
    y = np.array([[0.,.1,1.,1.,.1],[np.nan]*5])
    empty = audit.scalar_bounds(y,np.array([False,False]),np.ones(2))
    assert empty['easy_selected_risk_upper'] is None and not empty['finite_completion_supported']
    got = audit.scalar_bounds(y,np.array([True,True]),np.ones(2))
    assert got['selected_unknown'] == 1 and got['easy_selected_risk_upper'] == 1.1


def test_reader_does_not_import_torch_in_isolated_process():
    code = 'import runpy,sys;runpy.run_path(sys.argv[1],run_name="verifier_test");assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(audit.__file__).resolve())],check=True)


def test_compressed_packet_materialized_once_and_matches_dict(monkeypatch):
    z,g,old,_ = fixture()
    buf = io.BytesIO();np.savez_compressed(buf,**z)
    original = np.lib.npyio.NpzFile.__getitem__
    calls = {}
    def counted(self,key):
        calls[key] = calls.get(key,0)+1
        return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',counted)
    arrays = audit.load_packet(buf.getvalue())
    assert calls == {key:1 for key in z}
    assert audit.verify_group(arrays,g,old) == audit.verify_group(z,g,old)
    assert calls == {key:1 for key in z}


def bundle_fixture():
    ins,outs,ir,orr = {},{},{},{}
    for i in range(72):
        name = 'group'+str(i);raw = b'input'+str(i).encode();out = json.dumps({'i':i})
        ins[name] = base64.b64encode(raw).decode();outs[name] = out
        ir[name] = dict(group=name,bytes=len(raw),sha256=audit.digest(raw))
        orr[name] = dict(group=name,bytes=len(out.encode()),sha256=audit.digest(out.encode()))
    manifest = dict(packets=list(ir.values()))
    summary = '{}'
    complete = dict(groups=list(orr.values()),job_id='123',source_heads=72,exact_replay=True,
        new_neural_updates=0,transfer_evaluated=False,independent_roles_read=False,
        registration_sha256='reg',summary_sha256=audit.digest(summary.encode()))
    return dict(inputs=ins,outputs=outs,manifest=manifest,complete=complete,summary_text=summary,
        registration_sha256='reg',accounting=dict(returncode=0,stdout='123|COMPLETED|0:0|00:01:00\n'))


def test_hash_bound_completed_bundle():
    b = bundle_fixture()
    assert audit.validate_bundle(b,b['manifest'],'reg','123') > 0


@pytest.mark.parametrize('mutation',['pending','no_replay','missing','changed_input','changed_summary','wrong_registration'])
def test_bundle_rejects_unverified_completion(mutation):
    b = bundle_fixture()
    if mutation == 'pending': b['accounting']['stdout'] = '123|PENDING|0:0|00:00:00\n'
    if mutation == 'no_replay': b['complete']['exact_replay'] = False
    if mutation == 'missing': b['outputs'].pop('group0')
    if mutation == 'changed_input': b['inputs']['group0'] = base64.b64encode(b'changed').decode()
    if mutation == 'changed_summary': b['summary_text'] = '{"passes":72}'
    if mutation == 'wrong_registration': b['complete']['registration_sha256'] = 'wrong'
    with pytest.raises(AssertionError): audit.validate_bundle(b,b['manifest'],'reg','123')
