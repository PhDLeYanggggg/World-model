import copy
import hashlib
import json
import numpy as np
import pytest
from src.data_unification import m3w_model_selection_inputs as api
from src.data_unification import m3w_european_squares_source as source
from src.evaluation.m3w_european_squares_roles import require_source_training
from tests.test_m3w_european_squares_source import rows


def manifest():
    return dict(status='restricted_source_training_admitted_reserved_roles_closed',
        recordings=[dict(source_member='selection.csv',role='model_selection_reserved',
            rows_sha256='raw_identity',training_access=False,predictive_readout_access=False)])


def test_admission_is_input_only_and_never_opens_other_roles():
    m=manifest(); assert api.require_input_role(m,'selection.csv') == m['recordings'][0]
    with pytest.raises(PermissionError):require_source_training(m,'selection.csv')
    for role in ('source_training','risk_calibration_reserved','confirmation_reserved'):
        changed=copy.deepcopy(m);changed['recordings'][0]['role']=role
        with pytest.raises(PermissionError):api.require_input_role(changed,'selection.csv')
    for field in ('training_access','predictive_readout_access'):
        changed=copy.deepcopy(m);changed['recordings'][0][field]=True
        with pytest.raises(PermissionError):api.require_input_role(changed,'selection.csv')
    with pytest.raises(ValueError):api.require_input_role(m,'missing')
    m['recordings']*=2
    with pytest.raises(ValueError):api.require_input_role(m,'selection.csv')


def test_same_causal_schema_retains_partial_neighbors_and_futureless_targets():
    values=api.build_inputs(rows(),[84,348]); old,labels=source.build_recording(rows(),[84,348])
    assert set(values)==set(source.INPUT_FIELDS) and not set(values)&set(source.LABEL_FIELDS)
    for key in values:np.testing.assert_array_equal(values[key],old[key])
    assert not labels['future_valid'][-1].any() and values['target_eligible'][-1]
    assert not values['target_eligible'][1] and values['history_valid'][1].sum()==2


def test_lookup_never_requests_future_and_future_mutation_changes_no_input(monkeypatch):
    original=source.lookup; seen=[]
    def lookup(track,grid):
        assert np.max(grid)<=84;seen.append(grid.copy());return original(track,grid)
    monkeypatch.setattr(source,'lookup',lookup)
    raw=rows();before=api.build_inputs(raw,[84]);changed=raw.copy()
    for f in ('x_min','y_min','x_max','y_max'):changed[f][changed['frame']>84]+=12345
    after=api.build_inputs(changed,[84]);prefix=api.build_inputs(raw[raw['frame']<=84],[84])
    assert len(seen)==6
    for key in before:
        np.testing.assert_array_equal(before[key],after[key]);np.testing.assert_array_equal(before[key],prefix[key])


def test_invalid_queries_rejected_and_empty_queries_are_valid():
    for q in ([84,84],[84,72],[84.5],[-1]):
        with pytest.raises(ValueError):api.build_inputs(rows(),q)
    empty=api.build_inputs(rows(),np.array([],np.int64))
    assert api.input_statistics(empty)['visible_rows']==0


def test_statistics_are_past_only_and_include_zero_support_queries():
    values=api.build_inputs(rows(),[0,84,400]); stats=api.input_statistics(values)
    assert stats['queries']==3 and stats['target_rows']==1
    assert stats['queries_with_at_least_targets']=={'1':1,'2':0,'5':0,'10':0}
    assert stats['future_target_arrays_constructed']==stats['forecast_errors_computed']==0
    bad=dict(values,future_valid=np.ones((len(values['agent_id']),12),bool))
    with pytest.raises(ValueError):api.input_statistics(bad)


def test_cache_consumer_refuses_label_fields_before_array_access(tmp_path,monkeypatch):
    r=dict(source_member='selection.csv',role='model_selection_inputs_only',arrays={'label_future_xy':{}})
    (tmp_path/'receipt.json').write_text(json.dumps(r))
    def deny(*a,**kw):raise AssertionError('No array should be opened')
    monkeypatch.setattr(np,'load',deny)
    with pytest.raises(ValueError):api.load_record(tmp_path,manifest(),'selection.csv',registration_sha256='frozen')


def test_lazy_cache_roundtrip_binds_raw_identity_registration_and_arrays(tmp_path):
    values=api.build_inputs(rows(),[84]);arrays={}
    for key,value in dict(**{'input_'+k:v for k,v in values.items()},index_query_frames=np.array([84])).items():
        path=tmp_path/(key+'.npy');np.save(path,value,allow_pickle=False)
        arrays[key]=dict(name=path.name,shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    r=dict(source_member='selection.csv',role='model_selection_inputs_only',rows_sha256='raw_identity',
        registration_sha256='frozen',arrays=arrays)
    (tmp_path/'receipt.json').write_text(json.dumps(r))
    restored,query=api.load_record(tmp_path,manifest(),'selection.csv',registration_sha256='frozen')
    np.testing.assert_array_equal(query,[84]);assert not query.flags.writeable
    for k,v in restored.items():np.testing.assert_array_equal(v,values[k]);assert not v.flags.writeable
    with pytest.raises(ValueError):api.load_record(tmp_path,manifest(),'selection.csv',registration_sha256='wrong')
    arrays['input_history_xy']['name']='label_future_xy.npy'
    (tmp_path/'receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):api.load_record(tmp_path,manifest(),'selection.csv',registration_sha256='frozen')
