import copy
import numpy as np
import pytest
from src.data_unification.m3w_model_selection_inputs import build_inputs
from src.data_unification import m3w_selection_feature_bridge as api
from src.world_model.m3w_detector_forecaster import DetectorGeometry,pack_geometry
from tests.test_m3w_observation_quality import source


def test_exact_partial_context_and_actual_confidence_not_legacy_complete_only():
    raw=source();inputs=build_inputs(raw,[84]);result=api.pack_record(inputs,raw)
    assert result['geometry'].shape==(1,476) and result['detector_history'].shape==(1,64)
    assert result['geometry'][:,230:294].sum()==2
    assert result['partial_neighbor_count'].tolist()==[1]
    np.testing.assert_array_equal(result['detector_history'][:,16:24],np.full((1,8),.9,np.float32))
    g,ids=api.pack_scene({k:v for k,v in inputs.items() if k!='query_offsets'})
    assert g[:,230:294].sum()==0
    np.testing.assert_array_equal(result['geometry'][:,:38],g[:,:38])
    np.testing.assert_array_equal(result['geometry'][:,294:],g[:,294:])


def test_future_geometry_confidence_mutation_and_truncation_do_not_change_features():
    raw=source();before=api.pack_record(build_inputs(raw,[84]),raw);changed=raw.copy()
    for field in ('x_min','y_min','x_max','y_max'):changed[field][changed['frame']>84]+=1111
    changed['confidence'][changed['frame']>84]=.01
    for value in (changed,raw[raw['frame']<=84]):
        actual=api.pack_record(build_inputs(value,[84]),value)
        for field in api.FIELDS:np.testing.assert_array_equal(before[field],actual[field])


def test_future_fields_and_misaligned_raw_boxes_are_rejected():
    raw=source();inputs=build_inputs(raw,[84]);bad=dict(inputs,future_valid=np.ones((2,12),bool))
    with pytest.raises(ValueError):api.pack_record(bad,raw)
    changed=raw.copy();changed['x_min'][0]+=1
    with pytest.raises(AssertionError):api.pack_record(inputs,changed)


def test_actual540field_model_pack_preserves_masked_partial_tokens():
    raw=source();v=api.pack_record(build_inputs(raw,[84]),raw)
    geometry=DetectorGeometry(v['geometry'],v['detector_history'],np.zeros(64),np.ones(64))
    inputs=pack_geometry(geometry[np.array([0])])
    assert inputs['neighbor_mask'].sum()==2 and inputs['detector_features'].shape==(1,8,8)
    assert not inputs['history'][...,2].gt(0).any()


def receipt_fixture():
    roles=dict(recordings=[dict(locality_group=f's{i}',role='source_training') for i in range(12)]+
        [dict(locality_group='selection',role='model_selection_reserved')])
    rows=[]
    for fold in range(3):
        for seed in (17,29,43):
            sites=[f's{i}' for i in range(4*fold,4*fold+4)]
            for arm in api.REPRESENTATIONS:
                rows.append(dict(trial=f'single{fold}_seed{seed}',representation=arm,
                    parent_trial=dict(fold=fold,seed=seed,fit_sites=sites),
                    normalizer=dict(fit_sites=sites,future_support_used=False,mean=[0]*64,std=[1]*64,clip=8.),
                    fit=dict(step=4000,held_rows_sampled=0,parameters=89026)))
    return rows,roles


def test_complete_real_seed_family_and_no_selection_normalization():
    rows,roles=receipt_fixture();assert len(api.validate_lineage(rows,roles))==36
    with pytest.raises(ValueError):api.validate_lineage(rows[:-1],roles)
    bad=copy.deepcopy(rows);bad[0]['normalizer']['fit_sites']=['selection']
    with pytest.raises(ValueError):api.validate_lineage(bad,roles)
    bad=copy.deepcopy(rows);bad[0]['fit']['held_rows_sampled']=1
    with pytest.raises(ValueError):api.validate_lineage(bad,roles)
