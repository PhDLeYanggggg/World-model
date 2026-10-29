import numpy as np
import pytest
from src.evaluation.m3w_forecast_support import reconstruct_fit_ids, aligned_predictions
from src.world_model.m3w_source_checkpoint import source_partition
from src.world_model.m3w_source_forest import fingerprint


def test_whole_record_alignment_and_tamper_rejection():
    data=dict(sites=np.array(['a']*12+['b']*4),recordings=np.repeat([1,2,3,4],4),frames=np.tile(np.arange(4),4))
    ids=np.arange(12);tr,_,part=source_partition(data['recordings'][ids],data['frames'][ids],'a')
    tid=ids[tr];keep=str(data['recordings'][tid[0]])
    fid=tid[data['recordings'][tid].astype(str)==keep]
    item=dict(packet=dict(identity=dict(source='a',partition=part,training_ids_sha256=fingerprint(tid))),
              partition=dict(fit_recordings=[keep]))
    args=(np.zeros((len(fid),380)),None,None,None,data['sites'][fid],data['recordings'][fid],data['frames'][fid],{})
    np.testing.assert_array_equal(reconstruct_fit_ids(data,item,args),fid)
    changed=(*args[:6],args[6]+1,{})
    with pytest.raises(AssertionError):reconstruct_fit_ids(data,item,changed)


def test_prediction_alignment_never_wraps_or_drops():
    bank=np.array([1,3,5]);prediction=np.arange(6).reshape(3,2)
    np.testing.assert_array_equal(aligned_predictions(np.array([5,1]),bank,prediction),prediction[[2,0]])
    for wrong in (np.array([0]),np.array([6]),np.array([2])):
        with pytest.raises(ValueError):aligned_predictions(wrong,bank,prediction)
