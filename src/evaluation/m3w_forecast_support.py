"""Exact source-packet row alignment for the fixed-floor support diagnostic."""
import numpy as np
from src.world_model.m3w_source_checkpoint import source_partition
from src.world_model.m3w_source_forest import fingerprint


def reconstruct_fit_ids(data,item,args):
    identity=item['packet']['identity'];source=identity['source']
    ids=np.flatnonzero(data['sites']==source)
    tr,_,partition=source_partition(data['recordings'][ids],data['frames'][ids],source)
    tid=ids[tr]
    if partition!=identity['partition'] or fingerprint(tid)!=identity['training_ids_sha256']:
        raise ValueError('Original TRAIN row identity does not reproduce')
    fit=np.isin(data['recordings'][tid].astype(str),item['partition']['fit_recordings'])
    fid=tid[fit]
    for expected,actual in ((data['sites'][fid],args[4]),
                           (data['recordings'][fid].astype(str),args[5].astype(str)),
                           (data['frames'][fid],args[6])):
        np.testing.assert_array_equal(expected,actual)
    if len(fid)!=len(args[0]):raise ValueError('Aligned feature rows required')
    return fid


def aligned_predictions(ids,bank_ids,prediction):
    ids,bank_ids=np.asarray(ids),np.asarray(bank_ids)
    if np.any(np.diff(bank_ids)<=0):raise ValueError('Sorted unique prediction IDs required')
    at=np.searchsorted(bank_ids,ids)
    if np.any(at>=len(bank_ids)) or not np.array_equal(bank_ids[at],ids):
        raise ValueError('Missing source-excluded prediction')
    return np.asarray(prediction)[at]
