import json

import numpy as np
import pytest

from scripts.fetch_m3w_european_squares import digest
from src.world_model.m3w_detector_feature_store import DetectorFeatureStore


def test_store_identity_mismatch_is_not_a_fake_future_or_unknown_zero(tmp_path):
    report=dict(rows=149510,recordings=74,query_future_labels_read=False,independent_roles_opened=False)
    p=tmp_path/'complete.json';p.write_text(json.dumps(report))
    (tmp_path/'verification.json').write_text(json.dumps(dict(raw_reparsed=False,all_arrays_exact=True,complete_sha256=digest(p))))
    with pytest.raises(ValueError,match='Full verified'):
        DetectorFeatureStore(tmp_path,tmp_path)


def test_unknown_row_request_is_rejected_instead_of_padded():
    store=object.__new__(DetectorFeatureStore);store.arrays=dict(ids=np.array([1,3,5]),features=np.zeros((3,64)),
        recordings=np.ones(3,int),agents=np.arange(3),frames=np.arange(3))
    with pytest.raises(ValueError,match='outside'):store.read(np.array([2]),'history')
    assert store.read(np.array([1,5]),'history').shape==(2,64)
