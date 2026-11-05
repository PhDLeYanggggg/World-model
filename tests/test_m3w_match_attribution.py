import numpy as np
import pytest
from src.evaluation.m3w_match_attribution import summarize


def test_block_distance_accounting_and_empty_strata():
    q = np.zeros((3,120)); d = np.ones((3,32,120))
    r = summarize(q,d,np.ones((3,32)),np.zeros((3,2)),np.ones((3,2)),np.ones(3))
    assert sum(b['squared_distance'] for b in r['all']['blocks'].values()) == 120
    assert r['admitted_unsafe']['blocks'] is None
    assert r['rejected']['rows'] == 3


def test_labels_change_strata_not_causal_distance():
    q = np.zeros((3,120)); d = np.ones((3,32,120)); w=np.ones((3,32))
    a = summarize(q,d,w,np.zeros((3,2)),-np.ones((3,2)),np.ones(3))
    b = summarize(q,d,w,np.ones((3,2)),-np.ones((3,2)),np.ones(3))
    assert a['all']['blocks'] == b['all']['blocks']
    assert b['admitted_unsafe']['rows'] == 3
    with pytest.raises(ValueError): summarize(q,d,w*0,np.ones((3,2)),np.ones((3,2)),np.ones(3))
