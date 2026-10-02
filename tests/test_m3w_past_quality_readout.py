import numpy as np
from scripts.verify_m3w_past_quality_auxiliary import ci,compare
from src.world_model.m3w_forest_projection import interval


def test_bootstrap_matches_independent_scalar_implementation():
    rows=[dict(source=str(i//3),result=dict(contrasts=dict(effect=i*.01))) for i in range(36)]
    direct=interval([(r['source'],r['result']['contrasts']['effect']) for r in rows],3000,20261002)
    compare(direct,ci(rows,'effect',3000,20261002))


def test_duplicated_head_is_not_an_extra_locality():
    rows=[dict(source=str(i),result=dict(contrasts=dict(effect=i))) for i in range(12)]
    compare(ci(rows,'effect',3000,20261002),ci(rows+rows,'effect',3000,20261002))


def test_empty_unknown_contrast_stays_undefined():
    rows=[dict(source='x',result=dict(contrasts=dict(effect=None)))]
    assert ci(rows,'effect',3000,20261002)==dict(mean=None,CI95=None,localities=0)
