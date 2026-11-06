import numpy as np
from scripts.diagnose_m3w_easy_reference_credit import attribution


def test_missing_and_noneasy_are_distinct_and_empty_stays_na():
    a=attribution(np.array([2.,3.,5.]),np.array([1.,0.,0.]),np.array([True,True,False]))
    assert a['known_noneasy_credit_fraction']==.3 and a['predicted_ER_unknown']==5
    assert a['known_easy_selected']==1 and a['unknown_selected']==1
    assert attribution(np.array([]),np.array([]),np.array([],bool))['known_noneasy_credit_fraction'] is None
