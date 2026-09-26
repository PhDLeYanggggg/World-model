import numpy as np
import pytest
from scripts.diagnose_m3w_european_nested_residual import accounting


@pytest.mark.parametrize('new,kind',[(.5,'improved'),(1.,'unchanged'),(2.,'wrong_aggregate_direction'),(-2.,'useful_direction_excess_magnitude')])
def test_error_accounting_categories(new,kind):
    a=accounting([1.,100.],[new,np.nan],[0.,np.nan],[True,True])
    assert a['rows']==1 and a['description']==kind
    assert a['MSE_change']==a['cross_term']+a['shift_energy']


def test_no_known_rows_rejected():
    with pytest.raises(ValueError): accounting([1.],[2.],[np.nan],[True])
