from scripts.diagnose_m3w_european_cost_shape import transport_counts


def test_fit_success_is_not_held_success():
    got=transport_counts([(1.,2.,1.,2.),(1.,2.,3.,2.),(3.,2.,1.,2.),(3.,2.,3.,2.)])
    assert got==dict(views=4,missing=0,fit_and_held_improved=1,fit_only=1,held_only=1,neither=1)


def test_missing_and_ties_are_retained():
    got=transport_counts([(None,2.,1.,2.),(2.,2.,2.,2.)])
    assert got['views']==2 and got['missing']==1 and got['neither']==1
