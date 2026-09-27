import copy
import pytest
from scripts import diagnose_m3w_european_cost_mass as r


def fixture():
    rows = []
    for seed, y, p in [(17,1.,2.),(29,10.,10.),(43,100.,50.)]:
        metric = dict(rows=8,positive=3,harm_MSE=float(seed),actual_harm_mean=y,
            predicted_harm_mean=p,harm_coverage=p/y)
        rows.append(dict(pair='full',producer=0,controller=1,seed=seed,folds=[
            dict(held='site',metrics=dict(cost_only_mass=dict(all=metric,envelope_positive=metric)))]))
    return rows


def test_seed_means_and_ratios_are_not_silently_pooled():
    rows = fixture(); before = copy.deepcopy(rows)
    got = r.absolute_rows(rows,[17,29,43]); assert len(got) == 2
    assert got[0]['seed_mean_harm_MSE'] == (17+29+43)/3
    assert got[0]['seed_mean_harm_coverage'] == (2+1+.5)/3
    assert got[0]['seed_mean_harm_coverage'] != (2+10+50)/(1+10+100)
    assert rows == before


def test_missing_seed_metric_is_retained_as_unavailable():
    rows = fixture(); rows[0]['folds'][0]['metrics']['cost_only_mass']['all'] = dict(status='not_estimable')
    got = r.absolute_rows(rows,[17,29,43])[0]
    assert got['seed_mean_harm_MSE'] is None and not got['all_seeds_supported']


def test_missing_seed_rejected_and_input_order_irrelevant():
    rows = fixture()
    with pytest.raises(AssertionError): r.absolute_rows(rows[:2],[17,29,43])
    assert r.absolute_rows(rows,[17,29,43]) == r.absolute_rows(rows[::-1],[17,29,43])
