import numpy as np
import pytest
from scripts import run_m3w_moving_role_separated_cost as run


def bank(ids,sites):
    n=len(ids)
    return dict(x=np.ones((n,120)),moments=np.ones((n,5)),observed=np.ones(n,bool),
                envelope=np.ones(n),sites=np.array(sites),ids=np.array(ids),weights=np.ones(n)/n)


def test_full_candidate_ids_not_encoded_twice_and_pool_mass_is_fixed():
    own=bank([71,72,78,79],['a','a','b','b']);added=bank([701,702],['c','c'])
    d=run.mixed(own,added,'a')
    np.testing.assert_array_equal(d['ids'],[71,72,78,79,701,702])
    assert (d['weights'][d['sites']=='a']==0).all()
    assert d['weights'][d['sites']=='b'].sum()==.5
    assert d['weights'][d['sites']=='c'].sum()==.5
    added['sites'][0]='a'
    with pytest.raises(ValueError):run.mixed(own,added,'a')


def test_primary_bootstrap_requires_all_blocks_and_averages_risk_seeds():
    cfg=dict(bootstrap_seed=62468)
    rows=[dict(source=str(i),effect=float(s)) for i in range(8) for s in (-1,0,1)]
    result=run.interval(rows,'effect',cfg)
    assert result['point']==0 and result['interval95']==[0,0]
    assert result['risk_seeds_averaged_before_resampling'] and not result['independent_confirmation']
    rows[0]['effect']=None
    assert run.interval(rows,'effect',cfg)['point'] is None


def test_repeated_record_views_are_not_counted_as_independent_sources():
    rows=[dict(source=str(i),effect=1.) for i in range(7) for _ in range(3)]
    with pytest.raises(ValueError):run.interval(rows,'effect',dict(bootstrap_seed=62468))
