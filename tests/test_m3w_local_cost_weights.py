import json
import numpy as np
from src.evaluation import m3w_local_cost_analogue as api
from tests.test_m3w_local_cost_analogue import inputs


def test_equal_source_weighting_and_source_specific_unit_change():
    source=np.array(['q']*10+['a']*20+['b']*40)
    env=np.array([2.]*10+[1.]*20+[3.]*40)
    f=np.array([.1]*10+[.2]*20+[.8]*40)
    labels=np.column_stack((np.zeros(70),env*f,env,env,env*f/2))
    bank=dict(x=np.zeros((70,4)),envelope=env,labels=labels,scale=env.copy(),sources=source,
        recordings=np.array([json.dumps([s,'r'],separators=(',',':')) for s in source]),eligible=np.ones(70,bool))
    q=np.ones(70,bool);frozen=labels.copy()
    out=api.retrieve(bank,q,frozen,k=60)
    np.testing.assert_allclose(out['predictions']['selected_nearest32'][:10,0],1.2)
    np.testing.assert_allclose(out['predictions']['local_signed32'][:10,0],1.)
    assert out['supported'][:10].all() and not out['supported'][10:].any()
    ref=next(r for r in out['references'] if r['query_source']=='q')
    assert ref['known_donors_by_source']=={'a':20,'b':40,'q':0}
    for key in ('envelope','scale'):bank[key][source=='b']*=11
    bank['labels'][source=='b']*=11;frozen[source=='b']*=11
    scaled=api.retrieve(bank,q,frozen,k=60)
    np.testing.assert_allclose(scaled['predictions']['local_signed32'][:10],out['predictions']['local_signed32'][:10])


def test_reference_offset_sign_and_bounded_solution_are_exact():
    bank,q,p=inputs()
    bank['labels'][:,1]=.2;bank['labels'][:,4]=.1
    p[:,2]=bank['labels'][:,2]+10
    p[:,3]=bank['labels'][:,3]-10
    out=api.retrieve(bank,q,p,k=4)
    np.testing.assert_allclose(out['predictions']['local_signed32'][:,0],.4)
    np.testing.assert_allclose(out['predictions']['local_signed32'][:,1],0.)
    np.testing.assert_allclose(out['predictions']['weighted_harm32'],np.tile([.2,.1],(30,1)))
