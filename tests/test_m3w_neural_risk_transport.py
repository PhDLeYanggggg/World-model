import numpy as np
import torch
import pytest
from src.world_model import m3w_neural_risk_transport as m


def data():
    rng=np.random.default_rng(17); x=rng.normal(size=(120,120))
    return dict(x=x,target=x[:,:2]*.02,weights=np.ones(120),
        sites=np.array(['fit']*100+['held']*20),ids=np.arange(120))


def test_query_labels_cannot_change_training_data():
    d=data(); a=m.prepare(d,'held'); d['target'][100:]=1e7; b=m.prepare(d,'held')
    for k in ('x','y','w'): torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    m.storage.base.exact(a['binding'],b['binding'])


def test_transport_exact_affine_risk_and_no_query_labels():
    q=np.arange(6).reshape(3,2)*.1; d=np.ones((3,32,2))
    p=m.transport(q,d,d,np.ones((3,32)))
    np.testing.assert_array_equal(p,q)
    z=m.transport(q,d,d+.5,np.ones((3,32)))
    np.testing.assert_allclose(z,q+.5)
    with pytest.raises(ValueError): m.transport(q,d,d,np.zeros((3,32)))


def test_real_optimizer_resume_exact(tmp_path):
    torch.set_num_threads(2); d=data(); ident={'test':True}
    a=m.fit(d,'held',m.SETTINGS,ident,tmp_path/'a.pt.gz',stop_steps=4)
    m.fit(d,'held',m.SETTINGS,ident,tmp_path/'b.pt.gz',stop_steps=2)
    b=m.fit(d,'held',m.SETTINGS,ident,tmp_path/'b.pt.gz',resume=True,stop_steps=4)
    for k in ('model','optimizer','generator','torch_rng','epoch','cursor','order','step','visits','trace'):
        m.storage.base.exact(a[k],b[k])
    assert not a['complete']
    np.testing.assert_array_equal(m.predict(a,d['x']),m.predict(b,d['x']))
