import json
import numpy as np
import pytest
from scripts.train_m3w_easy_hurdle_portable import unpack, array_hash
from src.world_model import m3w_easy_hurdle as api


def packet(path, held=False):
    n=4
    ident=dict(seed=17,roles=dict(training_sites=['a','b'],held_sites=['c','d']))
    fields=dict(x=np.zeros((n,4)),u=np.zeros((n,6)),y=np.zeros((n,3)),
        env=np.ones(n),ids=np.arange(n),sites=np.array(['a','a','b','c' if held else 'b']),
        recordings=np.repeat('r',n),frames=np.arange(n),pr_known=np.ones(n,bool),
        pr_weights=np.ones(n)/n,pr_mean=np.zeros(4),pr_std=np.ones(4),
        identity_json=np.array(json.dumps(ident)),
        preprocess_json=np.array(json.dumps(dict(clip=5.,cost_scale=1.,training_sites=['a','b']))))
    fields['y']=np.array([[1.,.1,0.],[0.,.3,.1],[1.,.2,.03],[0.,.2,0.]])
    ident['fitting_ids_hash']=array_hash(fields['ids'])
    ident['input_hashes']={k:array_hash(fields[v]) for k,v in [('x','x'),('descriptors','u'),('envelope','env'),('targets','y')]}
    fields['identity_json']=np.array(json.dumps(ident))
    np.savez_compressed(path,**fields)


def test_fitting_packet_uses_no_pickle_and_preserves_values(tmp_path):
    p=tmp_path/'fit.npz';packet(p)
    a,pr,ident=unpack(p)
    assert set(a['sites'])=={'a','b'} and ident['seed']==17
    np.testing.assert_array_equal(pr['weights'],np.ones(4)/4)
    assert 'future_endpoint' not in a


def test_held_rows_are_rejected(tmp_path):
    p=tmp_path/'bad.npz';packet(p,held=True)
    with pytest.raises(ValueError,match='non-fitting'):
        unpack(p)


def test_packet_and_direct_training_match(tmp_path):
    p=tmp_path/'fit.npz';packet(p)
    a,pr,ident=unpack(p)
    args=[a[k] for k in ('x','u','y','sites','recordings','frames','env')]+[pr]
    settings=dict(width=8,steps=4,query_batch_size=4,learning_rate=.001,gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    kw=dict(arm='supervised',seed=17,settings=settings,identity=ident,heartbeat=lambda **kw:None)
    first=api.fit(*args,path=tmp_path/'direct.pt.gz',**kw)
    a2,pr2,_=unpack(p)
    second=api.fit(*[a2[k] for k in ('x','u','y','sites','recordings','frames','env')],pr2,
                   path=tmp_path/'packet.pt.gz',**kw)
    for k in first:
        if k!='seconds':
            api.sampling.exact(first[k],second[k])


def test_packet_mutation_is_detected(tmp_path):
    p=tmp_path/'fit.npz';packet(p)
    with np.load(p,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    a['x'][0,0]=1
    np.savez_compressed(p,**a)
    with pytest.raises(AssertionError):
        unpack(p)
