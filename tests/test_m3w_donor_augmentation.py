import numpy as np
import pytest

from src.world_model import m3w_donor_augmentation as m


def data():
    own = dict(x=np.arange(720).reshape(6, 120).astype(float), target=np.ones((6, 2)),
        sites=np.array(['a', 'a', 'a', 'b', 'b', 'b']), ids=np.arange(6), weights=np.ones(6)/6)
    added = dict(x=np.ones((4, 120)), target=np.ones((4, 2)), sites=np.array(['c']*2+['d']*2),
                 ids=np.arange(10, 14)*7+1, weights=np.ones(4)/4)
    return own, added


@pytest.mark.parametrize('arm,old_mass,new_mass', [('own_only', 1, 0), ('A_only', 0, 1), ('augmented', .5, .5)])
def test_exact_pool_mass_and_exclusions(arm, old_mass, new_mass):
    own, added = data()
    d = m.combined(own, added, 'a', arm)
    p = m.parent.prepare(d, 'a')
    assert p['binding']['held_source'] == 'a'
    assert 'a' not in p['binding']['train_sources']
    assert np.isclose(d['weights'][d['sites']=='b'].sum(), old_mass)
    assert np.isclose(d['weights'][np.isin(d['sites'], ['c', 'd'])].sum(), new_mass)
    assert np.all(d['weights'][d['sites']=='a']==0)


def test_reject_source_overlap():
    own, added = data()
    added['sites'][0] = 'a'
    with pytest.raises(ValueError):
        m.combined(own, added, 'a', 'augmented')


def test_held_future_does_not_change_preprocessing():
    own, added = data()
    first = m.parent.prepare(m.combined(own, added, 'a', 'augmented'), 'a')
    own['target'][:3] = 1e8
    second = m.parent.prepare(m.combined(own, added, 'a', 'augmented'), 'a')
    m.parent.storage.base.exact(first, second)


def test_resume_exact(tmp_path):
    own, added = data()
    d = m.combined(own, added, 'a', 'augmented')
    m.fit(d, 'a', {'test': True}, tmp_path/'r.pt.gz', stop_at=2)
    a = m.fit(d, 'a', {'test': True}, tmp_path/'r.pt.gz', stop_at=4, resume=True)
    b = m.fit(d, 'a', {'test': True}, tmp_path/'d.pt.gz', stop_at=4)
    m.exact(a, b)
