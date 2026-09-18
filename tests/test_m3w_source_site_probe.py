import numpy as np
import pytest

from scripts.run_m3w_source_site_probe import SiteCorpus, site_partition


def test_physical_site_holdout_groups_all_videos_and_scoped_tracks():
    sites = np.array(['a','a','b','c','c'])
    tracks = np.array(['a/v0:1','a/v1:1','b/v0:1','c/v0:1','c/v0:2'])
    train, held, weights = site_partition(sites,tracks,'a')
    np.testing.assert_array_equal(train,[2,3,4])
    np.testing.assert_array_equal(held,[0,1])
    np.testing.assert_allclose(weights,np.ones(3)/3)
    with pytest.raises(ValueError,match='leakage'):
        site_partition(sites,np.array(['same','x','same','c','d']),'a')
    with pytest.raises(ValueError):
        site_partition(sites,tracks,'missing')


def fake_data():
    data = object.__new__(SiteCorpus)
    data.nmain = 2
    data.source_sites = np.array(['a','a','b','b','c','c'])
    data.source_tracks = np.array(['a:1','a:2','b:1','b:2','c:1','c:2'])
    data.x = np.arange(24,dtype=np.float32).reshape(8,3)
    data.y = np.array([0,1,0,1,0,1,0,1])
    data.allowed = np.zeros(8,bool)
    return data


def test_normalization_does_not_use_main_or_held_site():
    data = fake_data(); train,w,held,_ = data.source_design('a')
    assert min(train) >= data.nmain
    np.testing.assert_array_equal(train,[4,5,6,7])
    original = {k:v.copy() for k,v in data.normalizer.items()}
    data.x[:4] = 1e8; data.source_design('a')
    for k,v in original.items():
        np.testing.assert_array_equal(v,data.normalizer[k])
    with pytest.raises(ValueError,match='Held'):
        data.loss_labels(held)
    with pytest.raises(ValueError,match='Held'):
        data.loss_labels(np.array([0]))
    np.testing.assert_array_equal(data.loss_labels(train).numpy(),[0,1,0,1])


def test_source_features_and_membership_independent_of_supervision_values():
    data = fake_data(); train,w,held,_ = data.source_design('b'); z = data.z.copy()
    data.y = 1-data.y
    train2,w2,held2,_ = data.source_design('b')
    np.testing.assert_array_equal(z,data.z)
    np.testing.assert_array_equal(train,train2)
    np.testing.assert_array_equal(held,held2)
    np.testing.assert_array_equal(w,w2)


def test_single_class_fold_is_not_silently_scored():
    data=fake_data(); data.y[2:4]=0
    with pytest.raises(ValueError,match='Both classes'):
        data.source_design('a')
