import numpy as np

from scripts.diagnose_m3w_coupled_local_shuffle import stratified_shuffle


def test_shuffle_preserves_joint_record_and_eligibility_marginals_and_unknowns():
    rng=np.random.default_rng(7);target=rng.uniform(0,1,(120,7,5));known=np.ones(120,bool);known[8]=False
    recordings=np.array(['a']*60+['b']*60);allowed=np.ones((120,7),bool);allowed[::2,1]=False
    shuffled,counts=stratified_shuffle(target,known,recordings,allowed)
    assert counts['changed_rows']>0
    np.testing.assert_array_equal(shuffled[~known],target[~known])
    for record in ('a','b'):
        for active in (True,False):
            take=known & (recordings==record) & (allowed[:,1]==active)
            np.testing.assert_allclose(shuffled[take].sum(0),target[take].sum(0))
    np.testing.assert_array_equal(shuffled,stratified_shuffle(target,known,recordings,allowed)[0])
