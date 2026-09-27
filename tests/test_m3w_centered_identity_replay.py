import json
import pytest
from scripts.replay_m3w_centered_risk_policy import canonical_identity


def test_canonicalization_matches_persisted_contrasts_without_changing_values():
    a={'bindings':{'a':'hash'},'contrasts':[('new','old')],'groups':108,'threshold':.02}
    old=json.loads(json.dumps(a));new=canonical_identity(a)
    assert a!=old and new==old
    assert new['bindings']==a['bindings'] and new['threshold']==a['threshold']


def test_canonicalization_cannot_hide_different_hash_or_threshold():
    a={'binding':'a','threshold':.02,'contrasts':[('x','y')]}
    assert canonical_identity(a)!=canonical_identity(dict(a,threshold=.03))
    assert canonical_identity(a)!=canonical_identity(dict(a,binding='b'))


def test_nonfinite_identity_rejected():
    with pytest.raises(ValueError):canonical_identity({'x':float('nan')})
