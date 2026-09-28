import hashlib
import io
import json
import tarfile
import numpy as np
import pytest
from scripts.run_m3w_easy_risk_priority_policy import causal_view, action_group, gates_for, POLICIES, CONTRASTS, parent
from scripts.restore_m3w_easy_risk_priority_heads import install_bundle


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Future targets accessed during inference')


def test_causal_interface_strips_targets_and_rejects_extra_fields():
    data = {k: np.arange(3) for k in parent.CAUSAL_KEYS}
    data.update(future=Poison(), valid=Poison(), future_endpoint=Poison())
    causal = causal_view(data)
    assert set(causal) == set(parent.CAUSAL_KEYS)
    assert all(causal[k] is data[k] for k in causal)
    with pytest.raises(ValueError, match='whitelisted'):
        action_group({}, data, 0, None, None, None, None, {})


def test_registry_serialization_and_primary():
    value = dict(policies=POLICIES, contrasts=CONTRASTS)
    assert json.loads(json.dumps(value)) == value
    assert CONTRASTS[0] == ['risk_priority_matched','uncapped_matched']
    assert len(POLICIES) == 11


def passing_example():
    paired = {'risk_priority_matched_vs_uncapped_matched': {
        'ADE_gain_percent': {'ci95': [1., 2.]},
        'all_reference_harm_reduction_pp': {'ci95': [.1, .2]}}}
    rows = [dict(policy='risk_priority_matched', metric=dict(
        selected_positive_harm_ratio=.01, easy_gain_CV=0., zero_CV_harmed=0))]
    contrasts = [dict(policy='risk_priority_matched_vs_uncapped_matched',
                      metric=dict(intervention_difference_pp=0.))]
    return paired, rows, contrasts


def test_zero_denominator_abstention_does_not_pass_risk():
    paired, rows, contrasts = passing_example()
    assert gates_for(paired, rows, contrasts)['exploratory_screen_pass']
    rows[0]['metric']['selected_positive_harm_ratio'] = None
    gates = gates_for(paired, rows, contrasts)
    assert not gates['every_view_defined_risk_within_2percent']
    assert not gates['exploratory_screen_pass']
    assert not gates['independent_confirmation'] and not gates['deployment_changed']


@pytest.mark.parametrize('field,value', [('selected_positive_harm_ratio', .021),
                                       ('easy_gain_CV', -2.01), ('zero_CV_harmed', 1)])
def test_harm_is_not_hidden_by_positive_ade(field, value):
    paired, rows, contrasts = passing_example(); rows[0]['metric'][field] = value
    assert not gates_for(paired, rows, contrasts)['exploratory_screen_pass']


def test_count_mismatch_and_empty_readout_fail():
    paired, rows, contrasts = passing_example()
    contrasts[0]['metric']['intervention_difference_pp'] = 1
    assert not gates_for(paired, rows, contrasts)['matched_count']
    with pytest.raises(AssertionError):
        gates_for(paired, [], contrasts)


def bundle(name, payload=b'checkpoint', *, link=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:') as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE; member.linkname = '/tmp/forbidden'
            archive.addfile(member)
        else:
            member.size = len(payload); archive.addfile(member, io.BytesIO(payload))
    return buf.getvalue(), [dict(path=name, sha256=hashlib.sha256(payload).hexdigest())]


def test_checkpoint_restore_exact_and_idempotent(tmp_path):
    name = 'heads/g/uncapped/checkpoint.pt.gz'; content, refs = bundle(name)
    reserved = []
    assert install_bundle(content, refs, tmp_path, reserved.append) == 10
    assert reserved == [10] and (tmp_path/name).read_bytes() == b'checkpoint'
    install_bundle(content, refs, tmp_path, reserved.append)
    assert reserved == [10]
    bad = [dict(path=name, sha256='0'*64)]
    with pytest.raises(ValueError, match='checksum'):
        install_bundle(content, bad, tmp_path, reserved.append)


@pytest.mark.parametrize('name,link', [('../outside',False),
    ('heads/g/marginal/checkpoint.pt.gz',False), ('heads/g/uncapped/checkpoint.pt.gz',True)])
def test_no_unrelated_or_linked_archive_member(tmp_path, name, link):
    content, refs = bundle(name, link=link)
    with pytest.raises(ValueError, match='Unsafe'):
        install_bundle(content, refs, tmp_path, lambda n: None)


def test_existing_symlink_parent_cannot_escape(tmp_path):
    destination = tmp_path/'safe'; destination.mkdir()
    outside = tmp_path/'outside'; outside.mkdir()
    (destination/'heads').symlink_to(outside, target_is_directory=True)
    content, refs = bundle('heads/g/uncapped/checkpoint.pt.gz')
    with pytest.raises(ValueError, match='symlink'):
        install_bundle(content, refs, destination, lambda n: None)
