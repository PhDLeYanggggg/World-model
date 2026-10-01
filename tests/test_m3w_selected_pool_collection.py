import hashlib
import json

import pytest

from scripts.collect_m3w_selected_pool_create import validate_bundle, verify_existing


def fixture_bundle():
    files = {}; refs = []
    for i in range(216):
        name = 'view'+str(i)
        text = json.dumps(dict(rows=[dict(view=name, role='transfer', mode=m)
            for m in ('harm', 'reference', 'joint')], parent_risk_checks=12))
        files[name] = text
        refs.append(dict(group=name, bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest()))
    manifest = dict(packets=[dict(group=r['group']) for r in refs], local_parity_groups=144)
    complete = dict(job_id='123', transfer_views=216, exact_replay=True, parent_risk_checks=2592,
        local_parity_groups=144, new_parameter_updates=0, policy_changed=False, groups=refs)
    return dict(complete=complete, manifest=manifest, registration_sha256='registered', files=files,
        accounting=dict(returncode=0, stdout='123|COMPLETED|0:0|00:01|\n')), manifest


def test_complete_hash_bound_collection():
    bundle, manifest = fixture_bundle()
    assert validate_bundle(bundle, manifest, 'registered', '123') > 0


def test_local_parity_ignores_key_order_but_preserves_exact_numbers(tmp_path):
    p = tmp_path/'result.json'
    p.write_text('{"b": 2, "a": 1.25}')
    assert verify_existing(p, '{"a":1.25,"b":2}') == (True, False)
    with pytest.raises(AssertionError): verify_existing(p, '{"a":1.25000000001,"b":2}')
    assert verify_existing(tmp_path/'missing.json', '{}') == (False, False)


@pytest.mark.parametrize('mutation', ['failed_job', 'changed_output', 'wrong_registration', 'missing_group', 'no_replay'])
def test_refuse_unverified_collection(mutation):
    bundle, manifest = fixture_bundle()
    if mutation == 'failed_job': bundle['accounting']['stdout'] = '123|FAILED|1:0|00:01|\n'
    if mutation == 'changed_output': bundle['files']['view0'] += ' '
    if mutation == 'wrong_registration': bundle['registration_sha256'] = 'different'
    if mutation == 'missing_group': bundle['files'].pop('view0')
    if mutation == 'no_replay': bundle['complete']['exact_replay'] = False
    with pytest.raises(AssertionError): validate_bundle(bundle, manifest, 'registered', '123')
