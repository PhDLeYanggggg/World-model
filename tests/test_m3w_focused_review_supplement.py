from hashlib import sha256
import json

import pytest

from reproducibility.focused_review_supplement import moving_completion, verify, wepdtof_completion


def test_moving_extension_cannot_change_the_certified_view_roster():
    key = dict(recording=0, source=0, seed=17)
    base = dict(views=[dict(**key, queries=[])])
    extension = dict(views=[dict(**key, completion_queries=[])])
    assert moving_completion.attach(base, extension) == [dict(**key, queries=[], completion_queries=[])]
    extension['views'][0]['seed'] = 29
    with pytest.raises(ValueError, match='do not align'):
        moving_completion.attach(base, extension)


def test_undefined_source_is_excluded_not_filled_with_zero():
    arm = 'fixed_independent'
    group = lambda observed, lower: [[{'paired': {arm: [observed, lower]}}]]
    assert wepdtof_completion.group_bounds({'groups': [group(1., 0.), group(None, None)]}, arm) == [(0., 1., 2.)]


def test_top_level_manifest_detects_changed_file(tmp_path):
    payload = b'unchanged'
    hashes = {name: sha256(payload).hexdigest() for name in verify.MEMBERS}
    for name in hashes:
        (tmp_path/name).write_bytes(payload)
    (tmp_path/'manifest.json').write_text(json.dumps(hashes))
    verify.verify_members(tmp_path)
    (tmp_path/'moving_completion.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from the manifest'):
        verify.verify_members(tmp_path)
