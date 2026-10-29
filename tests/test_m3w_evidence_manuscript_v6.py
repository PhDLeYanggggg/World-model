from scripts.build_m3w_evidence_manuscript_v6 import manuscript, STUDIES


def test_manuscript_keeps_failed_support_and_identity_boundaries():
    rows = []
    for study in STUDIES:
        for policy in ('physical_only', 'physical_neural'):
            rows.append(dict(study=study, scope='inner_held', policy=policy,
                known_ADE_gain_floor_percent=1., supported_record_head_views=0,
                required_record_head_views=186, full_record_roster_supported=False))
    text = manuscript({'policies':rows})
    assert text.count('|0/186|') == 0
    assert text.count('0/186') == 4
    assert 'does not establish a deployable safe learned upgrade' in text
    assert 'LLM Usage Statement' in text
    assert 'not limited to grammar correction' in text
    assert 'whole-pipeline out-of-fold evidence' in text
    assert 'No new JEPA contribution' in text
    assert '~Le_Yang' not in text
