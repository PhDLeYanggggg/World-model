import pytest
from scripts.build_m3w_evidence_manuscript_v7 import insert_before, manuscript
from scripts.report_m3w_feature_cost_evidence import STUDIES


def test_revision_insertion_rejects_ambiguous_section_boundaries():
    assert insert_before('a\n## section', '## section', 'new') == 'a\nnew\n\n## section'
    with pytest.raises(ValueError):
        insert_before('## section\n## section', '## section', 'new')


def test_new_draft_retains_negative_result_and_anonymous_scope():
    policies = [dict(study=study, policy=arm, scope='inner_held',
        known_ADE_gain_floor_percent=1., supported_record_head_views=23)
        for study in STUDIES[-2:] for arm in ('physical_only', 'physical_neural')]
    value = dict(policies=policies, inner_held_mixed_minus_physical={STUDIES[-1]:dict(
        full_known_pp=dict(point=-.1, CI95=[-.2, -.01]))}, calibration_MSE_new_minus_original={STUDIES[-1]:dict(
        inner_held=dict(physical_neural=dict(harm=dict(point=-.1, CI95=[-.2, -.05]))))})
    original = '\n'.join(('Anonymous research draft.', '## 1. Introduction', '## 6. Experimental Protocol',
        '## 8. Failure Analysis and Limitations', '## 9. Reproducibility and Responsible Reporting', '## References'))
    text = manuscript(value, original)
    assert 'not robust positive utility' in text
    assert text.startswith('Anonymous research draft.')
    assert 'Future positions and label availability never' in text
    assert 'three controller seeds' in text
    assert '-0.100000 points' in text
    assert 'No independent outcomes are opened' in text
    assert 'not a theorem' in text
    assert '## 8. Failure Analysis and Limitations' in text
