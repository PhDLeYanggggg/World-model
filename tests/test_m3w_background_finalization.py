from types import SimpleNamespace

import pytest

from src.evaluation.m3w_background_finalization import summarize_roles_once


def module_with_duplicated_role(role=False):
    module=SimpleNamespace(conservative_exclusions=lambda *args:dict(
        scientific_roles_assigned=role,conservative_exclusion_components=[['moving_a','drone_b']],
        negative_matching_certifies_independence=False))

    def summarize(*args):
        return dict(**module.conservative_exclusions(*args),scientific_roles_assigned=False,clip_pairs=1)

    module.summarize=summarize
    return module


def test_preserved_failure_reproduces_and_repair_only_deduplicates_false_role():
    module=module_with_duplicated_role();original=module.conservative_exclusions
    with pytest.raises(TypeError,match='scientific_roles_assigned'):
        module.summarize()
    value=summarize_roles_once(module)
    assert value['scientific_roles_assigned'] is False
    assert value['conservative_exclusion_components']==[['moving_a','drone_b']]
    assert value['negative_matching_certifies_independence'] is False
    assert module.conservative_exclusions is original


def test_repair_never_changes_true_role_to_false_and_restores_original_on_error():
    module=module_with_duplicated_role(True);original=module.conservative_exclusions
    with pytest.raises(ValueError,match='cannot grant'):
        summarize_roles_once(module)
    assert module.conservative_exclusions is original
