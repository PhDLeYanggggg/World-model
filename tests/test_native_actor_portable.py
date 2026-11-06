import ast
import copy
import json
from pathlib import Path

import pytest

from scripts.build_m3w_native_actor_reproduction import make_files
from src.evaluation import native_actor_portable as portable


@pytest.fixture(scope='module')
def files():
    return make_files()


def test_all_three_original_primaries_and_original_seed_difference_preserved(files):
    evidence = json.loads(files['evidence.json'])
    assert [e['bootstrap_seed'] for e in evidence['experiments']] == [39017, 39107, 39107]
    result = portable.reconstruct(evidence)
    assert len(result['experiments']) == 3
    assert all(not e['component_screen_pass'] for e in result['experiments'])
    assert result['source_groups'] == 14 and result['rows_per_experiment'] == 125540
    assert result['independent_confirmation'] is False
    assert all(b'moving_source_component_' not in value for value in files.values())


def test_missing_source_value_is_not_dropped_or_zero_filled(files):
    evidence = json.loads(files['evidence.json'])
    evidence['experiments'][0]['sources'][0]['prefixes'][0]['known_pp'] = None
    with pytest.raises(ValueError, match='Undefined'):
        portable.reconstruct(evidence)


def test_changing_registered_bootstrap_seed_is_detected(files):
    evidence = json.loads(files['evidence.json'])
    evidence['experiments'][0]['bootstrap_seed'] = 39107
    with pytest.raises(ValueError, match='Scalar reconstruction'):
        portable.reconstruct(evidence)


def test_missing_population_and_changed_negative_screen_are_rejected(files):
    evidence = json.loads(files['evidence.json'])
    reduced = copy.deepcopy(evidence); reduced['experiments'][0]['sources'].pop()
    with pytest.raises(ValueError, match='complete source population'):
        portable.reconstruct(reduced)
    evidence['experiments'][0]['archived_primary']['component_progression_screen_pass'] = True
    with pytest.raises(ValueError, match='negative screen'):
        portable.reconstruct(evidence)


def test_portable_module_imports_only_standard_library_and_exports_are_deterministic(files):
    tree = ast.parse(files['reproduce.py'])
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports == {'argparse', 'csv', 'hashlib', 'io', 'json', 'math', 'pathlib'}
    assert files == make_files()
    for name in files:
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
