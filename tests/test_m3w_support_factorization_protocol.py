import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import run_m3w_european_support_factorization as run


def test_fixed_matrix_and_no_new_threshold_fit():
    cfg = json.loads((run.ROOT/run.CONFIG).read_text()); run.validate(cfg)
    assert 2*3*3*2*2*13 == cfg['views']
    for key, value in [('minimum_source_count', 1), ('support_quantiles', [.05, .95]),
                       ('threshold_refit', True), ('producer_retraining', True), ('reserved_roles_opened', True)]:
        bad = copy.deepcopy(cfg); bad[key] = value
        with pytest.raises(ValueError): run.validate(bad)


def test_decisions_do_not_use_labels():
    tree = ast.parse(Path(run.__file__).read_text())
    decide = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decide')
    text = ast.unparse(decide)
    assert 'target_eval' not in text and 'native_errors' not in text and 'partition_ledger' not in text
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    text = ast.unparse(evaluate)
    assert text.index('ensure_both_frozen()') < text.index('independent_verify(') < text.index('native_errors(')


def test_all_preceding_anchor_controls_retained():
    assert run.OLD_NAMES == dict(stop='stop', joint='combined', joint_risk='combined_risk', joint_random='combined_random')
    assert run.PRIVATE != run.prior.PRIVATE and run.PUBLIC != run.prior.PUBLIC


def test_registered_sources_are_complete():
    assert len(run.FILES) == 6
    for file in run.FILES:
        assert (run.ROOT/file).is_file()
