import copy
import json
from pathlib import Path
import pytest
from scripts.run_m3w_european_producer_conditioned import CONFIG, validate


def test_registered_budget_and_controls():
    cfg = json.loads(Path(CONFIG).read_text()); validate(cfg)
    assert cfg['new_heads'] == 3*3*2*3*2
    assert cfg['views'] == 3*3*2*2*5
    assert cfg['updates'] == cfg['new_heads']*cfg['head_training']['steps']


@pytest.mark.parametrize('key,value', [('reserved_roles_opened', True), ('threshold_refit', True), ('mode', 'batch'), ('arms', ['producer']), ('risk_budget', .1)])
def test_no_silent_scope_or_budget_change(key, value):
    cfg = json.loads(Path(CONFIG).read_text()); changed = copy.deepcopy(cfg); changed[key] = value
    with pytest.raises(ValueError): validate(changed)
