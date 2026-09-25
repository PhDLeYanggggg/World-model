import copy
import json
from pathlib import Path
import pytest
from scripts.run_m3w_european_floor_relative import validate


def test_all_registered_families_and_no_reserved_roles():
    cfg = json.loads(Path('configs/m3w_european_floor_relative_v1.json').read_text())
    validate(cfg)
    for key, value in [('reserved_roles_opened', True), ('risk_budget', .05), ('modes', ['batch']),
                       ('references', ['floor']), ('total_new_heads', 72), ('variants', {'floor_both': ['floor', 'floor']})]:
        bad = copy.deepcopy(cfg); bad[key] = value
        with pytest.raises(ValueError): validate(bad)


def test_registered_fitting_budget_accounting():
    cfg = json.loads(Path('configs/m3w_european_floor_relative_v1.json').read_text())
    heads = cfg['shared_inner_utility_heads']+len(cfg['modes'])*(
        cfg['inner_risk_heads_per_mode']+cfg['main_heads_per_mode'])
    assert heads == cfg['total_new_heads'] == 234
    assert heads*cfg['head_training']['steps'] == cfg['total_updates'] == 468000
