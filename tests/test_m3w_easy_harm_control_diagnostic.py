import copy
import inspect

import torch

from scripts import diagnose_m3w_easy_harm_control as diagnostic
from scripts.replay_m3w_dimensionless_training import exact


def state():
    out = {k: 1 for k in ('initial_model', 'preprocess', 'settings', 'input_hashes',
                         'seed', 'step', 'sampler_rng', 'torch_rng', 'draw_hash',
                         'row_draws', 'queries')}
    out.update(model={'w': torch.tensor([1., 2.])}, optimizer={'m': torch.tensor([0., 1.])},
               trace=[dict(step=1, monitor={'primary_total': .2})])
    return out


def test_records_small_numeric_mismatch_without_waiving_exact_control():
    a = state(); b = copy.deepcopy(a); b['model']['w'][0] += 1e-6
    out = diagnostic.compare(a, b, exact)
    assert not out['all_control_fields_exact']
    assert out['initial_model']['exact'] and not out['model']['exact']
    assert out['tensor_deltas'][0]['different'] == 1
    assert 0 < out['tensor_deltas'][0]['max_abs'] < 2e-6


def test_exact_control_and_structure_mismatch():
    a = state()
    assert diagnostic.compare(a, copy.deepcopy(a), exact)['all_control_fields_exact']
    b = copy.deepcopy(a); b['optimizer']['extra'] = 3
    out = diagnostic.compare(a, b, exact)
    assert not out['all_control_fields_exact']
    assert out['tensor_deltas'] == [dict(path='optimizer', structure_mismatch=True)]


def test_no_runtime_numeric_import_before_allocation_guard():
    source = inspect.getsource(diagnostic.main)
    assert source.index('SLURM_JOB_ID') < source.index('from src.world_model')
    assert "validation_scored=False" in source
    assert "acceptance_rule_relaxed=False" in source
