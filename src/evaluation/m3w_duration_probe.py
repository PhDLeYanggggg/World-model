"""A fixed fourfold budget check, not new capacity or a changed loss."""
from copy import deepcopy

from src.world_model.m3w_multiaction_cost import exact

PREFIX_FIELDS = ('model', 'optimizer', 'sampler', 'torch_rng', 'preprocess',
    'fit_ids', 'initial_model', 'draw_hash', 'seed', 'source_sites', 'parameters',
    'step', 'observed_supervised_draws', 'unknown_bound_supervised_draws')


def extended_settings(settings, factor=4):
    if factor != 4:
        raise ValueError('One fixed fourfold duration test, not a budget sweep')
    result = deepcopy(settings)
    for key in ('minimum_updates', 'largest_source_cycles'):
        result[key] *= factor
    return result


def require_exact_prefix(original, prefix):
    for key in PREFIX_FIELDS:
        exact(original[key], prefix[key])
    if prefix['budget'] != 4*original['budget']:
        raise ValueError('Only the registered fourfold budget extension')
