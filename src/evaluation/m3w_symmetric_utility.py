"""Single-factor guards for the source-only utility-loss experiment."""
import numpy as np


def validate_config(current, previous):
    if set(current) != set(previous):
        raise ValueError('Same registered configuration keys required')
    for key in current:
        if key not in ('scope', 'utility_head') and current[key] != previous[key]:
            raise ValueError('Simultaneous unregistered change: '+key)
    if current['utility_head'] != 'candidate_specific_neural_mse_same_budget':
        raise ValueError('Only symmetric utility loss is registered')
    if current['scope'] != 'source_development_single_factor_symmetric_utility':
        raise ValueError('Source-development scope required')


def verify_matched_checkpoint(new, old):
    if new['arm'] != 'mse' or old['arm'] != 'underharm4':
        raise ValueError('Expected paired utility objectives')
    for key in ('settings', 'seed', 'step'):
        if new[key] != old[key]:
            raise ValueError('Unmatched training budget: '+key)
    if new['step'] != new['settings']['steps']:
        raise ValueError('Incomplete training budget')
    for key in ('mean', 'std', 'constant', 'weights', 'known'):
        np.testing.assert_array_equal(new['preprocess'][key], old['preprocess'][key])
    if new['preprocess']['cost_scale'] != old['preprocess']['cost_scale']:
        raise ValueError('Changed native cost scale')
    for key in ('draws', 'sampler_rng'):
        np.testing.assert_array_equal(new[key], old[key])
    if np.any(new['draws'][~new['preprocess']['known']]):
        raise ValueError('Unknown-label rows entered training')
    if new['draws'].sum() != new['step'] * new['settings']['batch_size']:
        raise ValueError('Training draw accounting mismatch')


def complete_budget(heads, *, expected=18, steps=2000):
    if len(heads) != expected:
        raise ValueError('Complete fixed matrix required before readout')
    for receipt in heads.values():
        fit = receipt['fit']
        if not fit['complete'] or fit['step'] != steps or fit['unknown_rows_sampled'] != 0:
            raise ValueError('Incomplete or invalid utility fit')
    return expected * steps
