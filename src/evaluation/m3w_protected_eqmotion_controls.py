"""Strictly matched cached-head budgets for the full-EqMotion control extension."""
import numpy as np


def check_reused_head(cp, receipt, *, settings, seed, input_hash, label_hash, pr, reference_draws, held):
    if (cp['identity'] != receipt['identity'] or not receipt['fit']['complete']
            or cp['settings'] != settings or cp['seed'] != seed
            or cp['step'] != settings['steps'] or cp['arm'] != 'bounded_fraction'
            or cp['identity']['inputs_sha256'] != input_hash
            or cp['identity']['labels_sha256'] != label_hash
            or held in cp['preprocess']['training_sites']):
        raise ValueError('Changed, exposed or incomplete cached neural head')
    for key in ('mean', 'std', 'known', 'weights', 'constant'):
        np.testing.assert_array_equal(cp['preprocess'][key], pr[key])
    if cp['preprocess']['cost_scale'] != pr['cost_scale']:
        raise ValueError('Changed training-only cost scale')
    np.testing.assert_array_equal(cp['draws'], reference_draws)
    if (cp['draws'].sum() != settings['steps']*settings['batch_size']
            or cp['draws'][~pr['known']].any()):
        raise ValueError('Unmatched fitting budget or unknown-label sampling')


def verify_same_population(left, right):
    if set(left) & {'target', 'valid', 'future_endpoint'} or set(right) & {'target', 'valid', 'future_endpoint'}:
        raise ValueError('Past-only population loaders required')
    for key in ('geometry', 'scale', 'sites', 'recordings'):
        np.testing.assert_array_equal(left[key], right[key])
