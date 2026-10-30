"""One-candidate bank replacement with frozen physical preprocessing and draws."""
import hashlib
import numpy as np
import torch

from src.world_model import m3w_completion_cost as completion
from src.world_model import m3w_detector_cost as detector
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_source_forest import fingerprint
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask

VARIANTS = ('physical_only', 'old_neural', 'new_neural')
BANKS = dict(physical_only='physical_only', old_neural='physical_neural', new_neural='physical_neural')


class ReplacedActionArray:
    """Keep the six unchanged action columns shared, materializing row chunks."""
    def __init__(self, base, replacement):
        if (len(base.shape) != 3 or base.shape[1] != 7
                or replacement.shape != (base.shape[0], base.shape[2])
                or replacement.dtype != base.dtype or not np.isfinite(replacement).all()):
            raise ValueError('Same-dtype aligned seventh-action replacement required')
        self.base, self.replacement = base, replacement
        self.shape, self.dtype, self.ndim = tuple(base.shape), base.dtype, 3

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        value = self.base[rows].copy()
        if value.ndim != 3:
            raise ValueError('Explicit action-bank row batches required')
        value[:, 6] = self.replacement[rows]
        return value

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('A replaced action array requires materialization')
        value = self[:]
        return value.astype(dtype, copy=False) if dtype is not None else value


def concatenated_fingerprint(arrays, chunk=1024):
    if not arrays or any(v.shape[1:] != arrays[0].shape[1:] or v.dtype != arrays[0].dtype for v in arrays):
        raise ValueError('Homogeneous row-concatenated bank arrays required')
    shape = (sum(len(v) for v in arrays), *arrays[0].shape[1:])
    state = hashlib.sha256(str((shape, arrays[0].dtype.str)).encode())
    for value in arrays:
        for start in range(0, len(value), chunk):
            state.update(np.ascontiguousarray(value[start:start + chunk]).tobytes())
    return state.hexdigest()


def causal_neural_fields(*, geometry, cv, floor, candidate, floor_switch, scene):
    """No target, eventual future support or future endpoint is accepted here."""
    if scene.shape != (len(candidate), 18) or not np.isfinite(scene).all():
        raise ValueError('Past/static/predicted scene context required')
    x, envelope = matched_features(geometry, cv, floor, candidate, floor_switch)
    return dict(x=x, envelope=envelope, rollout=np.asarray(candidate), scene=np.asarray(scene, np.float32))


def replace_packet(original, fields, *, causal_eligible, labels, ade, fde):
    """Attach supervision separately; never reuse old neural costs for a new path."""
    if set(fields) != {'x', 'envelope', 'rollout', 'scene'}:
        raise ValueError('Only the declared causal replacement fields are admitted')
    n = len(original['ids'])
    expected = dict(x=(n, 380), envelope=(n,), rollout=(n, 12, 2), scene=(n, 18))
    if any(fields[k].shape != shape or not np.isfinite(fields[k]).all() for k, shape in expected.items()):
        raise ValueError('Complete finite aligned causal seventh action required')
    if causal_eligible.shape != (n,) or causal_eligible.dtype != bool or labels.shape != (n, 5):
        raise ValueError('Frozen causal cohort and separate five-moment labels required')
    known = np.isfinite(original['floor_ade'])
    if (ade.shape != (n,) or fde.shape != (n,) or not np.array_equal(np.isfinite(ade), known)
            or not np.array_equal(np.isfinite(labels).all(1), known)
            or not np.array_equal(np.isfinite(fde), np.isfinite(original['floor_fde']))
            or not np.isnan(labels[~known]).all() or (labels[known] < 0).any()):
        raise ValueError('Original future-label support must remain unchanged')
    np.testing.assert_array_equal(labels[:, [2, 3]], original['y'][:, 0][:, [2, 3]])
    output = {k: v for k, v in original.items() if k != 'args'}
    output['x'] = ReplacedActionArray(original['x'], fields['x'])
    output['scene'] = ReplacedActionArray(original['scene'], fields['scene'])
    for key, replacement in (('envelope', fields['envelope']), ('rollouts', fields['rollout']),
                              ('y', labels), ('ade', ade), ('fde', fde)):
        output[key] = original[key].copy()
        output[key][:, 6] = replacement
    output['allowed'] = original['allowed'].copy()
    output['allowed'][:, 6] = causal_eligible & real_intervention_mask(fields['rollout'], original['floor_rollout'])
    output.update(source_packet_features_labels_replayed=False, original_physical_columns_preserved=True,
                  original_causal_cohort_preserved=True)
    return output


class ReplacedNormalizedFeatures:
    def __init__(self, original, replacement):
        if (original.shape[1:] != (7, 462) or replacement.shape != (original.shape[0], 398)
                or not torch.isfinite(replacement).all()):
            raise ValueError('398 replacement fields; unchanged64detector fields required')
        self.original, self.replacement, self.shape = original, replacement, original.shape

    def __getitem__(self, rows):
        value = self.original[rows].clone()
        value[:, 6, :398] = self.replacement[rows]
        return value


def frozen_training_data(original, old_packets, new_packets):
    """The changed proposal must not refit normalization or alter the draw pool."""
    names = sorted(old_packets)
    if set(names) != set(new_packets) or original['z'].shape[1:] != (7, 462):
        raise ValueError('Same original fitting sources and full history cost schema required')
    for name in names:
        left, right = old_packets[name], new_packets[name]
        for key in ('ids', 'sites', 'recordings', 'frames', 'scales', 'floor_rollout', 'floor_ade', 'floor_fde', 'cv_ade', 'hard'):
            np.testing.assert_array_equal(left[key], right[key])
        for key in ('x', 'scene'):
            if not isinstance(right[key], ReplacedActionArray) or right[key].base is not left[key]:
                raise ValueError('Exactly six shared original physical columns required')
        for key in ('envelope', 'y', 'ade', 'fde', 'allowed', 'rollouts'):
            np.testing.assert_array_equal(left[key][:, :6], right[key][:, :6])
    fields = {key: np.concatenate([new_packets[n][key] for n in names])
              for key in ('y', 'envelope', 'scales', 'allowed')}
    bounds, known = completion.completion_endpoints(fields['y'], fields['envelope'])
    np.testing.assert_array_equal(known, original['known'].numpy())
    pr = original['preprocess']
    raw = np.concatenate([np.column_stack((new_packets[n]['x'].replacement,
                                          new_packets[n]['scene'].replacement)) for n in names])
    z = np.clip((raw - pr['mean']) / pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
    data = dict(original, z=ReplacedNormalizedFeatures(original['z'], torch.from_numpy(z)),
                envelope=torch.from_numpy((fields['envelope'] / fields['scales'][:, None]).astype(np.float32)),
                truth=torch.from_numpy(np.where(known[:, None, None],
                    fields['y'] / fields['scales'][:, None, None], 0.).astype(np.float32)),
                bound_truth=torch.from_numpy((bounds / fields['scales'][:, None, None]).astype(np.float32)),
                allowed=torch.from_numpy(fields['allowed']), input_hashes=dict(original['input_hashes']))
    for key in ('x', 'envelope', 'y', 'allowed'):
        data['input_hashes'][key] = concatenated_fingerprint([new_packets[n][key] for n in names])
    data['input_hashes']['scene_causal_features'] = concatenated_fingerprint([new_packets[n]['scene'] for n in names])
    data['input_hashes']['completion_endpoints'] = fingerprint(bounds)
    # Original sampler groups include unknown futures and stay identical across banks.
    if data['groups'] is not original['groups'] or data['preprocess'] is not pr:
        raise AssertionError('Original sampling and preprocessing must be reused exactly')
    return data


def fit(data, *, variant, settings, seed, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if variant not in VARIANTS:
        raise ValueError('Fixed matched bank variant required')
    with detector.installed('history'):
        return completion.fit(data, mode='known_only', bank=BANKS[variant], settings=settings,
            seed=seed, identity=identity, path=path, heartbeat=heartbeat, guard=guard, resume=resume, stop_at=stop_at)


def assert_scientific_equal(left, right):
    if set(left) != set(right):
        raise ValueError('Same scientific checkpoint schema required')
    for key in left:
        if key not in ('identity', 'seconds'):
            completion.base.exact(left[key], right[key])


def assert_common_design(left, right):
    for key in ('settings', 'seed', 'initial_model', 'preprocess', 'draw_hash', 'row_draws',
                'sampler_rng', 'torch_rng', 'parameters', 'draw_pool', 'unknown_rows_drawn'):
        completion.base.exact(left[key], right[key])
    if left['unknown_rows_with_loss'] or right['unknown_rows_with_loss']:
        raise AssertionError('Unknown futures cannot be treated as observed training costs')
