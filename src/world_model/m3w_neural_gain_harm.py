"""Fit-only coherent neural cost regression on the same OOF inputs as ridge.

Squared loss estimates benefit/harm magnitudes, not an oracle class or a
calibrated failure probability. No real protocol or safety budget is supplied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Refusing Rosetta before Torch import')

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_oof_identity import oof_feature_identity
from src.world_model.m3w_supervised_intervention import GainHarmHead, load_verified_forecaster


def training_spec(contract):
    contract._assert_frozen()
    spec = contract.protocol.get('gain_harm_training')
    if (not isinstance(spec, dict) or set(spec) != {'width', 'loss', 'fit_settings'}
            or type(spec['width']) is not int or spec['width'] < 1
            or spec['loss'] != 'squared_benefit_harm'):
        raise ValueError('Explicit protocol-bound neural gain/harm training specification required')
    settings = spec['fit_settings']
    if not isinstance(settings, dict) or set(settings) != {
            'steps', 'batch_size', 'learning_rate', 'checkpoint_every', 'heartbeat_every'}:
        raise ValueError('Explicit fixed-budget neural cost fit settings required')
    if any(type(settings[k]) is not int or settings[k] < 1 for k in settings if k != 'learning_rate'):
        raise ValueError('Positive integer fit settings required')
    rate = settings['learning_rate']
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
        raise ValueError('Finite positive learning rate required')
    return spec


class NeuralGainHarm(GainHarmHead):
    def __init__(self, mean, scale, *, width):
        mean, scale = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)
        if (mean.ndim != 1 or len(mean) == 0 or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Finite positive fit-only feature normalization required')
        super().__init__(len(mean), width=width)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not torch.isfinite(features).all():
            raise ValueError('Neural gain/harm feature schema mismatch')
        return super().forward((features - self.mean) / self.scale)


def regression_loss(scores, targets):
    prediction = torch.stack((scores['benefit'], scores['harm']), 1)
    if (targets.shape != prediction.shape or prediction.ndim != 2 or len(prediction) == 0
            or not torch.isfinite(targets).all() or (targets < 0).any()):
        raise ValueError('Finite nonnegative benefit/harm targets required')
    loss = (prediction - targets.detach()).square().mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite neural gain/harm loss')
    return loss


def validate_oof_groups(contract, groups, *, seed):
    contract._assert_frozen()
    if type(seed) is not int or seed not in contract.protocol['seeds'] or not groups:
        raise ValueError('Nonempty OOF supervision and protocol-listed seed required')
    folds = contract.protocol['fit_folds']
    seen_folds, parents, recordings, hashes, seen_rows = set(), set(), set(), [], set()
    architecture = None
    for group in groups:
        if (group['protocol_sha256'] != contract.digest or group['metric'] != contract.protocol['task']['primary_metric']
                or group['feature_source'] != 'past_and_frozen_rollouts_only'
                or group['target_source'] != 'held_fold_realized_relative_costs'):
            raise ValueError('Neural cost OOF provenance/schema mismatch')
        producer = group['predictor_id']
        if group['predictor_sha256'] != contract.artifacts[producer]['sha256']:
            raise ValueError('OOF predictor identity changed')
        x, y, rows = group['features'], group['targets'], group['identities']
        if (x.ndim != 2 or len(x) == 0 or x.shape[1] == 0 or y.shape != (len(x), 2)
                or len(rows) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0)):
            raise ValueError('Aligned finite OOF features and nonnegative relative costs required')
        used = {r['recording_id'] for r in rows}
        if any(r.get('data_role') != 'fit' or r['recording_id'] not in folds for r in rows):
            raise ValueError('Neural costs require fit-only OOF row identities')
        fold_ids = {folds[r] for r in used}
        if len(fold_ids) != 1 or seen_folds & fold_ids:
            raise ValueError('Each complete OOF fold must appear exactly once')
        held = sorted(r for r, f in folds.items() if f in fold_ids)
        if used != set(held):
            raise ValueError('OOF group must cover its complete fit fold')
        contract.assert_prediction_use(producer, held, purpose='oof_risk_training')
        for name in held:
            contract.open_recording(name, purpose='fit')
        model = load_verified_forecaster(contract, producer, device='cpu')
        if model._fitted_baseline_name != group['baseline_name']:
            raise ValueError('OOF cost baseline differs from producer feature baseline')
        state = torch.load(contract._path(contract.artifacts[producer]['path']), map_location='cpu', weights_only=True)
        if state['identity']['settings']['seed'] != seed:
            raise ValueError('Neural head and OOF producer seed mismatch')
        if architecture is not None and architecture != state['architecture']:
            raise ValueError('OOF producers must use the same forecaster architecture')
        architecture = state['architecture']
        for row in rows:
            key = tuple(row[k] for k in ('recording_id', 'agent_id', 'frame_id', 'horizon_raw'))
            if key in seen_rows:
                raise ValueError('Duplicated OOF query')
            seen_rows.add(key)
        digest = hashlib.sha256(json.dumps({k: v for k, v in group.items() if k not in {'features', 'targets'}},
                                          sort_keys=True, allow_nan=False).encode())
        for array in (x, y):
            digest.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        hashes.append(digest.hexdigest())
        seen_folds.update(fold_ids)
        parents.add(producer)
        recordings.update(used)
    if seen_folds != set(folds.values()) or len({g['baseline_name'] for g in groups}) != 1:
        raise ValueError('All fit folds and one common baseline required')
    return {'parents': sorted(parents), 'fit_recordings': sorted(recordings), 'group_sha256': hashes,
            'oof_feature_identity': oof_feature_identity(groups), 'baseline_name': groups[0]['baseline_name'],
            'metric': contract.protocol['task']['primary_metric'], 'producer_architecture': architecture}


def source_identity():
    return {name: file_digest(Path(__file__).with_name(name)) for name in
            ('m3w_neural_gain_harm.py', 'm3w_supervised_intervention.py', 'm3w_oof_identity.py')}


def train_neural_gain_harm(contract, groups, *, seed, output_dir, device='cpu', resume=False, stop_after=None):
    spec = training_spec(contract)
    provenance = validate_oof_groups(contract, groups, seed=seed)
    settings = spec['fit_settings']
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= settings['steps']):
        raise ValueError('stop_after must lie within the fixed training budget')
    x, y = (torch.as_tensor(np.concatenate([g[k] for g in groups]), dtype=torch.float32)
            for k in ('features', 'targets'))
    mean, scale = x.mean(0), x.std(0, correction=0).clamp_min(1e-6)
    output = Path(output_dir).resolve()
    if not output.is_relative_to(contract.root):
        raise ValueError('Output must remain inside experiment workspace')
    checkpoint = output / 'latest.pt'
    if checkpoint.exists() and not resume:
        raise ValueError('Checkpoint exists; use resume')
    if resume and not checkpoint.exists():
        raise ValueError('Cannot resume missing checkpoint')
    identity = {'protocol_sha256': contract.digest, **provenance, 'spec': spec, 'seed': seed,
                'source_identity': source_identity()}
    runtime = {'device': str(device), 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
               'compute_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
               'dataloader_workers': 0}
    torch.manual_seed(seed)
    model = NeuralGainHarm(mean, scale, width=spec['width']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    step, cursor, elapsed, losses, segments = 0, 0, 0., [], []
    if resume:
        receipt = output / 'fit_report.json'
        if receipt.exists():
            completed = json.loads(receipt.read_text())
            if completed.get('training_complete') and completed.get('checkpoint_sha256') != file_digest(checkpoint):
                raise ValueError('Completed neural cost checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Neural gain/harm resume identity changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        step, cursor, order = state['step'], state['cursor'], state['order']
        losses, elapsed, segments = state['losses'], state['elapsed_seconds'], state['runtime_segments']
    if step > settings['steps'] or (stop_after is not None and step < settings['steps'] and stop_after <= step):
        raise ValueError('Invalid resumed training budget')
    output.mkdir(parents=True, exist_ok=True)
    began, resumed_from = time.monotonic(), step
    if step < settings['steps']:
        segments = [*segments, {'start_step': step, **runtime}]

    def save():
        if str(device).startswith('mps'):
            torch.mps.synchronize()
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'identity': identity, 'runtime': runtime, 'runtime_segments': segments,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                    'cursor': cursor, 'order': order, 'sampler_rng': generator.get_state(),
                    'torch_rng': torch.get_rng_state(), 'losses': losses,
                    'elapsed_seconds': elapsed + time.monotonic() - began}, temporary)
        os.replace(temporary, checkpoint)

    if not resume:
        save()
    while step < settings['steps']:
        if cursor == len(order):
            order, cursor = torch.randperm(len(x), generator=generator), 0
        ids = order[cursor:cursor + settings['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss = regression_loss(model(x[ids].to(device)), y[ids].to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        step, cursor = step + 1, cursor + len(ids)
        if step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            with (output / 'heartbeat.jsonl').open('a') as stream:
                stream.write(json.dumps({'pid': os.getpid(), 'step': step, 'cost_mse': losses[-1],
                    'elapsed_seconds': elapsed + time.monotonic() - began}) + '\n')
        if step % settings['checkpoint_every'] == 0 or step == settings['steps'] or step == stop_after:
            save()
        if step == stop_after:
            break
    report = {'result_source': 'cached_verified' if resumed_from == settings['steps'] else 'fresh_run',
              'family': 'neural_gain_harm', 'scope': 'fit_only_OOF_relative_cost_regression',
              **provenance, 'protocol_sha256': contract.digest, 'seed': seed, 'spec': spec,
              'runtime': runtime, 'runtime_segments': segments, 'steps_completed': step,
              'resumed_from_step': resumed_from, 'training_complete': step == settings['steps'],
              'losses': losses, 'training_rows': len(x), 'feature_dimension': x.shape[1],
              'normalization_source': 'fit_OOF_rows_only', 'checkpoint': str(checkpoint),
              'checkpoint_sha256': file_digest(checkpoint), 'source_identity': identity['source_identity'],
              'code_sha256': identity['source_identity']['m3w_supervised_intervention.py'],
              'test_evaluated': False, 'calibrated_risk': False, 'deployment_approved': False}
    temporary = output / 'fit_report.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, output / 'fit_report.json')
    return report


def load_verified_neural_gain_harm(contract, artifact_id, *, device='cpu'):
    contract._assert_frozen()
    record = contract.artifacts[artifact_id]
    if record['kind'] != 'risk_head' or record.get('family') != 'neural_gain_harm':
        raise ValueError('Expected neural gain/harm risk-head artifact')
    contract._verify_artifact(record)
    state = torch.load(contract._path(record['path']), map_location='cpu', weights_only=True)
    identity = state['identity']
    if (identity['protocol_sha256'] != contract.digest or identity['spec'] != training_spec(contract)
            or identity['source_identity'] != source_identity() or identity['seed'] not in contract.protocol['seeds']):
        raise ValueError('Neural cost protocol/source identity changed')
    if (identity['parents'] != sorted(record['parents']) or identity['fit_recordings'] != sorted(record['fit_recordings'])
            or record['selection_recordings'] or record['calibration_recordings']):
        raise ValueError('Neural cost fit provenance changed')
    if state['step'] != identity['spec']['fit_settings']['steps']:
        raise ValueError('Neural cost fixed-budget checkpoint incomplete')
    model = NeuralGainHarm(state['model']['mean'], state['model']['scale'], width=identity['spec']['width']).to(device)
    model.load_state_dict(state['model'])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite neural cost weights')
    model.fitted_identity = identity
    return model.eval()


def read_verified_oof_cache(contract, directory):
    """Reuse the existing ridge CLI's complete fold caches, not a new extractor."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to(contract.root):
        raise ValueError('OOF cache must remain in experiment workspace')
    identity = json.loads((directory / 'run_identity.json').read_text())
    if (identity['protocol_sha256'] != contract.digest
            or identity['code_sha256'] != source_identity()['m3w_supervised_intervention.py']
            or identity['oof_identity_source_sha256'] != source_identity()['m3w_oof_identity.py']
            or identity['script_sha256'] != file_digest(Path(__file__).parents[2] / 'scripts/train_m3w_oof_cost_head.py')):
        raise ValueError('OOF cache protocol or feature code identity changed')
    folds = contract.protocol['fit_folds']
    if set(identity['fold_models']) != {str(f) for f in folds.values()}:
        raise ValueError('OOF cache missing fit folds')
    groups = []
    for fold, producer in sorted(identity['fold_models'].items()):
        if identity['predictor_sha256'][producer] != contract.artifacts[producer]['sha256']:
            raise ValueError('OOF cache producer changed')
        receipt = json.loads((directory / f'fold_{fold}.json').read_text())
        cache = directory / f'fold_{fold}.npz'
        if receipt['run_identity'] != identity or file_digest(cache) != receipt['cache_sha256']:
            raise ValueError('OOF cache bytes or receipt changed')
        with np.load(cache, allow_pickle=False) as arrays:
            group = {**receipt['group_metadata'], 'features': arrays['features'].copy(), 'targets': arrays['targets'].copy(),
                     'identities': json.loads(str(arrays['identities_json']))}
        if (group['predictor_id'] != producer or group['baseline_name'] != identity['baseline']
                or {r['recording_id'] for r in group['identities']} != {r for r, f in folds.items() if str(f) == fold}):
            raise ValueError('OOF cache fold identities changed')
        groups.append(group)
    return groups
