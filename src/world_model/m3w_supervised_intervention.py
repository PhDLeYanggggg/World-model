"""Clean causal forecaster and out-of-fold realized-cost supervision.

This is a deterministic training backend, not a calibrated/deployable policy.
It never opens calibration/confirmation data. The approved experiment contract
must be supplied by the caller; there is no implicit approval or legacy teacher.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use arm64 .venv-pytorch before importing Torch; Rosetta is refused')

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data_unification.m3w_causal_recordings import BASELINES, causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.world_model.m3w_joint_intervention import realized_relative_costs


INPUT_KEYS = {'history', 'history_mask', 'neighbors', 'neighbor_mask',
              'baseline', 'prediction_time', 'request_mask'}


def pack_inputs(inputs, baseline_name):
    """All times/coordinates use requested horizon and past-derived transforms."""
    offsets = np.asarray(inputs['prediction_frame_offsets'])
    if not len(offsets) or np.any(np.diff(offsets) <= 0) or offsets[0] <= 0:
        raise ValueError('Positive exact requested future grid required')
    horizon = offsets[-1]
    return {
        'history': np.column_stack((inputs['history_xy'], inputs['history_frame_offsets'] / horizon)).astype(np.float32),
        'history_mask': np.asarray(inputs['history_mask'], dtype=bool).copy(),
        'neighbors': np.concatenate((inputs['neighbor_xy'], inputs['neighbor_frame_offsets'][..., None] / horizon), axis=-1).astype(np.float32),
        'neighbor_mask': np.asarray(inputs['neighbor_mask'], dtype=bool).copy(),
        'baseline': np.asarray(inputs['baseline_rollouts'][BASELINES.index(baseline_name)], dtype=np.float32).copy(),
        'prediction_time': (offsets / horizon).astype(np.float32),
        'request_mask': np.ones(len(offsets), dtype=bool),
    }


class ContractForecastDataset(Dataset):
    def __init__(self, contract: ExperimentContract, recordings, *, purpose, baseline_name):
        if purpose not in {'fit', 'development'} or not recordings or len(set(recordings)) != len(recordings):
            raise ValueError('Only explicit nonempty fit/development recording lists may enter training')
        if baseline_name not in BASELINES:
            raise ValueError('Explicit causal baseline identity required')
        self.contract, self.recordings, self.purpose = contract, list(recordings), purpose
        self.baseline_name, self.readers, self.rows = baseline_name, [], []
        for name in recordings:
            reader, indices = contract.open_recording(name, purpose=purpose)
            self.rows.extend((len(self.readers), int(i)) for i in indices)
            self.readers.append(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        r, i = self.rows[item]
        reader = self.readers[r]
        inputs = reader.get_inputs(i)
        packed = pack_inputs(inputs, self.baseline_name)
        row = reader.index[i]
        history = reader.points[int(row['history_start']):int(row['current_row']) + 1]
        transform = causal_coordinate_transform(history[:, 2:4], history[:, 0], int(row['horizon_raw']))
        # Labels are read only after constructing the entire inference payload.
        labels = reader.get_labels(i)
        expected = history[-1, 0] + inputs['prediction_frame_offsets']
        if not np.array_equal(labels['future_frame_ids'], expected):
            raise ValueError('Target grid must match the exact request, not nearest future frame')
        target = ((labels['future_xy_dataset_local'] - transform['origin_xy']) @ transform['rotation'] / transform['scale'])
        if not np.isfinite(target).all():
            raise ValueError('Indexed complete target became nonfinite')
        identity = reader.identity(i)
        return {'inputs': packed, 'target': target.astype(np.float32),
                'target_mask': np.ones(len(target), dtype=bool), 'scale': transform['scale'],
                'identity': {**identity, 'data_role': self.purpose}}


def collate_inputs(rows):
    if not rows:
        raise ValueError('Nonempty batch required')
    if any(set(r) != INPUT_KEYS for r in rows):
        raise ValueError('Inference schema mismatch; labels/metadata are not input features')
    horizon = max(len(r['baseline']) for r in rows)
    inputs = {}
    for key in ('history', 'history_mask', 'neighbors', 'neighbor_mask'):
        inputs[key] = torch.from_numpy(np.stack([r[key] for r in rows]))
    for key, trailing, dtype in (('baseline', (2,), np.float32), ('prediction_time', (), np.float32),
                                ('request_mask', (), bool)):
        values = np.zeros((len(rows), horizon, *trailing), dtype=dtype)
        for i, row in enumerate(rows):
            value = row[key]
            values[i, :len(value)] = value
        inputs[key] = torch.from_numpy(values)
    return inputs


def collate_forecasts(rows):
    inputs = collate_inputs([r['inputs'] for r in rows])
    horizon = inputs['baseline'].shape[1]
    targets, masks = np.zeros((len(rows), horizon, 2), np.float32), np.zeros((len(rows), horizon), bool)
    for i, row in enumerate(rows):
        targets[i, :len(row['target'])] = row['target']
        masks[i, :len(row['target_mask'])] = row['target_mask']
    return {'inputs': inputs, 'target': torch.from_numpy(targets), 'target_mask': torch.from_numpy(masks),
            'scale': torch.tensor([r['scale'] for r in rows], dtype=torch.float32),
            'identities': [r['identity'] for r in rows]}


class PastContextForecaster(nn.Module):
    """Past-context Transformer with deterministic future queries; no rollout."""
    def __init__(self, *, width, heads, layers, neighbor_policy='observed_tokens'):
        super().__init__()
        if width % heads or min(width, heads, layers) <= 0:
            raise ValueError('Invalid Transformer dimensions')
        self.architecture = {'width': width, 'heads': heads, 'layers': layers}
        if neighbor_policy not in {'observed_tokens', 'complete_aligned_history'}:
            raise ValueError('Unknown past-neighbor support policy')
        self.neighbor_policy = neighbor_policy
        if neighbor_policy != 'observed_tokens':
            self.architecture['neighbor_policy'] = neighbor_policy
        self.embed = nn.Linear(3, width)
        self.modality = nn.Embedding(2, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.query = nn.Linear(3, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Only declared causal input fields are accepted')
        history, neighbors = inputs['history'], inputs['neighbors'].flatten(1, 2)
        hmask, nmask = inputs['history_mask'], inputs['neighbor_mask'].flatten(1, 2)
        mask = torch.cat((hmask, nmask), 1)
        values = torch.cat((history, neighbors), 1)
        if not hmask.any(1).all() or not torch.isfinite(values[mask]).all():
            raise ValueError('At least one finite observed history token required')
        if (values[..., 2][mask] > 0).any():
            raise ValueError('Context contains a post-current timestamp')
        if self.neighbor_policy == 'complete_aligned_history':
            aligned = torch.isclose(inputs['neighbors'][..., 2], history[:, None, :, 2],
                                    atol=1e-6, rtol=1e-6).all(-1)
            eligible = inputs['neighbor_mask'].all(-1) & aligned
            nmask = (inputs['neighbor_mask'] & eligible[..., None]).flatten(1, 2)
            mask = torch.cat((hmask, nmask), 1)
        values = torch.where(mask[..., None], values, 0.)
        token = self.embed(values)
        token[:, :history.shape[1]] = token[:, :history.shape[1]] + self.modality.weight[0]
        token[:, history.shape[1]:] = token[:, history.shape[1]:] + self.modality.weight[1]
        memory = self.encoder(token, src_key_padding_mask=~mask)
        request = inputs['request_mask']
        query_values = torch.cat((inputs['baseline'], inputs['prediction_time'][..., None]), -1)
        if not torch.isfinite(query_values[request]).all() or (inputs['prediction_time'][request] <= 0).any():
            raise ValueError('Invalid causal rollout/request')
        queries = self.query(torch.where(request[..., None], query_values, 0.))
        decoded, _ = self.attention(queries, memory, memory, key_padding_mask=~mask, need_weights=False)
        return torch.where(request[..., None], self.output(decoded + queries), 0.)


def risk_features(inputs, candidate):
    """Past context plus forecast disagreement; no target or target-valid mask."""
    request = inputs['request_mask']
    if candidate.shape != inputs['baseline'].shape or not torch.isfinite(candidate[request]).all():
        raise ValueError('Candidate forecast mismatch')
    features = [torch.where(inputs['history_mask'][..., None], inputs['history'], 0.).flatten(1),
                inputs['history_mask'].float().flatten(1),
                torch.where(inputs['neighbor_mask'][..., None], inputs['neighbors'], 0.).flatten(1),
                inputs['neighbor_mask'].float().flatten(1)]
    for prediction in (inputs['baseline'], candidate, candidate - inputs['baseline']):
        valid = torch.where(request[..., None], prediction, 0.)
        count = request.sum(1).clamp_min(1)[:, None]
        mean = valid.sum(1) / count
        var = torch.where(request[..., None], (prediction - mean[:, None]) ** 2, 0.).sum(1) / count
        endpoint = prediction[torch.arange(len(prediction), device=prediction.device), request.sum(1) - 1]
        features.extend((mean, var.sqrt(), endpoint))
    return torch.cat(features, 1)


class GainHarmHead(nn.Module):
    def __init__(self, feature_dim, *, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, 2), nn.Softplus())

    def forward(self, features):
        benefit, harm = self.network(features).unbind(-1)
        return {'benefit': benefit, 'harm': harm, 'gain': benefit - harm}


def cost_targets(baseline, candidate, target, target_mask, request_mask, *, metric):
    if metric not in {'ade', 'fde'}:
        raise ValueError('Explicit ADE or FDE cost target required')
    requests = np.asarray(request_mask)
    if requests.dtype != bool or requests.shape != np.shape(target_mask) or np.any(target_mask & ~requests):
        raise ValueError('Target validity must be contained in the requested grid')
    lengths = requests.sum(1)
    if np.any(lengths == 0) or not np.array_equal(requests, np.arange(requests.shape[1])[None] < lengths[:, None]):
        raise ValueError('Each requested grid must be a nonempty prefix, not a future-label mask')
    costs, valid = np.full((len(baseline), 2), np.nan, np.float32), np.zeros(len(baseline), bool)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        b, c, y = (np.asarray(a)[rows, :length] for a in (baseline, candidate, target))
        labels = realized_relative_costs(b, c, y, np.asarray(target_mask)[rows, :length], np.ones(len(rows)))
        gain = labels[f'gain_{metric}']
        costs[rows] = np.column_stack((np.maximum(gain, 0), labels[f'harm_{metric}']))
        valid[rows] = labels[f'{metric}_valid']
    return costs, valid


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def make_oof_cost_rows(contract, artifact_id, dataset, predictor, *, batch_size, device, progress=None):
    """Artifact verification precedes labels, including all upstream fit folds."""
    if dataset.purpose != 'fit' or dataset.contract.digest != contract.digest:
        raise ValueError('OOF costs require the same protocol and fit-only target rows')
    contract.assert_prediction_use(artifact_id, dataset.recordings, purpose='oof_risk_training')
    if getattr(predictor, '_verified_artifact_sha256', None) != contract.artifacts[artifact_id]['sha256']:
        raise ValueError('Predictor must be loaded from the verified artifact checkpoint')
    if parameter_digest(predictor) != predictor._verified_parameter_digest:
        raise ValueError('Verified predictor weights changed in memory')
    if dataset.baseline_name != predictor._fitted_baseline_name or batch_size < 1:
        raise ValueError('Baseline feature identity changed or invalid batch size')
    predictor.eval()
    features, targets, identities = [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collate_forecasts([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            candidate = predictor(inputs)
            costs, valid = cost_targets(batch['inputs']['baseline'].numpy(), candidate.cpu().numpy(),
                                        batch['target'].numpy(), batch['target_mask'].numpy(), batch['inputs']['request_mask'].numpy(),
                                        metric=contract.protocol['task']['primary_metric'])
            features.append(risk_features(inputs, candidate).cpu().numpy()[valid])
            targets.append(costs[valid])
            identities.extend(identity for identity, keep in zip(batch['identities'], valid) if keep)
            if progress is not None:
                progress(min(start + batch_size, len(dataset)), len(dataset))
    return {'features': np.concatenate(features), 'targets': np.concatenate(targets), 'identities': identities,
            'predictor_id': artifact_id, 'protocol_sha256': contract.digest,
            'predictor_sha256': contract.artifacts[artifact_id]['sha256'],
            'metric': contract.protocol['task']['primary_metric'], 'baseline_name': dataset.baseline_name,
            'feature_source': 'past_and_frozen_rollouts_only', 'target_source': 'held_fold_realized_relative_costs'}


def fit_linear_gain_harm(contract, groups, *, alpha):
    """Simple cost-aware control. Alpha is explicit; selection is not done here."""
    from sklearn.linear_model import Ridge

    if not groups or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Nonempty OOF supervision and explicit positive ridge penalty required')
    keys, parents, recordings = set(), set(), set()
    for group in groups:
        if (group['protocol_sha256'] != contract.digest or group['metric'] != contract.protocol['task']['primary_metric']
                or group['feature_source'] != 'past_and_frozen_rollouts_only'
                or group['target_source'] != 'held_fold_realized_relative_costs'):
            raise ValueError('Cost supervision provenance/schema mismatch')
        producer = group['predictor_id']
        if group['predictor_sha256'] != contract.artifacts[producer]['sha256']:
            raise ValueError('OOF predictor identity changed')
        used = sorted({r['recording_id'] for r in group['identities']})
        contract.assert_prediction_use(producer, used, purpose='oof_risk_training')
        if len(group['features']) != len(group['targets']) or len(group['features']) != len(group['identities']):
            raise ValueError('OOF features/labels/identities misaligned')
        for row in group['identities']:
            key = tuple(row[k] for k in ('recording_id', 'agent_id', 'frame_id', 'horizon_raw'))
            if key in keys:
                raise ValueError('Duplicated OOF target sample')
            keys.add(key)
        parents.add(producer)
        recordings.update(used)
    if len({g['baseline_name'] for g in groups}) != 1:
        raise ValueError('Cannot silently mix baseline risk targets')
    x, y = (np.concatenate([g[k] for g in groups]) for k in ('features', 'targets'))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.shape != (len(x), 2) or np.any(y < 0):
        raise ValueError('Finite nonnegative benefit/harm regression targets required')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    regressor = Ridge(alpha=alpha).fit((x - mean) / scale, y)
    return {'mean': mean, 'scale': scale, 'coef': regressor.coef_, 'intercept': regressor.intercept_,
            'alpha': float(alpha), 'fit_recordings': sorted(recordings), 'parents': sorted(parents),
            'protocol_sha256': contract.digest, 'normalization_source': 'fit_OOF_rows_only',
            'baseline_name': groups[0]['baseline_name'], 'metric': contract.protocol['task']['primary_metric']}


def predict_linear_gain_harm(head, features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != len(head['mean']) or not np.isfinite(features).all():
        raise ValueError('Cost-head feature schema mismatch')
    costs = np.maximum(0., ((features - head['mean']) / head['scale']) @ head['coef'].T + head['intercept'])
    return {'benefit': costs[:, 0], 'harm': costs[:, 1], 'gain': costs[:, 0] - costs[:, 1]}


def load_verified_forecaster(contract, artifact_id, *, device):
    contract._assert_frozen()
    record = contract.artifacts[artifact_id]
    if record['kind'] != 'forecaster':
        raise ValueError('Expected forecaster artifact')
    contract._verify_artifact(record)
    state = torch.load(contract._path(record['path']), map_location='cpu', weights_only=True)
    if state['step'] != state['identity']['settings']['steps']:
        raise ValueError('Fixed-budget predictor checkpoint is incomplete')
    if state['identity']['protocol_sha256'] != contract.digest:
        raise ValueError('Checkpoint belongs to another protocol')
    if sorted(state['identity']['fit_recordings']) != sorted(record['fit_recordings']):
        raise ValueError('Checkpoint fit provenance differs from artifact declaration')
    if state['identity']['source_code_sha256'] != file_digest(Path(__file__)):
        raise ValueError('Checkpoint training code identity changed')
    if state['architecture'] != state['identity']['architecture']:
        raise ValueError('Checkpoint architecture differs from its frozen training identity')
    model = build_forecaster(state['architecture']).to(device)
    if state['identity'].get('model_dependencies') != model_dependencies(model):
        raise ValueError('Checkpoint model dependency identity changed')
    model.load_state_dict(state['model'])
    model._verified_artifact_sha256 = record['sha256']
    model._verified_parameter_digest = parameter_digest(model)
    model._fitted_baseline_name = state['identity']['baseline_name']
    return model.eval()


def forecast_mse(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape[:2] or not valid.any():
        raise ValueError('Aligned supervised targets and at least one valid label required')
    error = (prediction[valid] - target[valid]).square().mean()
    if not torch.isfinite(error):
        raise ValueError('Nonfinite supervised loss')
    return error


def forecast_smooth_l1(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape[:2] or not valid.any():
        raise ValueError('Aligned supervised targets and at least one valid label required')
    error = nn.functional.smooth_l1_loss(prediction[valid], target[valid], beta=1., reduction='mean')
    if not torch.isfinite(error):
        raise ValueError('Nonfinite supervised loss')
    return error


def build_forecaster(architecture):
    options = dict(architecture)
    family = options.pop('family', 'past_context_transformer')
    output_mode = options.pop('output_parameterization', None)
    if output_mode is not None:
        from src.world_model.m3w_baseline_relative_forecaster import BaselineRelativeForecaster, MODES
        if output_mode not in MODES:
            raise ValueError('Unknown output parameterization')
        if family != 'past_context_transformer':
            raise ValueError('Residual parameterizations are registered only for the Transformer')
        return BaselineRelativeForecaster(build_forecaster({'family': family, **options}), mode=output_mode)
    conditioning = options.pop('input_conditioning', None)
    if conditioning not in {None, 'observed_joint_max_norm'}:
        raise ValueError('Unknown predictor input conditioning')
    if conditioning is not None:
        from src.world_model.m3w_context_conditioning import ObservedContextConditioner
        return ObservedContextConditioner(build_forecaster({'family': family, **options}))
    if family == 'past_context_transformer':
        return PastContextForecaster(**options)
    if family == 'eqmotion_fixed_head':
        from src.world_model.m3w_eqmotion_adapter import EqMotionFixedHead
        return EqMotionFixedHead(**options)
    raise ValueError('Unrecognized forecaster family')


def model_dependencies(model):
    return getattr(model, 'source_identity', {})


def train_forecaster(dataset, *, architecture, settings, output_dir, device='cpu', resume=False, stop_after=None):
    """Fixed-budget fitting only. Development selection is a separate caller step.

    Checkpoints include optimizer, sampler position, CPU RNG and complete identity.
    No dropout or stochastic accelerator operation is used by the supported cores.
    """
    if dataset.purpose != 'fit':
        raise ValueError('Gradient updates may only read fit recordings')
    contract = dataset.contract
    contract._assert_frozen()
    required = {'seed', 'steps', 'batch_size', 'learning_rate', 'checkpoint_every', 'heartbeat_every'}
    if set(settings) - {'objective'} != required or settings['seed'] not in contract.protocol['seeds']:
        raise ValueError('Explicit settings and protocol-listed seed required')
    objective = settings.get('objective', 'mse')
    if objective not in {'mse', 'smooth_l1'}:
        raise ValueError('Forecaster objective must be mse or smooth_l1')
    loss_fn = forecast_mse if objective == 'mse' else forecast_smooth_l1
    if any(type(settings[k]) is not int or settings[k] <= 0 for k in required - {'seed', 'learning_rate'}):
        raise ValueError('Positive integer iteration settings required')
    if not np.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0:
        raise ValueError('Positive learning rate required')
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(contract.root):
        raise ValueError('Output must remain inside the experiment workspace')
    identity = {'protocol_sha256': contract.digest, 'fit_recordings': dataset.recordings,
                'baseline_name': dataset.baseline_name, 'settings': settings, 'architecture': architecture,
                'source_code_sha256': file_digest(Path(__file__))}
    device = torch.device(device)
    runtime = {'device': str(device), 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
               'compute_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
               'dataloader_workers': 0}
    checkpoint = output_dir / 'latest.pt'
    if checkpoint.exists() and not resume:
        raise ValueError('Checkpoint exists; use identity-preserving resume')
    if resume and not checkpoint.exists():
        raise ValueError('Cannot resume a missing checkpoint')
    torch.manual_seed(settings['seed'])
    model = build_forecaster(architecture).to(device)
    identity['model_dependencies'] = model_dependencies(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(settings['seed'])
    step, cursor, elapsed, losses = 0, 0, 0., []
    runtime_history = []
    order = torch.randperm(len(dataset), generator=generator)
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Resume protocol/data/model/config/code identity changed')
        model.load_state_dict(state['model'])
        # AdamW restores moments to the parameter device while keeping its
        # non-capturable step counter on CPU. Do not override that policy.
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['sampler_rng'])
        step, cursor, order, losses, elapsed = state['step'], state['cursor'], state['order'], state['losses'], state['elapsed_seconds']
        runtime_history = state['runtime_history']
    if step > settings['steps']:
        raise ValueError('Checkpoint exceeds the declared fixed budget')
    if step == settings['steps']:
        return {'checkpoint': str(checkpoint), 'steps_completed': step, 'training_complete': True,
                'result_source': 'cached_verified', 'losses': losses, 'protocol_sha256': contract.digest,
                'fit_recordings': dataset.recordings, 'elapsed_seconds': elapsed,
                'new_confirmation_result': False, 'scope': 'fit_only_no_model_selection_or_risk_calibration',
                'dataloader_workers': 0, 'runtime': state['runtime'], 'runtime_history': runtime_history}
    resumed_from = step
    runtime_history = [*runtime_history, {'from_step': step, **runtime}]
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    def save():
        if device.type == 'mps':
            torch.mps.synchronize()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'architecture': architecture,
                 'identity': identity, 'runtime': runtime, 'runtime_history': runtime_history,
                 'step': step, 'cursor': cursor, 'order': order,
                 'torch_rng': torch.get_rng_state(), 'sampler_rng': generator.get_state(),
                 'losses': losses, 'elapsed_seconds': elapsed + time.monotonic() - began}
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)

    if not resume:
        save()
    try:
        while step < settings['steps']:
            if cursor == len(order):
                order, cursor = torch.randperm(len(dataset), generator=generator), 0
            ids = order[cursor:cursor + settings['batch_size']].tolist()
            batch = collate_forecasts([dataset[i] for i in ids])
            inputs = {k: value.to(device) for k, value in batch['inputs'].items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), batch['target'].to(device), batch['target_mask'].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step, cursor = step + 1, cursor + len(ids)
            if step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                with (output_dir / 'heartbeat.jsonl').open('a') as stream:
                    stream.write(json.dumps({'pid': os.getpid(), 'step': step, 'loss': losses[-1],
                                              'elapsed_seconds': elapsed + time.monotonic() - began}) + '\n')
            if step % settings['checkpoint_every'] == 0 or step == settings['steps'] or step == stop_after:
                save()
            if step == stop_after:
                break
    except KeyboardInterrupt:
        # Resume the last completed atomic checkpoint, never a half-applied update.
        raise
    result = {'checkpoint': str(checkpoint), 'steps_completed': step, 'training_complete': step == settings['steps'],
              'result_source': 'fresh_run', 'runtime': runtime, 'runtime_history': runtime_history,
              'resumed_from_step': resumed_from,
              'losses': losses, 'protocol_sha256': contract.digest, 'fit_recordings': dataset.recordings,
              'elapsed_seconds': elapsed + time.monotonic() - began, 'new_confirmation_result': False,
              'scope': 'fit_only_no_model_selection_or_risk_calibration', 'dataloader_workers': 0}
    (output_dir / 'fit_report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
