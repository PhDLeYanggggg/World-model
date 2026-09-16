"""Two-action, fixed-predictor regression-deferral comparator.

Adapted from Mao, Mohri & Zhong (ICML 2024), Eq. (3), single expert.
This is a bounded-cost control, not calibrated harm or a safety guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from src.world_model.m3w_supervised_intervention import (
    collate_forecasts, cost_targets, parameter_digest, risk_features,
)


def absolute_cost_targets(baseline, candidate, target, target_mask, request_mask, *, metric):
    """Labels only, in the existing past-normalized coordinate system.

    ADE uses available requested labels; FDE requires the requested endpoint.
    Padding and missing endpoints must not become an earlier evaluation time.
    """
    if metric not in {'ade', 'fde'}:
        raise ValueError('Explicit ADE/FDE metric required')
    _, valid = cost_targets(baseline, candidate, target, target_mask, request_mask, metric=metric)
    requests, mask = np.asarray(request_mask), np.asarray(target_mask)
    if requests.dtype != bool or mask.dtype != bool:
        raise ValueError('Boolean label and request masks required')
    b, c, y = (np.asarray(a) for a in (baseline, candidate, target))
    costs = np.full((len(b), 2), np.nan, np.float32)
    for i in np.flatnonzero(valid):
        positions = np.flatnonzero(mask[i]) if metric == 'ade' else [int(requests[i].sum()) - 1]
        costs[i] = [np.linalg.norm(p[i, positions] - y[i, positions], axis=-1).mean() for p in (b, c)]
    return costs, valid


def deferral_surrogate(logits, bounded_costs):
    """c_candidate * CE(action=baseline) + c_baseline * CE(action=candidate).

    Keep each row's cost mass: normalizing weights within a row would change
    the expected-risk objective. Base-2 log matches the paper's logistic form.
    """
    if (logits.ndim != 2 or logits.shape[1] != 2 or len(logits) == 0
            or logits.shape != bounded_costs.shape):
        raise ValueError('Aligned nonempty [rows, two actions] logits/costs required')
    if (not torch.isfinite(logits).all() or not torch.isfinite(bounded_costs).all()
            or torch.any(bounded_costs < 0) or torch.any(bounded_costs > 1)):
        raise ValueError('Finite logits and costs bounded in [0, 1] required')
    return -(bounded_costs.detach().flip(1) * logits.log_softmax(1)).sum(1).mean() / math.log(2)


def make_oof_deferral_rows(contract, artifact_id, dataset, predictor, *, batch_size, device, progress=None):
    if dataset.purpose != 'fit' or dataset.contract.digest != contract.digest:
        raise ValueError('OOF deferral requires the same protocol and fit-only rows')
    contract.assert_prediction_use(artifact_id, dataset.recordings, purpose='oof_risk_training')
    if getattr(predictor, '_verified_artifact_sha256', None) != contract.artifacts[artifact_id]['sha256']:
        raise ValueError('Predictor must come from a verified checkpoint')
    if parameter_digest(predictor) != predictor._verified_parameter_digest:
        raise ValueError('Verified predictor weights changed in memory')
    if dataset.baseline_name != predictor._fitted_baseline_name or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Baseline identity changed or invalid batch size')
    predictor.eval()
    features, targets, identities = [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collate_forecasts([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            candidate = predictor(inputs)
            # Inputs and rollout features are constructed before target-derived costs.
            x = risk_features(inputs, candidate).cpu().numpy()
            costs, valid = absolute_cost_targets(
                batch['inputs']['baseline'].numpy(), candidate.cpu().numpy(), batch['target'].numpy(),
                batch['target_mask'].numpy(), batch['inputs']['request_mask'].numpy(),
                metric=contract.protocol['task']['primary_metric'])
            features.append(x[valid])
            targets.append(costs[valid])
            identities.extend(row for row, keep in zip(batch['identities'], valid) if keep)
            if progress is not None:
                progress(min(start + batch_size, len(dataset)), len(dataset))
    if not features:
        raise ValueError('No OOF query rows')
    return {'features': np.concatenate(features), 'targets': np.concatenate(targets), 'identities': identities,
            'predictor_id': artifact_id, 'predictor_sha256': contract.artifacts[artifact_id]['sha256'],
            'held_recordings': sorted(dataset.recordings), 'protocol_sha256': contract.digest,
            'metric': contract.protocol['task']['primary_metric'], 'baseline_name': dataset.baseline_name,
            'feature_source': 'past_and_frozen_rollouts_only',
            'target_source': 'held_fold_absolute_past_normalized_errors'}


def validate_groups(contract, groups):
    contract._assert_frozen()
    if not groups:
        raise ValueError('Nonempty verified OOF groups required')
    keys, parents, recordings, digests = set(), set(), set(), []
    for group in groups:
        if (group['protocol_sha256'] != contract.digest or group['metric'] != contract.protocol['task']['primary_metric']
                or group['feature_source'] != 'past_and_frozen_rollouts_only'
                or group['target_source'] != 'held_fold_absolute_past_normalized_errors'):
            raise ValueError('OOF deferral provenance/schema mismatch')
        producer, held = group['predictor_id'], group['held_recordings']
        if group['predictor_sha256'] != contract.artifacts[producer]['sha256']:
            raise ValueError('OOF predictor identity changed')
        contract.assert_prediction_use(producer, held, purpose='oof_risk_training')
        for name in held:
            contract.open_recording(name, purpose='fit')
        x, y, rows = group['features'], group['targets'], group['identities']
        if (x.ndim != 2 or x.shape[1] < 1 or len(x) == 0 or y.shape != (len(x), 2) or len(rows) != len(x)
                or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0)):
            raise ValueError('Aligned finite features and nonnegative absolute costs required')
        for row in rows:
            if row['recording_id'] not in held or row['data_role'] != 'fit':
                raise ValueError('Row outside declared fit-only OOF fold')
            key = tuple(row[k] for k in ('recording_id', 'agent_id', 'frame_id', 'horizon_raw'))
            if key in keys:
                raise ValueError('Duplicated OOF query')
            keys.add(key)
        digest = hashlib.sha256(json.dumps({k: v for k, v in group.items() if k not in {'features', 'targets'}},
                                          sort_keys=True, allow_nan=False).encode())
        for array in (x, y):
            digest.update(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        digests.append(digest.hexdigest())
        parents.add(producer)
        recordings.update(held)
    if len({g['baseline_name'] for g in groups}) != 1:
        raise ValueError('Cannot mix baseline identities')
    return {'parents': sorted(parents), 'fit_recordings': sorted(recordings), 'group_sha256': digests,
            'baseline_name': groups[0]['baseline_name'], 'metric': groups[0]['metric']}


def comparator_spec(contract):
    contract._assert_frozen()
    spec = contract.protocol.get('comparators', {}).get('cost_sensitive_deferral')
    if not isinstance(spec, dict) or set(spec) != {'cost_bound', 'width', 'fit_settings'}:
        raise ValueError('Protocol must explicitly bind deferral cost_bound, width and fit_settings')
    if (isinstance(spec['cost_bound'], bool) or not np.isfinite(spec['cost_bound']) or spec['cost_bound'] <= 0
            or type(spec['width']) is not int or spec['width'] < 0):
        raise ValueError('Finite positive cost bound and nonnegative hidden width required')
    settings = spec['fit_settings']
    if set(settings) != {'steps', 'batch_size', 'learning_rate', 'checkpoint_every', 'heartbeat_every'}:
        raise ValueError('Explicit fixed-budget fit settings required')
    if any(type(settings[k]) is not int or settings[k] < 1 for k in settings if k != 'learning_rate'):
        raise ValueError('Positive integer fit settings required')
    if not np.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0:
        raise ValueError('Positive learning rate required')
    return spec


class DeferralHead(nn.Module):
    def __init__(self, mean, scale, *, width):
        super().__init__()
        mean, scale = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(scale, dtype=torch.float32)
        if (mean.ndim != 1 or len(mean) < 1 or mean.shape != scale.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Finite fit-only normalization required')
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.network = (nn.Linear(len(mean), 2) if width == 0 else
                        nn.Sequential(nn.Linear(len(mean), width), nn.GELU(), nn.Linear(width, 2)))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not torch.isfinite(features).all():
            raise ValueError('Deferral feature schema mismatch')
        return self.network((features - self.mean) / self.scale)


def deferral_decision(head, features, *, support):
    """Independent routing only; the softmax is NOT a calibrated harm probability."""
    with torch.no_grad():
        logits = head(features)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite deferral logits')
        if support.shape != (len(features),) or support.dtype != torch.bool:
            raise ValueError('Past-only boolean support required')
        choice = (logits[:, 1] > logits[:, 0]) & support  # Ties go to the baseline.
        return {'use_candidate': choice, 'logit_margin': logits[:, 1] - logits[:, 0],
                'calibrated_risk': False, 'deployment_approved': False}


def train_deferral(contract, groups, *, seed, output_dir, device='cpu', resume=False, stop_after=None):
    spec, provenance = comparator_spec(contract), validate_groups(contract, groups)
    if type(seed) is not int or seed not in contract.protocol['seeds']:
        raise ValueError('Protocol-listed seed required')
    settings = spec['fit_settings']
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= settings['steps']):
        raise ValueError('stop_after must be a positive step within the fixed budget')
    x, raw = (np.concatenate([g[k] for g in groups]) for k in ('features', 'targets'))
    x = torch.as_tensor(x, dtype=torch.float32)
    bounded = torch.as_tensor(np.minimum(raw / spec['cost_bound'], 1), dtype=torch.float32)
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
                'oof_feature_identity': oof_feature_identity(groups),
                'source_sha256': file_digest(Path(__file__)),
                'oof_identity_source_sha256': file_digest(Path(__file__).with_name('m3w_oof_identity.py')),
                'feature_backend_sha256': file_digest(Path(__file__).with_name('m3w_supervised_intervention.py'))}
    runtime = {'device': str(device), 'architecture': platform.machine(), 'torch_version': str(torch.__version__),
               'compute_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
               'dataloader_workers': 0}
    torch.manual_seed(seed)
    head = DeferralHead(mean, scale, width=spec['width']).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    step, cursor, elapsed, losses = 0, 0, 0., []
    order = torch.randperm(len(x), generator=generator)
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Deferral resume identity changed')
        head.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        generator.set_state(state['sampler_rng'])
        step, cursor, order = state['step'], state['cursor'], state['order']
        losses, elapsed = state['losses'], state['elapsed_seconds']
    if step > settings['steps']:
        raise ValueError('Checkpoint exceeds fixed budget')
    if stop_after is not None and step < settings['steps'] and stop_after <= step:
        raise ValueError('stop_after must follow the resumed checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    began, resumed_from = time.monotonic(), step

    def save():
        if str(device).startswith('mps'):
            torch.mps.synchronize()
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'identity': identity, 'runtime': runtime, 'model': head.state_dict(),
                    'optimizer': optimizer.state_dict(), 'step': step, 'cursor': cursor, 'order': order,
                    'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                    'losses': losses, 'elapsed_seconds': elapsed + time.monotonic() - began}, temporary)
        os.replace(temporary, checkpoint)

    if not resume:
        save()
    while step < settings['steps']:
        if cursor == len(order):
            order, cursor = torch.randperm(len(x), generator=generator), 0
        ids = order[cursor:cursor + settings['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss = deferral_surrogate(head(x[ids].to(device)), bounded[ids].to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        step, cursor = step + 1, cursor + len(ids)
        if step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            with (output / 'heartbeat.jsonl').open('a') as stream:
                stream.write(json.dumps({'pid': os.getpid(), 'step': step, 'surrogate_loss': losses[-1],
                                         'elapsed_seconds': elapsed + time.monotonic() - began}) + '\n')
        if step % settings['checkpoint_every'] == 0 or step == settings['steps'] or step == stop_after:
            save()
        if step == stop_after:
            break
    report = {'result_source': 'cached_verified' if resumed_from == settings['steps'] else 'fresh_run',
              'scope': 'fit_only_cost_sensitive_deferral_not_forecasting_evaluation',
              **provenance, 'protocol_sha256': contract.digest, 'seed': seed, 'spec': spec, 'runtime': runtime,
              'oof_feature_identity': identity['oof_feature_identity'],
              'steps_completed': step, 'resumed_from_step': resumed_from, 'training_complete': step == settings['steps'],
              'losses': losses, 'training_rows': len(x), 'feature_dimension': x.shape[1],
              'normalization_source': 'fit_OOF_rows_only',
              'cost_clip_fraction_by_action': (raw > spec['cost_bound']).mean(0).tolist(),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
              'test_evaluated': False, 'calibrated_risk': False, 'deployment_approved': False}
    temporary = output / 'fit_report.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(temporary, output / 'fit_report.json')
    return report


def load_verified_deferral(contract, artifact_id, *, device='cpu'):
    contract._assert_frozen()
    record = contract.artifacts[artifact_id]
    if record['kind'] != 'policy' or record.get('family') != 'cost_sensitive_deferral':
        raise ValueError('Expected cost-sensitive deferral policy artifact')
    contract._verify_artifact(record)
    state = torch.load(contract._path(record['path']), map_location='cpu', weights_only=True)
    identity = state['identity']
    if (identity['protocol_sha256'] != contract.digest or identity['spec'] != comparator_spec(contract)
            or identity['source_sha256'] != file_digest(Path(__file__))
            or identity['oof_identity_source_sha256'] != file_digest(Path(__file__).with_name('m3w_oof_identity.py'))
            or identity['feature_backend_sha256'] != file_digest(Path(__file__).with_name('m3w_supervised_intervention.py'))):
        raise ValueError('Deferral protocol or source identity changed')
    if (sorted(record['parents']) != identity['parents'] or sorted(record['fit_recordings']) != identity['fit_recordings']
            or record['selection_recordings'] or record['calibration_recordings']):
        raise ValueError('Deferral fit provenance changed')
    if state['step'] != identity['spec']['fit_settings']['steps']:
        raise ValueError('Deferral fixed-budget checkpoint incomplete')
    model = DeferralHead(state['model']['mean'], state['model']['scale'], width=identity['spec']['width']).to(device)
    model.load_state_dict(state['model'])
    model.fitted_identity = identity
    return model.eval()
