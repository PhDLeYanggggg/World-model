"""Approved train-only SDD auxiliary experiment, isolated from diagnostic artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_objective_alignment import geometry_prediction
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter, masked_future_ade

MODALITIES = ('geometry', 'mask_only', 'past_rgb')
SCHEDULES = ('no_aux', 'sdd_aux')


def validate_registration(reg):
    approval = reg.get('approval', {})
    if (approval.get('user_reply') != '按这个方案继续'
            or approval.get('date') != '2026-09-18'
            or reg.get('data_role') != 'supervised_auxiliary_training'
            or reg.get('original_split') != 'train'
            or reg.get('original_train_recordings') != 40
            or reg.get('raw_frame_stride') != 12
            or reg.get('observed_steps') != 8 or reg.get('predicted_steps') != 12
            or reg.get('main_role') != 'fit_only_exploratory'
            or reg.get('main_primary_changed') is not False):
        raise ValueError('Explicit approved source role, sampling and unchanged main task required')
    if reg.get('modalities') != list(MODALITIES) or reg.get('schedules') != list(SCHEDULES):
        raise ValueError('Registered matched source/modality matrix required')


def load_registration(root, path):
    reg = json.loads(Path(path).read_text())
    validate_registration(reg)
    if not reg.get('bindings'):
        raise ValueError('Code/source identity bindings required')
    for name, digest in reg['bindings'].items():
        if file_digest(root/name) != digest:
            raise ValueError('Changed registered dependency: '+name)
    return reg


def source_entries(root, reg):
    validate_registration(reg)
    split = json.loads((root/reg['split_manifest']).read_text())
    links = json.loads((root/reg['source_manifest']).read_text())
    videos = {r['scene_id']+'/'+r['video_id']: r for r in split['video_reports']}
    result = []
    for entry in links['records']:
        key = entry['annotation_key']
        if videos[key]['split_id'] != 'train':
            continue
        if videos[key]['annotation_path'] != entry['annotations_path']:
            raise ValueError('Annotation/split identity mismatch')
        result.append(entry)
    if len(result) != 40 or len({e['annotation_key'] for e in result}) != 40:
        raise ValueError('Original train-40 roster required')
    return result


class AuxiliaryGeometry:
    """Explicit new admission wrapper; old diagnostic metadata is not rewritten."""
    def __init__(self, rows, labels, recording, reg, allowed_recordings):
        validate_registration(reg)
        if recording not in allowed_recordings or len(set(allowed_recordings)) != 40:
            raise ValueError('Only the original training roster is admitted')
        self.adapter = SDDStepAdapter(rows, labels, recording, 12)
        self.data_role = reg['data_role']

    def inputs(self, item):
        return self.adapter.get_inputs(item)

    def labels_for_loss(self, item):
        return self.adapter.get_labels(item)


def predict(model, batch, modality):
    if modality == 'geometry':
        return geometry_prediction(model, batch['geometry'], batch['observed'], batch['baseline'])
    return model(batch['geometry'], batch['rgb'], batch['coverage'], batch['baseline'], modality)


def masked_log_loss(prediction, target, valid):
    ade, supported = masked_future_ade(prediction, target, valid)
    if supported.any():
        loss = ade[supported].log1p().mean()
    else:
        loss = prediction.sum()*0.
    return loss, dict(supported_rows=int(supported.sum()),
                     masked_ADE=float(ade[supported].detach().mean()) if supported.any() else None)


def fit_two_phase(model, main_batch, auxiliary_batch, main_count, auxiliary_count,
                  *, modality, schedule, config, seed, identity, checkpoint, heartbeat,
                  stop_at=None):
    if modality not in MODALITIES or schedule not in SCHEDULES or min(main_count, auxiliary_count) < 1:
        raise ValueError('Valid registered arm and nonempty training populations required')
    pre, total = config['pretraining_updates'], config['pretraining_updates']+config['main_updates']
    limit = total if stop_at is None else min(total, stop_at)
    if not 0 < limit <= total:
        raise ValueError('Invalid pilot limit')
    def optimizer():
        return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    opt = optimizer()
    generator = torch.Generator().manual_seed(seed+7919)
    step, losses, prior_seconds = 0, [], 0.
    draws = {'main': np.zeros(main_count, np.int64), 'auxiliary': np.zeros(auxiliary_count, np.int64)}
    empty_batches = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['config'] != config
                or state['schedule'] != schedule or state['modality'] != modality
                or state['counts'] != [main_count, auxiliary_count]):
            raise ValueError('Checkpoint identity, population or budget changed')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, losses, prior_seconds = state['step'], state['losses'], state['fit_seconds']
        draws, empty_batches = state['draw_counts'], state['unsupported_batches']
    initial, started = step, time.monotonic()
    phase = 'main' if step >= pre else 'pretraining'
    def save(state_name):
        seconds = prior_seconds+time.monotonic()-started
        state = dict(identity=identity, config=config, modality=modality, schedule=schedule,
            counts=[main_count, auxiliary_count], model=model.state_dict(), optimizer=opt.state_dict(),
            sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state(), step=step,
            losses=losses, fit_seconds=seconds, draw_counts=draws, unsupported_batches=empty_batches)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix('.tmp'); torch.save(state, temporary); os.replace(temporary, checkpoint)
        heartbeat(dict(state=state_name, phase=phase, step=step, fit_seconds=seconds,
                       loss=losses[-1]['loss'] if losses else None))
    model.train()
    try:
        while step < limit:
            if step == pre:
                opt = optimizer()
                generator.manual_seed(seed+104729)
            phase = 'main' if step >= pre else 'pretraining'
            use_aux = step < pre and schedule == 'sdd_aux'
            count = auxiliary_count if use_aux else main_count
            indices = torch.randint(count, (config['batch_size'],), generator=generator).numpy()
            batch = auxiliary_batch(indices) if use_aux else main_batch(indices)
            opt.zero_grad(set_to_none=True)
            loss, details = masked_log_loss(predict(model, batch, modality), batch['target'], batch['valid'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite auxiliary/main loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step(); step += 1
            np.add.at(draws['auxiliary' if use_aux else 'main'], indices, 1)
            empty_batches += int(details['supported_rows'] == 0)
            if step == 1 or step % 100 == 0 or step == total:
                losses.append(dict(step=step, phase=phase, loss=float(loss.detach()),
                                   gradient_norm=float(grad), **details))
            if step % config['checkpoint_every'] == 0 or step in (1, pre, limit):
                save('fit_complete' if step == total else 'training')
    except KeyboardInterrupt:
        heartbeat(dict(state='interrupted_resume_last_atomic_checkpoint', step=step))
        raise
    model.eval()
    return dict(step=step, complete=step == total, new_updates_this_invocation=step-initial,
        fit_seconds=prior_seconds+time.monotonic()-started, losses=losses,
        unsupported_batches=empty_batches,
        sampled_main_rows=int(np.count_nonzero(draws['main'])),
        sampled_auxiliary_rows=int(np.count_nonzero(draws['auxiliary'])),
        main_draws=int(draws['main'].sum()), auxiliary_draws=int(draws['auxiliary'].sum()))
