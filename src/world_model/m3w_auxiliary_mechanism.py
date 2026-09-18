"""Fixed source-supervision ablation without changing the admitted source task."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_sdd_auxiliary import load_registration

NEW_ARMS = ('main4k', 'sdd_permuted')
ALL_ARMS = ('no_aux', 'sdd_aux', *NEW_ARMS)


def load_mechanism_registration(root, path):
    reg = json.loads(Path(path).read_text())
    if (reg.get('new_arms') != list(NEW_ARMS)
            or reg.get('main_primary_changed') is not False
            or reg.get('main_role') != 'fit_only_exploratory'
            or reg.get('seeds') != [17, 29, 43]
            or reg.get('permutation') != 'recording_and_exact_label_mask_residual_derangement'
            or reg.get('new_fits') != 54):
        raise ValueError('Frozen source-mechanism design required')
    for name, digest in reg['bindings'].items():
        if file_digest(root/name) != digest:
            raise ValueError('Changed registered dependency: '+name)
    parent = load_registration(root, root/reg['parent_registration'])
    if reg['modalities'] != parent['modalities']:
        raise ValueError('Same source/modalities required')
    return reg, parent


def arm_config(arm, parent):
    if arm not in NEW_ARMS:
        raise ValueError('Unregistered new arm')
    config = dict(parent['training'])
    if arm == 'main4k':
        config['pretraining_updates'] = 0
    return config, 'no_aux' if arm == 'main4k' else 'sdd_aux'


def residual_donors(records, valid, seed):
    """Training-label-only derangement preserves recording and exact support."""
    records = np.asarray(records)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.shape != (len(records), 12):
        raise ValueError('Twelve-point boolean support required')
    mask_code = (valid.astype(np.int64) * (1 << np.arange(12))).sum(1)
    _, groups = np.unique(np.column_stack((records, mask_code)), axis=0, return_inverse=True)
    order = np.argsort(groups, kind='stable')
    bounds = np.r_[0, np.flatnonzero(np.diff(groups[order]))+1, len(order)]
    donors = np.arange(len(records), dtype=np.int64)
    rng = np.random.default_rng(seed+32452843)
    for start, end in zip(bounds[:-1], bounds[1:]):
        ids = order[start:end]
        if len(ids) > 1:
            shuffled = rng.permutation(ids)
            donors[shuffled] = np.roll(shuffled, 1)
    np.testing.assert_array_equal(np.sort(donors), np.arange(len(records)))
    np.testing.assert_array_equal(records[donors], records)
    np.testing.assert_array_equal(valid[donors], valid)
    return donors


def permuted_batch(batch, ids, donors, source):
    """Modify loss labels only; prediction receives the unchanged input allowlist."""
    if not np.array_equal(source['valid'][donors[ids]], source['valid'][ids]):
        raise ValueError('Donor label support mismatch')
    residual = source['target'][donors[ids]]-source['baseline'][donors[ids]]
    target = batch['baseline'] + torch.from_numpy(np.asarray(residual).copy())
    result = dict(batch)
    result['target'] = torch.where(batch['valid'][..., None], target, torch.nan)
    return result


def donor_evidence(donors, records, valid, tracks):
    same = donors == np.arange(len(donors))
    supported = valid.any(1)
    return dict(rows=len(donors), sha256=hashlib.sha256(donors.tobytes()).hexdigest(),
        bijective=bool(np.array_equal(np.sort(donors), np.arange(len(donors)))),
        recording_preserved=bool(np.array_equal(records, records[donors])),
        label_support_preserved=bool(np.array_equal(valid, valid[donors])),
        singleton_rows=int(same.sum()), singleton_supported_rows=int((same & supported).sum()),
        same_recording_local_agent_rows=int((tracks == tracks[donors]).sum()),
        same_agent_supported_rows=int(((tracks == tracks[donors]) & supported).sum()),
        conditioning_uses_training_label_mask=True,
        input_features_changed=False, independent_training_examples_claim=False)
