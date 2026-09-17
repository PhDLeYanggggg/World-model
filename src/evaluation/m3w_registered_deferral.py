"""Supplementary fixed-predictor deferral, without modifying a parent protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_development_evaluation import content_digest
from src.world_model.m3w_cost_sensitive_deferral import (
    DeferralHead, deferral_decision, deferral_surrogate, validate_groups,
)
from src.world_model.m3w_oof_identity import oof_feature_identity


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def registered_spec(contract, reference):
    contract._assert_frozen()
    if set(reference) != {'path', 'sha256', 'variant'}:
        raise ValueError('Explicit registration path/hash/variant required')
    path = contract._path(reference['path'])
    if file_digest(path) != reference['sha256']:
        raise ValueError('Supplementary registration changed')
    registration = json.loads(path.read_text())
    if (registration.get('schema_version') != 1
            or registration.get('parent_protocol_sha256') != contract.digest
            or registration.get('scope') != 'development_diagnostic_only'
            or registration.get('eligible_for_selection') is not False
            or contract.protocol.get('study_design') != 'development_only'):
        raise ValueError('Supplement cannot change parent roles or selection rules')
    for name, digest in registration['bindings'].items():
        if file_digest(contract._path(name)) != digest:
            raise ValueError('Supplement source/decision binding changed: ' + name)
    spec = registration['variants'][reference['variant']]
    if (set(spec) != {'width', 'cost_bound', 'fit_settings'}
            or type(spec['width']) is not int or spec['width'] < 0
            or type(spec['cost_bound']) not in (int, float)
            or not np.isfinite(spec['cost_bound']) or spec['cost_bound'] <= 0):
        raise ValueError('Invalid explicit comparator spec')
    settings = spec['fit_settings']
    if set(settings) != {'steps', 'batch_size', 'learning_rate', 'checkpoint_every', 'heartbeat_every'}:
        raise ValueError('Explicit fixed fit budget required')
    if (any(type(settings[k]) is not int or settings[k] < 1 for k in settings if k != 'learning_rate')
            or not np.isfinite(settings['learning_rate']) or settings['learning_rate'] <= 0):
        raise ValueError('Invalid fit budget')
    return spec


def fit_registered(contract, groups, *, reference, seed, output, resume=False, stop_after=None):
    """CPU-safe resumable fit. Registration is explicit, never a forged parent."""
    spec = registered_spec(contract, reference)
    provenance = validate_groups(contract, groups)
    if seed not in contract.protocol['seeds'] or type(seed) is not int:
        raise ValueError('Parent-listed seed required')
    settings = spec['fit_settings']
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= settings['steps']):
        raise ValueError('stop_after outside budget')
    output = Path(output).resolve()
    if not output.is_relative_to(contract.root):
        raise ValueError('Workspace-local output required')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'parent_protocol_sha256': contract.digest, 'registration': reference, 'spec': spec,
                'seed': seed, **provenance, 'oof_feature_identity': oof_feature_identity(groups),
                'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'torch_version': str(torch.__version__), 'device': 'cpu', 'num_workers': 0}
    checkpoint = output / 'latest.pt'
    if checkpoint.exists() != resume:
        raise ValueError('Resume requires an existing checkpoint; fresh fit must not overwrite')
    x = torch.tensor(np.concatenate([g['features'] for g in groups]), dtype=torch.float32)
    raw = np.concatenate([g['targets'] for g in groups])
    costs = torch.tensor(np.minimum(raw / spec['cost_bound'], 1), dtype=torch.float32)
    torch.manual_seed(seed)
    head = DeferralHead(x.mean(0), x.std(0, correction=0).clamp_min(1e-6), width=spec['width'])
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    order, cursor, step, losses, elapsed = torch.randperm(len(x), generator=generator), 0, 0, [], 0.
    if resume:
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if saved['identity'] != identity:
            raise ValueError('Registered fit resume identity changed')
        head.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        generator.set_state(saved['sampler_rng'])
        order, cursor, step = saved['order'], saved['cursor'], saved['step']
        losses, elapsed = saved['losses'], saved['elapsed_seconds']
    if step > settings['steps'] or (stop_after is not None and step < settings['steps'] and stop_after <= step):
        raise ValueError('Invalid resumed budget')
    start, resumed_from = time.monotonic(), step

    def save():
        tmp = checkpoint.with_suffix('.tmp')
        torch.save({'identity': identity, 'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
                    'rng': torch.get_rng_state(), 'sampler_rng': generator.get_state(), 'order': order,
                    'cursor': cursor, 'step': step, 'losses': losses,
                    'elapsed_seconds': elapsed + time.monotonic() - start}, tmp)
        os.replace(tmp, checkpoint)

    if not resume:
        save()
    while step < settings['steps']:
        if cursor == len(order):
            order, cursor = torch.randperm(len(x), generator=generator), 0
        ids = order[cursor:cursor + settings['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss = deferral_surrogate(head(x[ids]), costs[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        step, cursor = step + 1, cursor + len(ids)
        if step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            atomic_json(output / 'heartbeat.json', {'pid': os.getpid(), 'step': step,
                        'loss': losses[-1], 'elapsed_seconds': elapsed + time.monotonic() - start})
        if step % settings['checkpoint_every'] == 0 or step in (settings['steps'], stop_after):
            save()
        if step == stop_after:
            break
    report = {'identity': identity, 'result_source': 'cached_verified' if resumed_from == step else 'fresh_run',
              'steps': step, 'training_complete': step == settings['steps'], 'resumed_from': resumed_from,
              'losses': losses, 'training_rows': len(x), 'feature_dimension': x.shape[1],
              'normalization_source': 'fit_OOF_rows_only', 'cost_clip_fraction': (raw > spec['cost_bound']).mean(0).tolist(),
              'checkpoint_sha256': file_digest(checkpoint), 'checkpoint': str(checkpoint.relative_to(contract.root)),
              'elapsed_seconds': elapsed + time.monotonic() - start, 'deployment_approved': False}
    atomic_json(output / 'fit_report.json', report)
    return report


def load_registered(contract, report):
    identity = report['identity']
    spec = registered_spec(contract, identity['registration'])
    path = contract._path(report['checkpoint'])
    if file_digest(path) != report['checkpoint_sha256']:
        raise ValueError('Deferral checkpoint changed')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (identity != state['identity'] or identity['spec'] != spec or identity['parent_protocol_sha256'] != contract.digest
            or not report['training_complete'] or state['step'] != spec['fit_settings']['steps']):
        raise ValueError('Deferral checkpoint identity/budget mismatch')
    development = [r for r, role in contract.protocol['assignments'].items() if role == 'development']
    for parent in identity['parents']:
        contract.assert_prediction_use(parent, development, purpose='development')
    head = DeferralHead(state['model']['mean'], state['model']['scale'], width=spec['width'])
    head.load_state_dict(state['model'])
    return head.eval()


def infer_scene(scene, forecaster, heads, *, baseline, geometry):
    """No reader or targets at the inference boundary. All head decisions at once."""
    from src.world_model.m3w_supervised_intervention import pack_inputs, collate_inputs, risk_features
    from src.data_unification.m3w_causal_recordings import restore_scene_rollouts
    from src.world_model.m3w_joint_intervention import past_proximity_edges, proximity_cost_table
    agents = scene['agents']
    inputs = collate_inputs([pack_inputs(a['inputs'], baseline) for a in agents])
    with torch.no_grad():
        candidate = forecaster(inputs)
        if candidate.shape != inputs['baseline'].shape:
            raise ValueError('Candidate shape differs from floor')
        finite = torch.isfinite(candidate).all(dim=(1, 2))
        candidate = torch.where(finite[:, None, None], candidate, inputs['baseline'])
        x = risk_features(inputs, candidate)
        decisions = {name: deferral_decision(head, x, support=finite) for name, head in heads.items()}
    b, c = inputs['baseline'].numpy(), candidate.numpy()
    ids = [a['agent_id'] for a in agents]
    common_b = restore_scene_rollouts(scene, dict(zip(ids, b)))['xy_dataset_local']
    common_c = restore_scene_rollouts(scene, dict(zip(ids, c)))['xy_dataset_local']
    edges = past_proximity_edges(np.stack([a['coordinate_transform']['origin_xy'] for a in agents]),
                                 radius=geometry['graph_radius'])
    pairs = proximity_cost_table(common_b, common_c, edges, distance_threshold=geometry['proximity_threshold'])
    switches = {'floor': np.zeros(len(ids), bool), 'uncontrolled': finite.numpy(),
                **{name: value['use_candidate'].numpy() for name, value in decisions.items()}}
    arms = {}
    for name, switch in switches.items():
        pair = pairs[np.arange(len(edges)), switch[edges[:, 0]].astype(int), switch[edges[:, 1]].astype(int)]
        arms[name] = {'switch': switch, 'prediction': np.where(switch[:, None, None], c, b),
                      'mean_pair_proxy': float(pair.mean()) if len(pair) else 0.,
                      'reason': 'fixed_argmax_no_threshold_selection',
                      'predicted_constraints_satisfied': None}
    return {'agent_ids': ids, 'baseline': b, 'candidate': c, 'arms': arms}


def summarize_registered(contract, rows):
    from src.evaluation.m3w_development_evaluation import _slice_metrics, paired_control_errors
    rules, task = contract.protocol['development_evaluation'], contract.protocol['task']
    metric = task['primary_metric']
    common = dict(metric=metric, aggregation=task['aggregation'], n_bootstrap=contract.protocol['bootstrap_resamples'],
                  seed=rules['bootstrap_seed'])
    slices = {'all': rows,
              'easy': [r for r in rows if r[f'baseline_{metric}'] is not None and r[f'baseline_{metric}'] <= rules['easy_threshold']],
              'hard': [r for r in rows if r[f'baseline_{metric}'] is not None and r[f'baseline_{metric}'] >= rules['hard_threshold']]}
    variants = [a for a in rows[0]['arms'] if a.startswith('deferral_')]
    arms = {arm: {name: _slice_metrics(part, arm, **common) for name, part in slices.items()}
            for arm in rows[0]['arms']}
    for arm in arms:
        arms[arm]['switch_rate_all_past_supported'] = float(np.mean([r['arms'][arm]['switch'] for r in rows]))
        arms[arm]['secondary_fde'] = _slice_metrics(rows, arm, **{**common, 'metric': 'fde'})
        arms[arm]['per_recording'] = {rec: _slice_metrics([r for r in rows if r['recording_id'] == rec], arm, **common)
                                    for rec in sorted({r['recording_id'] for r in rows})}
    return {'arms': arms, 'comparison_identity': content_digest([
                {k: v for k, v in r.items() if k != 'arms'} for r in rows]),
            'agent_queries': len(rows), 'physical_scenes': len({r['physical_scene'] for r in rows}),
            'comparison': 'same_forecasts_OOF_features_and_queries; deferral_unconstrained; no_equal_coverage_claim',
            'reference_budget_audit': 'not_run_no_risk_constraints_imposed_on_deferral',
            'paired_comparisons': {variant: {control: paired_control_errors(rows, variant, control, **common)
                                          for control in ('floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint')}
                                   for variant in variants},
            'eligible_for_selection': False, 'independent_confirmation': False, 'calibrated_risk': False}


def matched_rows(reference, fresh):
    """Reject population/forecast drift before reusing completed ordinary controls."""
    if len(reference) != len(fresh):
        raise ValueError('Query population changed')
    for old, new in zip(reference, fresh):
        if {k: v for k, v in old.items() if k != 'arms'} != {k: v for k, v in new.items() if k != 'arms'}:
            raise ValueError('Query identity/scale/label mismatch')
        for arm in ('floor', 'uncontrolled'):
            for metric in ('ade', 'fde'):
                if old['arms'][arm][metric] != new['arms'][arm][metric]:
                    raise ValueError('Fresh forecast scoring differs from completed parent')
    return [{**old, 'arms': {**old['arms'], **{k: v for k, v in new['arms'].items()
                                           if k not in ('floor', 'uncontrolled')}}}
            for old, new in zip(reference, fresh)]
