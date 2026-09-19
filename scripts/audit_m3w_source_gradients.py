"""Inspect fixed source heads on training complements only; never take a step."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash
from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
from src.world_model.m3w_source_temporal_centered import CenteredTemporalDynamics
from src.world_model.m3w_source_cost_dynamics import restore
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_gradient_diagnostic import (
    flat_gradient, gradient_relation, clipped_gradient, layer_energy, output_shrink_curve,
)
import numpy as np
import torch


def population(model, cached, data, train, scale, batch_size):
    params = tuple(model.parameters())
    dim = sum(p.numel() for p in params)
    static, nonzero = np.zeros(dim), np.zeros(dim)
    prediction, target, radii = [], [], []
    for start in range(0, len(train), batch_size):
        ids = train[start:start+batch_size]
        features, frame = cached.inputs(ids, training=True)
        y = data.loss_targets(ids)
        pred = restore(model(*features, 'mask_only'), *frame)
        loss = torch.linalg.vector_norm(pred-y, dim=-1).mean(1)/scale
        mask = torch.all(y == 0, dim=(1, 2))
        static += flat_gradient(loss[mask].sum()/len(train), params, retain_graph=True)
        nonzero += flat_gradient(loss[~mask].sum()/len(train), params)
        prediction.append(pred.detach().numpy())
        target.append(y.numpy())
        radii.append(frame[0].numpy())
    return static, nonzero, np.concatenate(prediction), np.concatenate(target), np.concatenate(radii)


def sampled_gradients(model, cached, data, train, scale, probs, factors, *, count, batch_size, seed, cap, full):
    params = tuple(model.parameters())
    rng = torch.Generator().manual_seed(seed)
    raw, clipped, draws, static_fraction = [], [], [], []
    for _ in range(count):
        local = torch.multinomial(torch.from_numpy(probs), batch_size, replacement=True, generator=rng).numpy()
        ids = train[local]
        features, frame = cached.inputs(ids, training=True)
        target = data.loss_targets(ids)
        prediction = restore(model(*features, 'mask_only'), *frame)
        loss = torch.linalg.vector_norm(prediction-target, dim=-1).mean(1)/scale
        gradient = flat_gradient((loss*torch.as_tensor(factors[local], dtype=loss.dtype)).mean(), params)
        raw.append(gradient); clipped.append(clipped_gradient(gradient, cap)); draws.append(ids)
        static_fraction.append(float(torch.all(target == 0, dim=(1, 2)).float().mean()))
    raw, clipped = np.asarray(raw), np.asarray(clipped)
    rm, cm = raw.mean(0), clipped.mean(0)
    raw_norm = np.linalg.norm(raw, axis=1)
    full_clip = clipped_gradient(full, cap)
    summary = dict(batches=count, batch_size=batch_size, seed=seed, draw_ids_sha256=array_hash(np.asarray(draws)),
        clipped_fraction=float(np.mean(raw_norm > cap)),
        raw_norm_quantiles=np.quantile(raw_norm, [0, .25, .5, .75, 1]).tolist(),
        mean_static_fraction=float(np.mean(static_fraction)),
        raw_mean_vs_population=gradient_relation(rm, full),
        clipped_mean_vs_population=gradient_relation(cm, full),
        clipped_mean_vs_clipped_population=gradient_relation(cm, full_clip),
        clipped_mean_vs_raw_mean=gradient_relation(cm, rm),
        raw_mean_mc_vector_standard_error=float(np.sqrt(np.var(raw, axis=0, ddof=1).sum()/count)),
        negative_population_projection_fraction=float(np.mean(raw @ full < 0)),
        estimate_is_finite_monte_carlo=True, optimizer_step_analyzed=False)
    return summary, rm, cm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--trial')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = json.loads(args.registration.read_text())
    if (reg['role'] != 'fixed_checkpoint_training_gradient_diagnostic' or reg['models'] != 24
            or reg['new_updates'] != 0 or reg['new_held_predictions'] != 0
            or reg['probe_batches'] != 128 or reg['probe_batch_size'] != 64
            or reg['clip_cap'] != 5 or reg['shrink_alphas'] != [0, .25, .5, .75, 1]
            or reg['new_deployment'] or reg['stage5c_executed'] or reg['smc_enabled']):
        raise ValueError('Fixed training-only diagnostic contract required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed registered input: '+name)
    source_reg = load_config(Path(reg['source_registration']))
    _, data, cached, _, _, ids, groups, _ = context(source_reg)
    report = json.loads((ROOT/reg['source_report']).read_text())
    assert len(report['trials']) == 24
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    identity = dict(registration_sha256=file_digest(args.registration), source_report_sha256=file_digest(ROOT/reg['source_report']),
                    torch=torch.__version__, numpy=np.__version__, machine=platform.machine(),
                    torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(private/'identity.json', identity)

    def beat(**state):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **state)
        json_write(private/'heartbeat.json', event)
        print(json.dumps(event), flush=True)

    if args.trial and args.trial not in {t['trial'] for t in report['trials']}:
        raise ValueError('Unknown frozen trial')
    receipts, completed, computed = [], 0, 0
    for trial in report['trials']:
        key = trial['trial']
        if args.trial and args.trial != key:
            continue
        cp = ROOT/trial['checkpoint_path']
        if file_digest(cp) != trial['checkpoint_sha256']:
            raise ValueError('Changed frozen checkpoint')
        receipt_path = private/'trials'/f'{key}.json'
        if receipt_path.exists() and not args.replay:
            receipt = json.loads(receipt_path.read_text())
            assert receipt['identity'] == identity
            assert file_digest(ROOT/receipt['gradient_archive']) == receipt['gradient_sha256']
            assert receipt['checkpoint_sha256'] == file_digest(cp)
            receipts.append(receipt); completed += 1
            continue
        train, _, held, outer, scale = data.configure(trial['site'])
        assert array_hash(train) == trial['identity']['fold']['training_ids_sha256']
        for illegal in (held[:1], outer[:1], np.array([0])):
            for accessor in (lambda q: cached.inputs(q, training=True), data.loss_targets):
                try:
                    accessor(illegal)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Nontraining row passed role guard')
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        assert saved['identity'] == trial['identity'] and saved['step'] == 10000
        np.testing.assert_array_equal(saved['train_ids'], train)
        assert saved['scale'] == scale
        model = TemporalSourceDynamics('geometry') if trial['arm'] == 'geometry' else CenteredTemporalDynamics('centered')
        model.load_state_dict(saved['model']); model.eval()
        beat(state='gradient_probe', trial=key, training_rows=len(train))
        t0 = time.monotonic()
        static, nonzero, pred, target, radii = population(model, cached, data, train, scale, reg['population_batch_size'])
        full = static+nonzero
        named = list(model.named_parameters())
        moving = np.any(target != 0, axis=(1, 2))
        p, _ = episode_weights(train, ids, groups)
        factor = uniform_risk_factors(p)
        seed = reg['probe_seed']+trial['seed']+source_reg['sites'].index(trial['site'])*10000
        proposals, arrays = {}, dict(full_gradient=full, static_gradient=static, nonzero_gradient=nonzero)
        for proposal, prob, weight in [('uniform', np.full(len(train), 1/len(train)), np.ones(len(train))),
                                       ('importance_episode', p, factor)]:
            result, raw, clipped = sampled_gradients(model, cached, data, train, scale, prob, weight,
                count=reg['probe_batches'], batch_size=reg['probe_batch_size'], seed=seed,
                cap=reg['clip_cap'], full=full)
            proposals[proposal] = result
            arrays[proposal+'_raw_mean'] = raw; arrays[proposal+'_clipped_mean'] = clipped
        for name, value in model.state_dict().items():
            assert torch.equal(value, saved['model'][name])
        assert file_digest(cp) == trial['checkpoint_sha256']
        archive = private/'gradients'/f'{key}.npz'
        save_arrays(archive, arrays)
        receipt = dict(identity=identity, trial=key, site=trial['site'], seed=trial['seed'], arm=trial['arm'],
            result_source='fresh_run_fixed_model_training_gradients', training_rows=len(train),
            train_ids_sha256=array_hash(train), checkpoint_sha256=file_digest(cp),
            checkpoint_path=trial['checkpoint_path'], gradient_archive=str(archive.relative_to(ROOT)),
            gradient_sha256=file_digest(archive), static_rows=int((~moving).sum()), nonzero_rows=int(moving.sum()),
            full_gradient_norm=float(np.linalg.norm(full)),
            static_vs_full=gradient_relation(static, full), nonzero_vs_full=gradient_relation(nonzero, full),
            static_vs_nonzero=gradient_relation(static, nonzero),
            layers=layer_energy(full, named), static_layers=layer_energy(static, named),
            nonzero_layers=layer_energy(nonzero, named), proposals=proposals,
            radius_over_loss_scale_quantiles=np.quantile(radii/scale, [0, .25, .5, .75, 1]).tolist(),
            shrink_curve=output_shrink_curve(pred, target, data.native_scale[train-data.nmain], scale, reg['shrink_alphas']),
            parameters_unchanged=True, role_guard_checks=6, held_rows_scored=0, main_outer_rows_scored=0,
            new_updates=0, new_deployment=False)
        immutable_json(receipt_path, receipt)
        receipts.append(receipt); computed += 1
        beat(state='exact_replay' if args.replay else 'probe_complete', trial=key, seconds=time.monotonic()-t0)
    if args.trial:
        return
    assert len(receipts) == 24
    audit = dict(identity=identity, result_source='fresh_run_gradients_cached_verified_checkpoints',
        models=24, trials=receipts, full_training_population_gradients=24, minibatch_gradient_probes=24*2*reg['probe_batches'],
        new_updates=0, new_held_predictions=0, main_outer_rows_scored=0, new_deployment=False,
        sensor_asof_certified=False, coordinate_unit='pixel', horizon='8_to_12_steps_stride12_raw_frames',
        stage5c_executed=False, smc_enabled=False,
        limitations=['Frozen final iterates only; no causal training ablation.',
                     'Monte Carlo mean discrepancies do not prove estimator bias.',
                     'Clipped gradient directions are not Adam update directions.',
                     'Shared source training folds; not independent evaluation.',
                     'Constructor loads source labels, but only guarded training labels enter computations.'])
    immutable_json(public/'audit.json', audit)
    beat(state='audit_complete', fresh_probes=computed, cached_verified=completed, new_updates=0)


if __name__ == '__main__':
    main()
