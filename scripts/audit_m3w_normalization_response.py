"""Train-only parameter-gradient and reversible-frame audit; zero optimizer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_sdd_auxiliary import Corpus
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_observed_unit_frame import observed_unit_frame, SUMMARY_NAMES
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_sdd_auxiliary import load_registration, predict
from src.world_model.m3w_track_event_sampling import training_event_labels, EVENT_NAMES


def internal_arrays(geometry):
    pieces = [observed_unit_frame(geometry[i:i+4096]) for i in range(0, len(geometry), 4096)]
    return [np.concatenate([p[j] for p in pieces]) for j in range(4)]


def gradient_summary(model, prediction, target, radius, objective):
    error = torch.linalg.vector_norm(prediction-target, dim=-1).mean(-1)
    if objective == 'internal_log':
        if (radius <= 0).any():
            raise ValueError('Gradient probe requires observed spatial support for internal loss')
        error = error/radius
    losses = error.log1p()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = None; row_norms = []; head_norms = []; encoder_norms = []
    for i in range(len(losses)):
        grads = torch.autograd.grad(losses[i], [p for _, p in params], retain_graph=True, allow_unused=True)
        vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                            for (_, p), g in zip(params, grads)]).double()
        if not torch.isfinite(vector).all():
            raise FloatingPointError('Nonfinite parameter gradient')
        total = vector.clone() if total is None else total+vector
        row_norms.append(float(vector.norm()))
        sizes = [(n, p.numel()) for n, p in params]
        offset = 0; head2 = enc2 = 0.
        for name, size in sizes:
            value = float((vector[offset:offset+size]**2).sum()); offset += size
            if name.startswith('output.'):
                head2 += value
            else:
                enc2 += value
        head_norms.append(head2**.5); encoder_norms.append(enc2**.5)
    mean = total/len(losses)
    return dict(rows=len(losses), mean_log_loss=float(losses.detach().mean()),
        mean_error_in_loss_units=float(error.detach().mean()),
        per_row_parameter_gradient_norm_quantiles=np.quantile(row_norms, [0,.25,.5,.75,1]).tolist(),
        mean_per_row_parameter_gradient_norm=float(np.mean(row_norms)),
        mean_per_row_output_head_gradient_norm=float(np.mean(head_norms)),
        mean_per_row_encoder_gradient_norm=float(np.mean(encoder_norms)),
        norm_of_mean_parameter_gradient=float(mean.norm()),
        cancellation_ratio=float(mean.norm()/max(float(np.mean(row_norms)), 1e-30)),
        actual_autograd_parameter_gradients=True, pre_clipping=True), mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    for name, sha in reg['bindings'].items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Changed diagnostic binding: '+name)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    parent = load_registration(ROOT, ROOT/reg['parent_registration'])
    data = Corpus(parent); data.set_fold(0)
    started = time.monotonic()
    output = ROOT/reg['reports']
    identity = dict(registration_sha256=file_digest(args.registration), **data.identity,
                    torch=torch.__version__, numpy=np.__version__)
    def heartbeat(**v):
        value = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(ROOT/reg['output']/'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    heartbeat(state='building_internal_past_frames')
    # The diagnostic intentionally reads only fold-0 training main rows.
    main_ids = data.train
    complete = data.aux['valid'].all(1)
    main_frame = internal_arrays(data.main['geometry'][main_ids])
    source_frame = internal_arrays(data.aux['geometry'])
    mx, normalizer = supported_standardization(main_frame[0], main_frame[0])
    source_z = (source_frame[0]-normalizer['mean'])/normalizer['std']
    sx = np.clip(source_z, -10, 10).astype(np.float32)
    sx[:, normalizer['constant']] = 0.
    domains = {}
    manifest = json.loads(data.manifest_path.read_text())
    keys = np.concatenate([np.load(data.manifest_path.parent/r['recording']/'query_keys.npy')
                           for r in manifest['records']])
    source_tracks = np.array([str(r)+':'+str(k) for r, k in zip(data.record_ids, keys[:, 1])])
    for domain, ids, frame, features, target, geo, tracks in (
            ('main_train', main_ids, main_frame, mx, data.main['targets'][main_ids],
             data.main['geometry'][main_ids], data.tracks[main_ids]),
            ('source_train', np.arange(data.auxiliary_rows), source_frame, sx, data.aux['target'],
             data.aux['geometry'], source_tracks)):
        labels = np.full(len(ids), -1, int)
        admitted = np.ones(len(ids), bool) if domain == 'main_train' else complete
        labels[admitted] = training_event_labels(geo[admitted], target[admitted])
        domains[domain] = dict(ids=ids, frame=frame, features=features, labels=labels,
                              admitted=admitted, tracks=tracks)

    selections, support = {}, {}
    rng = np.random.default_rng(reg['sample_seed'])
    for domain, d in domains.items():
        support[domain] = dict(rows=len(d['ids']), complete_label_rows=int(d['admitted'].sum()),
            no_spatial_anchor_rows=int((~d['frame'][3]).sum()),
            complete_labels_without_anchor=int((d['admitted'] & ~d['frame'][3]).sum()),
            radius_quantiles=np.quantile(d['frame'][1], [0,.25,.5,.75,.95,1]).tolist())
        selections[domain] = {}
        for event, name in enumerate(EVENT_NAMES):
            possible = np.flatnonzero((d['labels'] == event) & d['frame'][3])
            ids = np.sort(rng.choice(possible, min(len(possible), reg['rows_per_event']), replace=False))
            selections[domain][name] = ids
            support[domain][name] = dict(population_rows=len(possible), sampled_rows=len(ids),
                sampled_tracks=len(np.unique(d['tracks'][ids])),
                sampled_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest())

    parent_report = json.loads((ROOT/reg['parent_report']).read_text())
    lookup = {(t['schedule'],t['modality'],t['seed'],t['fold']):t for t in parent_report['trials']}
    records = []
    for seed in reg['seeds']:
        for modality in reg['modalities']:
            for condition in reg['conditions']:
                torch.manual_seed(seed); model = OfflineVisualForecast(476).eval()
                checkpoint_hash = None
                if condition == 'trained_legacy':
                    t = lookup['sdd_aux', modality, seed, 0]
                    cp = ROOT/t['checkpoint_path']
                    if file_digest(cp) != t['checkpoint_sha256']:
                        raise ValueError('Changed frozen source checkpoint')
                    model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False)['model'])
                    checkpoint_hash = t['checkpoint_sha256']
                state_before = {n:v.clone() for n,v in model.state_dict().items()}
                for domain, d in domains.items():
                    weighted_sum = None; weighted_norms = 0.; population = sum(support[domain][n]['population_rows'] for n in EVENT_NAMES)
                    for name in EVENT_NAMES:
                        local = selections[domain][name]
                        if not len(local):
                            continue
                        batch = data.batch(d['ids'][local], modality, auxiliary=domain=='source_train', training=True)
                        radius = torch.from_numpy(d['frame'][1][local])
                        if condition.startswith('unit_'):
                            batch['geometry'] = torch.from_numpy(d['features'][local])
                            batch['baseline'] = torch.zeros_like(batch['baseline'])
                            delta = predict(model, batch, modality)
                            q = torch.from_numpy(d['frame'][2][local])
                            original = (data.aux['baseline'][d['ids'][local]] if domain=='source_train'
                                        else data.main['baselines'][d['ids'][local],data.cv])
                            prediction = torch.from_numpy(original.copy())+torch.bmm(delta, q.transpose(1,2))*radius[:,None,None]
                        else:
                            prediction = predict(model, batch, modality)
                        detail, vector = gradient_summary(model,prediction,batch['target'],radius,
                            'internal_log' if condition=='unit_internal_log' else 'primary_log')
                        weight = support[domain][name]['population_rows']/population
                        weighted_sum = vector*weight if weighted_sum is None else weighted_sum+vector*weight
                        weighted_norms += weight*detail['mean_per_row_parameter_gradient_norm']
                        records.append(dict(seed=seed,modality=modality,condition=condition,domain=domain,event=name,
                            checkpoint_sha256=checkpoint_hash,estimated_population_weight=weight,
                            weighted_mean_per_row_norm=weight*detail['mean_per_row_parameter_gradient_norm'],**detail))
                    records.append(dict(seed=seed,modality=modality,condition=condition,domain=domain,event='population_mixture_estimate',
                        estimated_norm_of_population_mean_gradient=float(weighted_sum.norm()),
                        estimated_population_mean_per_row_norm=weighted_norms,
                        sample_is_correlated_window_diagnostic=True))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value,state_before[name],rtol=0,atol=0)
                heartbeat(state='gradient_condition_complete',seed=seed,modality=modality,condition=condition)
    result = dict(identity=identity,result_source='fresh_run_autograd_and_input_analysis_cached_verified_training_assets',
        records=records, support=support, elapsed_seconds=time.monotonic()-started,
        new_optimizer_updates=0, completed_checkpoints_unmodified=True,
        main_training_fold=0, main_held_rows_opened_for_analysis=False,
        data_cache_integrity_check_includes_main_cache_files=True,
        sealed_roles_opened=False, future_targets_in_inputs=False,
        independent_confirmation=False, predictive_lift_evaluated=False,
        source_internal_feature_clipping_fraction=float((np.abs(source_z[:,~normalizer['constant']])>10).mean()),
        internal_summary_schema=SUMMARY_NAMES,
        notes=['Complete-label supported-row gradients only; partial/absent labels remain in training population.',
               'Parameter gradients are pre-clipping and measured, not inferred from scalar derivatives.',
               'Initialization uses zero output head; zero encoder gradients there are expected, not collapse.',
               'Internal-log objective is a diagnostic alternative, not a changed primary metric.',
               'A zero spatial anchor cannot identify a positive physical/local length from these coordinates.'])
    json_write(output/'report.json',result)
    heartbeat(state='complete',records=len(records),seconds=result['elapsed_seconds'],new_optimizer_updates=0)


if __name__ == '__main__':
    main()
