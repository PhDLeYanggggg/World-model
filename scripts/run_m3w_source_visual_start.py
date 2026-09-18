"""Registered visual start-information comparison on previously exposed fit roles."""
import argparse
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
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_m3w_sdd_auxiliary import Corpus
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_motion_start_probe import paired_agent_interval
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_sdd_auxiliary import load_registration
from src.world_model.m3w_source_start_probe import (
    stationary_membership, start_supervision, training_rows, fit_normalizer, normalize, metrics,
)
from src.world_model.m3w_source_visual_start import ARMS, VisualStart, fit_visual, probabilities


class VisualCorpus(Corpus):
    def __init__(self, reg):
        super().__init__(load_registration(ROOT, ROOT/reg['parent_registration']))
        self.mid = np.flatnonzero(stationary_membership(self.main['geometry']))
        source = np.flatnonzero(stationary_membership(self.aux['geometry']))
        sy, complete = start_supervision(self.aux['target'][source], self.aux['valid'][source])
        self.sid = source[complete]; sy = sy[complete]
        my, _ = start_supervision(self.main['targets'][self.mid], np.ones((len(self.mid), 12), bool))
        mx = observed_unit_frame(self.main['geometry'][self.mid])[0]
        sx = observed_unit_frame(self.aux['geometry'][self.sid])[0]
        previous = json.loads((ROOT/reg['previous_input_receipt']).read_text())
        for key, array in (('main_feature_sha256', mx), ('source_feature_sha256', sx),
                           ('main_supervision_sha256', my), ('source_supervision_sha256', sy)):
            if array_hash(array) != previous[key]:
                raise ValueError('Previous stationary population/schema changed: '+key)
        self.nmain = len(my); self.y = np.r_[my, sy]; self.x = np.concatenate((mx, sx))
        self.folds = self.main['folds'][self.mid]; self.stationary_tracks = self.tracks[self.mid]
        self.receipt = dict(previous, previous_inputs='cached_verified', pixel_inputs='past_only_original_cached32x32',
            images_or_id_features='RGB_and_coverage_no_identifiers', visual_cohort_filtered=False)
        self.allowed = np.zeros(len(self.y), bool)
        self.z = None

    def design(self, schedule, fold):
        main = np.flatnonzero(self.folds != fold) if fold >= 0 else np.empty(0, int)
        source = np.arange(self.nmain, len(self.y))
        ids, _, w = training_rows(main[:, None], self.y[main], source[:, None], self.y[source], schedule)
        ids = ids[:, 0].astype(int)
        self.normalizer = fit_normalizer(self.x[ids], w)
        self.z, clipped = normalize(self.x, self.normalizer)
        self.allowed[:] = False; self.allowed[ids] = True
        held = np.flatnonzero(self.folds == fold) if fold >= 0 else np.arange(self.nmain)
        return ids, w, held, clipped

    def raw_images(self, ids):
        ids = np.asarray(ids, dtype=int)
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= len(self.y))):
            raise ValueError('Valid stationary query ids required')
        rgb = np.empty((len(ids), 8, 3, 32, 32), np.uint8)
        cov = np.empty((len(ids), 8, 32, 32), np.uint8)
        main = ids < self.nmain
        if main.any():
            rows = self.main['image_rows'][self.mid[ids[main]]]
            rgb[main], cov[main] = self.main['rgb'][rows], self.main['coverage'][rows]
        source_locs = np.flatnonzero(~main)
        source = self.sid[ids[~main]-self.nmain]
        records = self.record_ids[source]
        for record in np.unique(records):
            loc = np.flatnonzero(records == record); store = self.images[int(record)]
            rows = store['image_rows'][self.local_ids[source[loc]]]
            rgb[source_locs[loc]], cov[source_locs[loc]] = store['rgb'][rows], store['coverage'][rows]
        if np.any(cov > 9):
            raise ValueError('Known nine-subpixel coverage schema required')
        return rgb, cov

    def inputs(self, ids, *, training=False):
        ids = np.asarray(ids, dtype=int)
        if training and not self.allowed[ids].all():
            raise ValueError('Held main query requested for training')
        rgb, cov = self.raw_images(ids)
        return (torch.from_numpy(self.z[ids].copy()),
                torch.from_numpy(rgb.astype(np.float32)/255),
                torch.from_numpy(cov[:, :, None].astype(np.float32)/9))

    def loss_labels(self, ids):
        ids = np.asarray(ids, dtype=int)
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= len(self.y))) or not self.allowed[ids].all():
            raise ValueError('Held label requested for training')
        return torch.from_numpy(self.y[ids].astype(np.float32))

    def coverage_audit(self):
        summaries = []
        for name, ids in [('ETH', np.flatnonzero(self.folds == 0)), ('Hotel', np.flatnonzero(self.folds == 1)),
                          ('SDD_train', np.arange(self.nmain, len(self.y)))]:
            means = []; observed = []
            for start in range(0, len(ids), 256):
                _, cov = self.raw_images(ids[start:start+256])
                means.append(cov.mean((2, 3))/9); observed.append((cov > 0).any((2, 3)))
            means, observed = np.concatenate(means), np.concatenate(observed)
            summaries.append(dict(population=name, rows=len(ids), frames=observed.size,
                rows_without_any_image=int((~observed.any(1)).sum()), observed_frames=int(observed.sum()),
                mean_pixel_coverage=float(means.mean()), current_frame_coverage=float(means[:, -1].mean())))
        return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--trial'); parser.add_argument('--stop-at', type=int)
    parser.add_argument('--audit-only', action='store_true'); parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = json.loads(args.registration.read_text())
    if (str(args.registration) != reg['registration_path'] or reg['role'] != 'fit_only_information_probe'
            or reg['arms'] != list(ARMS) or reg['schedules'] != ['main_only', 'source_only', 'mixed']
            or reg['seeds'] != [17, 29, 43]):
        raise ValueError('Fixed role, arms and comparison required')
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Registered dependency changed: '+path)
    out, reports = ROOT/reg['output'], ROOT/reg['reports']
    def heartbeat(**v):
        value = dict(pid=os.getpid(), timestamp_unix=time.time(), **v)
        json_write(out/'heartbeat.json', value); print(json.dumps(value), flush=True)
    heartbeat(state='verifying_assets')
    data = VisualCorpus(reg)
    identity = dict(data.identity, registration_sha256=file_digest(args.registration),
        torch=torch.__version__, numpy=np.__version__, torch_threads=4, interop_threads=1, num_workers=0)
    ip = out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Runtime/population identity changed')
    json_write(ip, identity)
    checks = dict(identity=identity, **data.receipt, coverage=data.coverage_audit())
    json_write(reports/'input_checks.json', checks)
    if args.audit_only:
        heartbeat(state='input_audit_complete_no_training', coverage=checks['coverage']); return
    keys = {f'{arm}_{schedule}_seed{seed}_fold{fold}' for arm in ARMS for schedule in reg['schedules']
            for seed in reg['seeds'] for fold in ([-1] if schedule == 'source_only' else [0, 1])}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot must name one training trial')
    trials, predictions, replays = [], {}, []
    new_fits, new_updates = 0, 0
    for arm in ARMS:
        for schedule in reg['schedules']:
            for seed in reg['seeds']:
                for fold in ([-1] if schedule == 'source_only' else [0, 1]):
                    key = f'{arm}_{schedule}_seed{seed}_fold{fold}'
                    if args.trial and key != args.trial:
                        continue
                    train, weights, held, clipped = data.design(schedule, fold)
                    ti = dict(identity, arm=arm, schedule=schedule, seed=seed, fold=fold,
                        training_rows_sha256=array_hash(train, weights, data.y[train]),
                        normalizer_sha256=array_hash(data.normalizer['mean'], data.normalizer['std'], data.normalizer['constant']))
                    cp, pp, rp = (out/'checkpoints'/(key+'.pt'), out/'predictions'/(key+'.npz'), out/'trials'/(key+'.json'))
                    old = json.loads(rp.read_text()) if rp.exists() else None
                    if old:
                        if (old['identity'] != ti or file_digest(cp) != old['checkpoint_sha256']
                                or file_digest(pp) != old['prediction_sha256']):
                            raise ValueError('Completed visual trial changed')
                        with np.load(pp, allow_pickle=False) as a:
                            np.testing.assert_array_equal(a['held_indices'], held)
                            predictions[key] = (held, a['probability'].copy())
                        if not args.replay:
                            trials.append(old); continue
                    heartbeat(state='fit_or_replay', trial=key, training_rows=len(train))
                    torch.manual_seed(seed); model = VisualStart()
                    if args.replay:
                        if not old:
                            raise ValueError('Replay requires a complete receipt')
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if state['identity'] != ti:
                            raise ValueError('Checkpoint identity changed')
                        model.load_state_dict(state['model'])
                    else:
                        fit = fit_visual(model, lambda ids: data.inputs(ids, training=True), data.loss_labels,
                            train, weights, arm=arm, seed=seed, config=reg['training'], identity=ti,
                            checkpoint=cp, heartbeat=lambda **v: heartbeat(trial=key, **v), stop_at=args.stop_at)
                        new_updates += fit['new_updates']
                        if not fit['complete']:
                            heartbeat(state='pilot_complete_no_held_evaluation', trial=key, **fit); return
                    p = probabilities(model, data.inputs, held, arm)
                    if args.replay:
                        np.testing.assert_array_equal(p, predictions[key][1]); replays.append(key); continue
                    train_p = probabilities(model, data.inputs, train, arm)
                    pp.parent.mkdir(parents=True, exist_ok=True); tmp = pp.with_suffix('.tmp.npz')
                    np.savez(tmp, held_indices=held, probability=p); os.replace(tmp, pp)
                    evaluation = []
                    for f in ([0, 1] if fold == -1 else [fold]):
                        mask = data.folds[held] == f; other = data.folds != f
                        prior = float((data.y[:data.nmain][other].sum()+1)/(other.sum()+2))
                        e = metrics(data.y[held][mask], p[mask], prior)
                        evaluation.append(dict(fold=f, reference_prior=prior,
                            agents=len(np.unique(data.stationary_tracks[held][mask])), **e))
                    result = dict(identity=ti, trial=key, arm=arm, schedule=schedule, seed=seed, fold=fold,
                        result_source='fresh_run_torch', fit=fit, training_rows=len(train),
                        parameter_count=sum(v.numel() for v in model.parameters()),
                        training_weighted_brier=float(np.sum(weights*(train_p-data.y[train])**2)),
                        training_weighted_prior=float(np.sum(weights*data.y[train])),
                        normalized_population_clipping_fraction=clipped, evaluation=evaluation,
                        checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                        prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp))
                    json_write(rp, result); trials.append(result); predictions[key] = (held, p); new_fits += 1
                    heartbeat(state='trial_complete', trial=key, brier_lifts=[e['brier_lift'] for e in evaluation])
    if args.replay:
        json_write(reports/'replay.json', dict(identity=identity, exact_trials=replays, all_exact=True, new_updates=0))
        heartbeat(state='replay_complete', trials=len(replays)); return
    if args.trial:
        return
    if len(trials) != 30:
        raise ValueError('All30 fixed fits required')
    report_path = reports/'report.json'
    if report_path.exists() and new_fits == 0:
        old = json.loads(report_path.read_text())
        if old['identity'] != identity or old['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(state='completed_resume_verified', new_fits=0, new_updates=0); return
    lookup = {(t['arm'], t['schedule'], t['seed'], t['fold']): t for t in trials}
    summary = {}
    for arm in ARMS:
        for schedule in reg['schedules']:
            items = []
            for fold in (0, 1):
                ps, refs, controls, metrics_list = [], [], [], []
                for seed in reg['seeds']:
                    t = lookup[arm, schedule, seed, -1 if schedule == 'source_only' else fold]
                    e = next(e for e in t['evaluation'] if e['fold'] == fold); metrics_list.append(e)
                    held, p = predictions[t['trial']]; ps.append(p[data.folds[held] == fold])
                    refs.append(np.full((data.folds == fold).sum(), e['reference_prior']))
                    control = lookup['mask_only', schedule, seed, -1 if schedule == 'source_only' else fold]
                    ids, pred = predictions[control['trial']]; controls.append(pred[data.folds[ids] == fold])
                y = data.y[:data.nmain][data.folds == fold]; tracks = data.stationary_tracks[data.folds == fold]
                items.append(dict(fold=fold, mean_brier=float(np.mean([e['brier'] for e in metrics_list])),
                    mean_prior_lift=float(np.mean([e['brier_lift'] for e in metrics_list])),
                    mean_auroc=float(np.mean([e['auroc'] for e in metrics_list])),
                    mean_auprc=float(np.mean([e['auprc'] for e in metrics_list])),
                    mean_ece=float(np.mean([e['ece'] for e in metrics_list])),
                    mean_mask_lift=float(np.mean((np.array(controls)-y)**2-(np.array(ps)-y)**2)),
                    prior_interval=paired_agent_interval(y,np.array(ps),np.array(refs),tracks),
                    mask_interval=paired_agent_interval(y,np.array(ps),np.array(controls),tracks)))
            summary[arm+'_'+schedule] = items
    report = dict(identity=identity, trials=trials, summary=summary, complete=True, fresh_models=30,
        fresh_prediction_cells=36, total_torch_updates=sum(t['fit']['step'] for t in trials),
        summed_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials), main_primary_changed=False,
        sealed_roles_opened=False, independent_confirmation=False, new_deployment=False,
        stage5c_executed=False, smc_enabled=False)
    json_write(report_path, report)
    heartbeat(state='complete', fresh_models=new_fits, new_updates=new_updates)


if __name__ == '__main__':
    main()
