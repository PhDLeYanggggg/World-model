"""Fixed source-site information diagnostic within approved SDD training data."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The imported entry point checks arm64 before importing Torch.
from scripts.run_m3w_source_visual_start import VisualCorpus
import numpy as np
import torch
from scripts.run_m3w_source_start_probe import array_hash
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_source_start_probe import (
    stationary_membership, start_supervision, fit_normalizer, normalize, metrics,
)
from src.world_model.m3w_source_visual_start import VisualStart, fit_visual, probabilities

SITES = ('bookstore', 'coupa', 'deathCircle', 'gates', 'hyang')


def site_partition(sites, tracks, held_site):
    sites, tracks = np.asarray(sites), np.asarray(tracks)
    if sites.ndim != 1 or tracks.shape != sites.shape or held_site not in sites:
        raise ValueError('Aligned sites and scoped track identifiers required')
    train, held = np.flatnonzero(sites != held_site), np.flatnonzero(sites == held_site)
    if not len(train) or set(tracks[train]) & set(tracks[held]):
        raise ValueError('Empty training set or cross-site track leakage')
    return train, held, np.full(len(train), 1/len(train))


class SiteCorpus(VisualCorpus):
    def __init__(self, reg):
        super().__init__(reg)
        receipts = json.loads(self.manifest_path.read_text())['records']
        self.source_records = np.array([receipts[r]['recording'] for r in self.record_ids[self.sid]])
        self.source_sites = np.array([r.split('/')[0] for r in self.source_records])
        tracks = np.empty(len(self.sid), object)
        for record in np.unique(self.record_ids[self.sid]):
            rows = np.flatnonzero(self.record_ids[self.sid] == record)
            receipt = receipts[int(record)]
            path = self.manifest_path.parent/receipt['recording']/'query_keys.npy'
            if file_digest(path) != receipt['arrays']['query_keys.npy']:
                raise ValueError('Unverified source query metadata')
            keys = np.load(path, mmap_mode='r', allow_pickle=False)[self.local_ids[self.sid[rows]]]
            tracks[rows] = [f"{receipt['recording']}:{int(agent)}" for agent in keys[:, 1]]
        self.source_tracks = tracks.astype(str)
        if sorted(set(self.source_sites)) != list(SITES):
            raise ValueError('Source-site support changed')
        self.assignment_hash = array_hash(self.sid, self.source_sites, self.source_tracks)

    def source_design(self, site):
        train, held, weights = site_partition(self.source_sites, self.source_tracks, site)
        train, held = train+self.nmain, held+self.nmain
        self.normalizer = fit_normalizer(self.x[train], weights)
        self.z, clipped = normalize(self.x, self.normalizer)
        self.allowed[:] = False
        self.allowed[train] = True
        if len(np.unique(self.y[train])) != 2 or len(np.unique(self.y[held])) != 2:
            raise ValueError('Both classes required in source diagnostic fold')
        return train, weights, held, clipped

    def audit(self):
        receipts = json.loads(self.manifest_path.read_text())['records']
        stationary = np.flatnonzero(stationary_membership(self.aux['geometry']))
        _, complete = start_supervision(self.aux['target'][stationary], self.aux['valid'][stationary])
        all_sites = np.array([receipts[r]['recording'].split('/')[0] for r in self.record_ids[stationary]])
        rows = []
        for site in SITES:
            mask = self.source_sites == site
            train, weights, held, _ = self.source_design(site)
            records = []
            for recording in sorted(set(self.source_records[mask])):
                m = self.source_records == recording
                records.append(dict(recording=recording, rows=int(m.sum()),
                    agents=len(set(self.source_tracks[m])), positives=int(self.y[self.nmain:][m].sum())))
            rows.append(dict(site=site, stationary=int((all_sites == site).sum()),
                incomplete=int(((all_sites == site) & ~complete).sum()),
                rows=int(mask.sum()), positives=int(self.y[held].sum()),
                negatives=int(len(held)-self.y[held].sum()), positive_rate=float(self.y[held].mean()),
                agents=len(set(self.source_tracks[mask])), videos=len(records),
                training_rows=len(train), training_positive_rate=float(weights @ self.y[train]),
                records=records))
        return dict(result_source='fresh_run_support_audit_on_cached_verified_inputs',
            identity=self.identity, assignment_sha256=self.assignment_hash, sites=rows,
            rows=len(self.sid), local_agent_ids=len(set(self.source_tracks)),
            physical_sites=len(SITES), original_split='train40_only',
            no_recording_or_scoped_track_overlap=True, main_fit_used=False,
            formal_roles_changed=False, sealed_roles_opened=False,
            label='any_future_annotation_coordinate_change_complete_labels_only',
            label_is_human_gold=False, observation='offline_supplied_annotations_not_sensor_asof',
            horizon='8_to_12_stride12_plus144_raw_frames_no_seconds_equivalence')


def load_config(path):
    reg = json.loads(path.read_text())
    if (str(path) != reg['registration_path'] or reg['role'] != 'source_fit_internal_diagnostic'
            or reg['sites'] != list(SITES) or reg['arms'] != ['mask_only', 'past_rgb']
            or reg['seeds'] != [17, 29, 43] or reg['model_or_threshold_selection']):
        raise ValueError('Fixed diagnostic protocol required')
    for p, sha in reg['bindings'].items():
        if file_digest(ROOT/p) != sha:
            raise ValueError('Registered dependency changed: '+p)
    return reg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--trial')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = load_config(args.registration)
    out, reports = ROOT/reg['output'], ROOT/reg['reports']
    def heartbeat(**values):
        value = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(out/'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    heartbeat(state='verifying_source_assets')
    data = SiteCorpus(reg)
    identity = dict(data.identity, source_assignment_sha256=data.assignment_hash,
        registration_sha256=file_digest(args.registration), torch=torch.__version__,
        numpy=np.__version__, torch_threads=4, interop_threads=1, num_workers=0)
    ip = out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Runtime/population identity changed')
    json_write(ip, identity)
    checks = dict(data.audit(), registration_sha256=identity['registration_sha256'])
    json_write(reports/'support_audit.json', checks)
    if args.audit_only:
        heartbeat(state='source_support_complete_no_training', rows=checks['rows'],
                  sites=[{k:r[k] for k in ('site','rows','agents','videos','positives')} for r in checks['sites']])
        return
    keys = {f'{arm}_{site}_seed{seed}' for arm in reg['arms'] for site in SITES for seed in reg['seeds']}
    if args.trial and args.trial not in keys:
        raise ValueError('Unregistered trial')
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot must name one training-only trial')
    trials, replay = [], []
    new_fits, new_updates = 0, 0
    for site in SITES:
        train, weights, held, clipped = data.source_design(site)
        for seed in reg['seeds']:
            for arm in reg['arms']:
                key = f'{arm}_{site}_seed{seed}'
                if args.trial and args.trial != key:
                    continue
                ti = dict(identity, arm=arm, site=site, seed=seed,
                    training_rows_sha256=array_hash(train, weights, data.y[train]),
                    held_rows_sha256=array_hash(held),
                    normalizer_sha256=array_hash(data.normalizer['mean'],data.normalizer['std'],data.normalizer['constant']))
                cp, pp, rp = out/'checkpoints'/f'{key}.pt', out/'predictions'/f'{key}.npz', out/'trials'/f'{key}.json'
                old = json.loads(rp.read_text()) if rp.exists() else None
                if old:
                    if (old['identity'] != ti or file_digest(cp) != old['checkpoint_sha256']
                            or file_digest(pp) != old['prediction_sha256']):
                        raise ValueError('Completed source-site trial changed')
                    with np.load(pp, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['held_indices'], held)
                        saved_p = saved['probability'].copy()
                    if not args.replay:
                        trials.append(old); continue
                heartbeat(state='fit_or_replay', trial=key, training_rows=len(train), held_rows=len(held))
                torch.manual_seed(seed); model = VisualStart()
                if args.replay:
                    if old is None:
                        raise ValueError('Replay needs completed receipt')
                    state = torch.load(cp, map_location='cpu', weights_only=False)
                    if state['identity'] != ti or state['step'] != reg['training']['updates']:
                        raise ValueError('Checkpoint identity/budget changed')
                    model.load_state_dict(state['model'])
                else:
                    fit = fit_visual(model, lambda ids:data.inputs(ids,training=True), data.loss_labels,
                        train, weights, arm=arm, seed=seed, config=reg['training'], identity=ti,
                        checkpoint=cp, heartbeat=lambda **v:heartbeat(trial=key,**v), stop_at=args.stop_at)
                    new_updates += fit['new_updates']
                    if not fit['complete']:
                        heartbeat(state='pilot_complete_no_held_evaluation', trial=key, **fit); return
                p = probabilities(model, data.inputs, held, arm)
                if args.replay:
                    np.testing.assert_array_equal(saved_p, p); replay.append(key); continue
                train_p = probabilities(model, data.inputs, train, arm)
                pp.parent.mkdir(parents=True,exist_ok=True)
                tmp = pp.with_suffix('.tmp.npz')
                np.savez(tmp,held_indices=held,probability=p); os.replace(tmp,pp)
                prior = float(weights @ data.y[train])
                result = dict(identity=ti, trial=key, site=site, seed=seed, arm=arm,
                    result_source='fresh_run_torch', fit=fit, training_rows=len(train),
                    held_rows=len(held), training_prior=prior,
                    training_weighted_brier=float(weights @ ((train_p-data.y[train])**2)),
                    parameter_count=sum(v.numel() for v in model.parameters()),
                    normalized_population_clipping_fraction=clipped,
                    evaluation=metrics(data.y[held],p,prior),
                    checkpoint_path=str(cp.relative_to(ROOT)),checkpoint_sha256=file_digest(cp),
                    prediction_path=str(pp.relative_to(ROOT)),prediction_sha256=file_digest(pp))
                json_write(rp,result); trials.append(result); new_fits += 1
                heartbeat(state='trial_complete',trial=key,held_brier=result['evaluation']['brier'])
    if args.replay:
        json_write(reports/'replay.json',dict(identity=identity,exact_trials=replay,all_exact=True,new_updates=0))
        heartbeat(state='replay_complete',trials=len(replay)); return
    if args.trial:
        return
    if len(trials) != reg['fresh_models']:
        raise ValueError('Complete fixed source-site matrix required')
    path = reports/'report.json'
    if path.exists() and new_fits == 0:
        old = json.loads(path.read_text())
        if old['identity'] != identity or old['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(state='completed_resume_verified',new_fits=0,new_updates=0); return
    json_write(path,dict(identity=identity,result_source='fresh_run_torch',trials=trials,
        completed_models=len(trials),optimizer_updates=sum(t['fit']['step'] for t in trials),
        summed_fit_seconds=sum(t['fit']['fit_seconds'] for t in trials),
        main_primary_changed=False,sealed_roles_opened=False,new_deployment=False))
    heartbeat(state='matrix_complete',new_fits=new_fits,new_updates=new_updates)


if __name__ == '__main__':
    main()
