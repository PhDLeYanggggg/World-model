"""Fixed train-scale readout repair with matched inverse-probability exposure."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import immutable_json, save_arrays, array_hash, metrics
from src.world_model.m3w_source_conditioned_readout import ConditionedTemporalDynamics, training_readout_gain
from src.world_model.m3w_source_importance_sampling import fit_importance_candidate
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_cost_dynamics import forecast
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
from src.world_model.m3w_offline_visual_data import json_write
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def registration(path):
    reg = json.loads(path.read_text())
    if (reg['role'] != 'train_scale_readout_conditioning' or reg['models'] != 24
            or reg['updates'] != 240000 or reg['selection'] or reg['new_deployment']
            or reg['stage5c_executed'] or reg['smc_enabled']):
        raise ValueError('Fixed matched source comparison required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed registered artifact: '+name)
    return reg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--trial'); p.add_argument('--stop-at', type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.stop_at is not None and (not args.trial or args.replay):
        raise ValueError('Pilot must name a fixed trial')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = registration(args.registration)
    base = load_config(Path(reg['source_registration']))
    _, data, cached, _, _, ids, groups, episode_sha = context(base)
    control = json.loads((ROOT/reg['control_report']).read_text())
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    identity = dict(registration_sha256=file_digest(args.registration), data=data.identity,
        assignment=data.assignment_hash, episode_mapping_sha256=episode_sha,
        control_report_sha256=file_digest(ROOT/reg['control_report']),
        torch=torch.__version__, numpy=np.__version__, machine=platform.machine(),
        torch_threads=4, interop_threads=1, num_workers=0)
    immutable_json(private/'identity.json', identity)

    def beat(**state):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **state)
        json_write(private/'heartbeat.json', event); print(json.dumps(event), flush=True)

    if args.trial and args.trial not in {t['trial'] for t in control['trials']}:
        raise ValueError('Unknown trial')
    trials, replayed, updates = [], [], 0
    for old in control['trials']:
        key, arm, seed, site = old['trial'], old['arm'], old['seed'], old['site']
        if args.trial and key != args.trial:
            continue
        train, _, held, outer, scale = data.configure(site)
        probs, _ = episode_weights(train, ids, groups)
        gain = training_readout_gain(data.radius[train-data.nmain], scale)
        fold, hard = old['identity']['fold'], old['identity']['training_hard_cut']
        assert array_hash(train) == fold['training_ids_sha256']
        assert array_hash(held) == fold['held_ids_sha256'] and scale == fold['cost_scale']
        assert array_hash(train, probs) == old['identity']['sampler_sha256']
        for illegal in (held[:1], outer[:1], np.array([0])):
            try:
                cached.inputs(illegal, training=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Nontraining row passed training guard')
        ti = dict(identity, site=site, arm=arm, seed=seed, fold=fold, training_hard_cut=hard,
                  readout_gain=gain, sampler_sha256=array_hash(train, probs))
        cp, pp = private/'checkpoints'/f'{key}.pt', private/'predictions'/f'{key}.npz'
        rp = private/'trials'/f'{key}.json'
        receipt = json.loads(rp.read_text()) if rp.exists() else None
        if receipt:
            assert receipt['identity'] == ti and receipt['checkpoint_sha256'] == file_digest(cp)
            assert receipt['prediction_sha256'] == file_digest(pp)
            if not args.replay:
                trials.append(receipt); continue
        elif args.replay:
            raise ValueError('Cannot replay unfinished trial')
        torch.manual_seed(seed)
        model = ConditionedTemporalDynamics(arm, gain)
        if not args.replay:
            beat(state='fit_or_resume', trial=key, readout_gain=gain)
            fit = fit_importance_candidate(model, lambda q: cached.inputs(q, training=True), data.loss_targets,
                train, probs, scale=scale, seed=seed, config=base['training'], identity=ti,
                checkpoint=cp, heartbeat=lambda **kw: beat(trial=key, **kw), stop_at=args.stop_at)
            updates += fit['new_updates']
            if not fit['complete']:
                beat(state='pilot_complete_no_forecast', trial=key, step=fit['step']); return
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        assert saved['identity'] == ti and saved['step'] == 10000
        assert saved['config'] == base['training'] and saved['scale'] == scale
        assert file_digest(ROOT/old['checkpoint_path']) == old['checkpoint_sha256']
        previous = torch.load(ROOT/old['checkpoint_path'], map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(saved['draw_counts'], previous['draw_counts'])
        np.testing.assert_array_equal(saved['importance_factors'], previous['importance_factors'])
        assert torch.equal(saved['sampler_rng'], previous['sampler_rng'])
        model.load_state_dict(saved['model'])
        assert float(model.readout_gain) == gain
        arrays = dict(train_ids=train, held_ids=held,
            train_prediction=forecast(model, cached.inputs, train, 'mask_only'),
            held_prediction=forecast(model, cached.inputs, held, 'mask_only'))
        if args.replay:
            with np.load(pp, allow_pickle=False) as a:
                for name, value in arrays.items():
                    np.testing.assert_array_equal(a[name], value)
            replayed.append(key); beat(state='exact_replay', trial=key); continue
        save_arrays(pp, arrays)
        trial = dict(identity=ti, trial=key, site=site, seed=seed, arm=arm, fit=fit,
            parameters=sum(v.numel() for v in model.parameters()), result_source='fresh_run_conditioned_readout_fit',
            training=metrics(data, train, arrays['train_prediction'], scale, hard),
            held=metrics(data, held, arrays['held_prediction'], scale, hard),
            checkpoint_path=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
            prediction_path=str(pp.relative_to(ROOT)), prediction_sha256=file_digest(pp),
            matched_control_draws=True, sampled_draws_sha256=array_hash(saved['draw_counts']))
        immutable_json(rp, trial); trials.append(trial); beat(state='trial_complete', trial=key)
    if args.trial:
        return
    if args.replay:
        assert len(replayed) == 24
        immutable_json(public/'replay.json', dict(exact_replays=replayed, new_updates=0))
        beat(state='replay_complete', models=24); return
    assert len(trials) == 24 and sum(t['fit']['step'] for t in trials) == 240000
    archives = []
    for arm in base['arms']:
        for seed in base['seeds']:
            pieces = []
            for t in trials:
                if t['arm'] != arm or t['seed'] != seed:
                    continue
                with np.load(ROOT/t['prediction_path'], allow_pickle=False) as a:
                    pieces.append(dict(ids=a['held_ids'], prediction=a['held_prediction'],
                                       cost_scale=np.full(len(a['held_ids']), t['identity']['fold']['cost_scale'])))
            pred, scales = assemble_oof(ids, pieces)
            path = private/'oof'/f'{arm}_seed{seed}.npz'
            save_arrays(path, dict(ids=ids, prediction=pred, cost_scale=scales,
                                  **cost_labels(pred, data.target[ids-data.nmain], scales)))
            archives.append(dict(arm=arm, seed=seed, path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
    immutable_json(public/'report.json', dict(identity=identity, models=24, optimizer_updates=240000,
        rows=len(ids), trials=trials, oof_labels=archives, main_outer_rows_scored=0,
        new_deployment=False, stage5c_executed=False, smc_enabled=False))
    beat(state='training_complete', models=24, new_updates=updates)


if __name__ == '__main__':
    main()
