"""Verify all fixed readout fits, retained bounds and zero-update resumption."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_source_conditioned_readout import registration
from scripts.run_m3w_source_importance_sampling import load_config, context
from scripts.run_m3w_source_crossfit import array_hash
from scripts.verify_m3w_source_motion_quality import preserve_verification
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_source_conditioned_readout import training_readout_gain, ConditionedTemporalDynamics
from src.world_model.m3w_source_episode_sampler import episode_weights
from src.world_model.m3w_source_importance_sampling import uniform_risk_factors
from src.world_model.m3w_source_crossfit import assemble_oof, cost_labels
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg = registration(args.registration); base = load_config(Path(reg['source_registration']))
    _, data, cached, _, _, ids, groups, _ = context(base)
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    report = json.loads((public/'report.json').read_text())
    replay = json.loads((public/'replay.json').read_text())
    control = json.loads((ROOT/reg['control_report']).read_text())
    assert len(report['trials']) == len(set(replay['exact_replays'])) == 24
    assert report['optimizer_updates'] == 240000
    hashes, gains, paired = {}, {}, {}
    pieces = {(arm, seed): [] for arm in base['arms'] for seed in base['seeds']}
    for trial in report['trials']:
        train, _, held, outer, scale = data.configure(trial['site'])
        p, _ = episode_weights(train, ids, groups); w = uniform_risk_factors(p)
        gain = training_readout_gain(data.radius[train-data.nmain], scale)
        gains[trial['site']] = gain
        assert not np.intersect1d(train, held).size and not np.intersect1d(train, outer).size
        saved = torch.load(ROOT/trial['checkpoint_path'], map_location='cpu', weights_only=False)
        assert saved['step'] == 10000 and saved['identity'] == trial['identity']
        assert saved['config'] == base['training'] and saved['scale'] == scale
        assert saved['identity']['readout_gain'] == gain
        np.testing.assert_array_equal(train, saved['train_ids'])
        np.testing.assert_array_equal(p, saved['probabilities'])
        np.testing.assert_array_equal(w, saved['importance_factors'])
        assert saved['draw_counts'].sum() == 640000
        rng = torch.Generator().manual_seed(trial['seed']+7919)
        draw_counts = np.zeros(len(train), dtype=np.int64)
        for _ in range(10000):
            draw = torch.multinomial(torch.from_numpy(p), 64, replacement=True, generator=rng).numpy()
            np.add.at(draw_counts, draw, 1)
        np.testing.assert_array_equal(draw_counts, saved['draw_counts'])
        assert torch.equal(rng.get_state(), saved['sampler_rng'])
        previous = next(t for t in control['trials'] if t['trial'] == trial['trial'])
        assert file_digest(ROOT/previous['checkpoint_path']) == previous['checkpoint_sha256']
        old = torch.load(ROOT/previous['checkpoint_path'], map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(saved['draw_counts'], old['draw_counts'])
        assert torch.equal(saved['sampler_rng'], old['sampler_rng'])
        model = ConditionedTemporalDynamics(trial['arm'], gain); model.load_state_dict(saved['model'])
        assert sum(v.numel() for v in model.parameters()) == 63960
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        key = trial['site'], trial['seed']
        if key in paired:
            np.testing.assert_array_equal(saved['draw_counts'], paired[key])
        paired[key] = saved['draw_counts']
        for kind in ('checkpoint', 'prediction'):
            path = ROOT/trial[kind+'_path']; assert file_digest(path) == trial[kind+'_sha256']
            hashes[trial[kind+'_path']] = file_digest(path)
        with np.load(ROOT/trial['prediction_path'], allow_pickle=False) as a:
            for subset, admitted in [('train', train), ('held', held)]:
                np.testing.assert_array_equal(a[subset+'_ids'], admitted)
                pred = a[subset+'_prediction']; loc = admitted-data.nmain
                assert np.isfinite(pred).all() and not pred[~data.support[loc]].any()
                assert np.all(np.linalg.norm(pred.astype(float), axis=-1) <= data.radius[loc, None]*1.00001+1e-7)
            pieces[(trial['arm'], trial['seed'])].append(dict(ids=held, prediction=a['held_prediction'].copy(),
                                                            cost_scale=np.full(len(held), scale)))
    for site in base['sites']:
        train, _, held, outer, _ = data.configure(site)
        for query in (held[:1], outer[:1], np.array([0])):
            for accessor in (lambda q: cached.inputs(q, training=True), data.loss_targets):
                try:
                    accessor(query)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Prohibited training role')
        before = cached.inputs(held[:8]); targets = data.target.copy()
        try:
            data.target[:] = np.nan
            after = cached.inputs(held[:8])
        finally:
            data.target[:] = targets
        assert all(torch.equal(a, b) for x, y in zip(before, after) for a, b in zip(x, y))
    for item in report['oof_labels']:
        pred, scale = assemble_oof(ids, pieces[(item['arm'], item['seed'])])
        path = ROOT/item['path']; assert file_digest(path) == item['sha256']
        with np.load(path, allow_pickle=False) as a:
            for name, value in dict(ids=ids, prediction=pred, cost_scale=scale,
                                    **cost_labels(pred, data.target[ids-data.nmain], scale)).items():
                np.testing.assert_array_equal(value, a[name])
        hashes[item['path']] = item['sha256']
    for path in [private/'identity.json', *private.joinpath('trials').glob('*.json'),
                 public/'report.json', public/'analysis.json', public/'replay.json']:
        hashes[str(path.relative_to(ROOT))] = file_digest(path)
    child = subprocess.run([sys.executable, 'scripts/run_m3w_source_conditioned_readout.py',
        '--registration', str(args.registration)], cwd=ROOT, text=True, capture_output=True, check=True)
    event = [json.loads(line) for line in child.stdout.splitlines() if line.startswith('{')][-1]
    assert event['state'] == 'training_complete' and event['new_updates'] == 0
    assert hashes == {path: file_digest(ROOT/path) for path in hashes}
    result = dict(result_source='fresh_run_verification_cached_verified_controls',
        registration_sha256=file_digest(args.registration), exact_replayed_heads=24,
        sampling_streams_regenerated=24, control_stream_matches=24, paired_arm_streams=len(paired),
        training_scale_by_excluded_site=gains, future_label_poison_input_checks=32,
        prohibited_training_role_checks=24, oof_archives_recomputed=6,
        immutable_artifacts=len(hashes), artifact_hashes=hashes, completed_resume=event,
        new_updates_on_resume=0, actual_training_updates=240000, main_outer_rows_scored=0,
        new_deployment=False, sensor_asof_certified=False, stage5c_executed=False, smc_enabled=False)
    preserve_verification(public/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
