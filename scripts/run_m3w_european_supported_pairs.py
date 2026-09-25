"""Test supported-event pairing with all other risk-training factors frozen."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_hurdle_risk as parent
from scripts import run_m3w_european_ranked_hurdle as ranked
from scripts.report_m3w_european_ranked_hurdle import verify as verified_ranked
from scripts import run_m3w_european_hurdle_support_coverage as coverage
from scripts import verify_m3w_european_hurdle_coverage as independent
from scripts.report_m3w_european_hurdle_coverage import verify as verified_coverage
from src.world_model.m3w_supported_rank_pairs import fit, ranking_loss
from src.world_model.m3w_ranked_hurdle import ranking_loss as old_ranking_loss
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_hurdle_risk import initialize, predict
from src.evaluation.m3w_hurdle_support_coverage import ARMS, support_choices, support_decomposition
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_supported_pairs_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_supported_pairs_v1'
CONFIG = 'configs/m3w_european_supported_pairs_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_supported_pairs.py', 'src/world_model/m3w_supported_rank_pairs.py',
         'tests/test_m3w_supported_rank_pairs.py',
         'outputs/publication_readiness_2026_09/european_supported_pairs_v1/registration.md')
digest, immutable_json, json_write, array_hash = coverage.digest, coverage.immutable_json, coverage.json_write, coverage.array_hash
ARM_NAMES = {name: name.replace('product', 'control').replace('hurdle', 'ranked') for name in ARMS}


def beat(state, **kwargs):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kwargs)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    coverage.assert_identity(identity['coverage_identity'])
    ranked.assert_identity(identity['ranked_identity'])
    if digest(ranked.PUBLIC/'analysis.json') != identity['ranked_analysis_sha256']:
        raise ValueError('Frozen ranked control changed')
    if digest(coverage.PUBLIC/'analysis.json') != identity['coverage_analysis_sha256']:
        raise ValueError('Prior readout changed')
    for file, sha in identity['bindings'].items():
        if digest(ROOT/file) != sha:
            raise ValueError('Registered ranking binding changed: '+file)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    preg, previous, data, pid, originals = parent.load()
    prior = verified_coverage()
    ranked_prior = verified_ranked()
    if ranked_prior['identity']['parent_identity'] != pid:
        raise ValueError('Ranked control producer lineage differs')
    if cfg['prior_ranked_analysis_sha256'] != digest(ranked.PUBLIC/'analysis.json'):
        raise ValueError('Wrong pairing-control readout')
    if prior['identity']['parent_identity'] != pid:
        raise ValueError('Parent training lineage differs')
    if cfg['prior_coverage_analysis_sha256'] != digest(coverage.PUBLIC/'analysis.json'):
        raise ValueError('Wrong starting diagnosis')
    for key in ('seeds', 'candidates', 'events', 'head_training', 'risk_budget', 'bootstrap_resamples', 'bootstrap_seed'):
        if cfg[key] != preg[key]:
            raise ValueError('Unmatched design: '+key)
    if any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
                           'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled')):
        raise ValueError('Source-only frozen protocol required')
    identity = dict(parent_identity=pid, coverage_identity=prior['identity'],
        coverage_analysis_sha256=digest(coverage.PUBLIC/'analysis.json'),
        ranked_identity=ranked_prior['identity'], ranked_analysis_sha256=digest(ranked.PUBLIC/'analysis.json'),
        bindings={f: digest(ROOT/f) for f in FILES})
    assert_identity(identity)
    immutable_json(PRIVATE/'identity.json', identity)
    return cfg, previous, data, identity, originals


def checked(key, identity):
    r = json.loads((PRIVATE/'heads'/key/'complete.json').read_text())
    if r['identity']['experiment'] != identity or not r['fit']['complete'] or r['fit']['step'] != 2000:
        raise ValueError('Incomplete or changed ranking head')
    for ref in r['artifacts'].values():
        if digest(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Changed ranking artifact')
    return r


def train(ctx, *, pilot, resume):
    cfg, previous, data, identity, originals = ctx
    for name, candidate, fold, seed, design in parent.jobs(previous, data, identity['parent_identity']):
        a = parent.assemble(candidate, fold, seed, design, data, identity['parent_identity'])
        envelope = parent.geo.rollout_envelope(a['b'], a['p'])
        fitting, held = design['train_ids'], design['held_ids']
        for event in cfg['events']:
            key = name+'_'+event
            y, pr = parent.old.task_data(a, design, data, event)
            control = ranked.checked(key, identity['ranked_identity'])
            if control['identity']['target_sha256'] != array_hash(y) or control['identity']['lineage'] != a['lineage']:
                raise ValueError('Unmatched training features/targets')
            hid = dict(experiment=identity, lineage=a['lineage'], event=event, target_sha256=array_hash(y),
                envelope_train_sha256=array_hash(envelope[fitting]), control_checkpoint=control['artifacts']['checkpoint'])
            directory = PRIVATE/'heads'/key
            if (directory/'complete.json').exists():
                if checked(key, identity)['identity'] != hid:
                    raise ValueError('Changed fitting identity')
                continue
            if shutil.disk_usage(PRIVATE).free < 10*1024**3:
                raise OSError('Below10GiB; preserve completed work')
            beat('head_training', head=key)
            model, report = fit(a['x'][fitting], y, data['sites'][fitting], envelope[fitting], pr,
                seed=seed, settings=cfg['head_training'], identity=hid, directory=directory,
                rank_weight=cfg['rank_weight'], epsilon=cfg['rank_log_epsilon_in_training_cost_units'],
                resume=resume, stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(head=key, **kw))
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=report,
                    checkpoint=parent.geo.artifact(directory/'checkpoint.pt')))
                return
            values = predict(model, a['x'][held], envelope[held], pr, components=True)
            path = directory/'scores.npz'; tmp = path.with_suffix('.tmp.npz')
            np.savez(tmp, ids=held, scores=values[:, :2], factors=values[:, 2:]); os.replace(tmp, path)
            immutable_json(directory/'complete.json', dict(identity=hid, fit=report,
                artifacts=dict(checkpoint=parent.geo.artifact(directory/'checkpoint.pt'), scores=parent.geo.artifact(path))))
            assert_identity(identity); beat('head_complete', head=key, steps=report['step'])


def replay(ctx):
    cfg, previous, data, identity, _ = ctx
    checks = []
    for name, candidate, fold, seed, design in parent.jobs(previous, data, identity['parent_identity']):
        a = parent.assemble(candidate, fold, seed, design, data, identity['parent_identity'])
        ids = design['held_ids'][:cfg['replay_rows']]
        envelope = parent.geo.rollout_envelope(a['b'][ids], a['p'][ids])
        for event in cfg['events']:
            key = name+'_'+event; r = checked(key, identity)
            state = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
            ref = ranked.checked(key, identity['ranked_identity'])
            control = torch.load(ROOT/ref['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
            if state['identity'] != r['identity'] or state['step'] != 2000 or state['prior'] != control['prior']:
                raise ValueError('Wrong saved fitting state')
            np.testing.assert_array_equal(state['draws'], control['draws'])
            for field in ('mean', 'std', 'constant', 'weights', 'known'):
                np.testing.assert_array_equal(state['preprocess'][field], control['preprocess'][field])
            model = initialize(state['preprocess'], state['prior'], cfg['head_training']['width'], seed)
            model.load_state_dict(state['model'])
            values = predict(model, a['x'][ids], envelope, state['preprocess'], components=True)
            np.testing.assert_array_equal(values[:, :2], parent.old.scores(r, ids))
            np.testing.assert_array_equal(values[:, 2:], parent.factors(r, ids))
            assert np.all(values[:, 1] <= envelope+1e-4)
            checks.append(dict(head=key, checkpoint=r['artifacts']['checkpoint'], rows=len(ids), exact=True,
                               sampler_exact=True, parameters=r['fit']['parameters']))
        beat('head_group_replayed', group=name)
    assert len(checks) == 36
    immutable_json(PUBLIC/'head_replay.json', dict(identity=identity, checks=checks, all_passed=True))


def jobs(ctx):
    cfg, previous, data, identity, originals = ctx
    for name, candidate, fold, seed, design in parent.jobs(previous, data, identity['parent_identity']):
        ids = design['held_ids']
        values = parent.old.scores(originals[name+'_utility'], ids)
        moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
        for event in cfg['events']:
            key = name+'_'+event
            # Internal helper aliases: product=frozen ranked head, hurdle=supported-pair head.
            risks = dict(product_mse=parent.old.scores(ranked.checked(key, identity['ranked_identity']), ids),
                         hurdle=parent.old.scores(checked(key, identity), ids))
            yield key, candidate, fold, seed, design, values[:, 0]-values[:, 1], moving, risks


def decide(ctx):
    cfg, _, data, identity, _ = ctx
    receipt = json.loads((PUBLIC/'head_replay.json').read_text())
    if receipt['identity'] != identity or not receipt['all_passed'] or len(receipt['checks']) != 36:
        raise ValueError('Every fitted head must replay before any new readout')
    prior = json.loads((ranked.PUBLIC/'analysis.json').read_text())
    records = []
    for key, _, _, _, design, utility, moving, risks in jobs(ctx):
        ids = design['held_ids']
        choices, counts = support_choices(utility, risks['product_mse'], risks['hurdle'], moving,
                                          data['sites'][ids], ids, budget=cfg['risk_budget'])
        if array_hash(ids, choices['product_original']) != prior['views'][key+'_hurdle_original']['decision_sha256']:
            raise ValueError('Frozen hurdle control changed')
        path = PRIVATE/'decisions'/(key+'.npz')
        coverage.base.write_arrays(path, dict(ids=ids, **choices))
        records.append(dict(group=key, path=str(path.relative_to(ROOT)), sha256=digest(path), counts=counts))
        beat('decisions_frozen', group=key)
    assert len(records) == 36
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, archives=records,
        outcome_labels_used=False, future_masks_used=False))


def read_decisions(identity):
    r = json.loads((PRIVATE/'decisions_complete.json').read_text())
    if r['identity'] != identity or len(r['archives']) != 36:
        raise ValueError('Complete outcome-blind decisions required')
    for ref in r['archives']:
        if digest(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Decision archive changed')
    return {r['group']: r for r in r['archives']}


def evaluate(ctx, *, verify):
    cfg, _, data, identity, _ = ctx
    archives = read_decisions(identity)
    prior = json.loads((ranked.PUBLIC/'analysis.json').read_text())
    views, decompositions, versus, pair = {}, {}, {}, {}
    controls = checks = reductions = 0
    for key, candidate, fold, seed, design, utility, moving, risks in jobs(ctx):
        ids = design['held_ids']; sites = data['sites'][ids]; roster = sorted(set(sites))
        with np.load(ROOT/archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            bits = {name: z[name].copy() for name in ARMS}
        if verify:
            separate = independent.manual_choices(utility, risks, moving, sites, ids, cfg['risk_budget'])
            for arm in ARMS:
                np.testing.assert_array_equal(separate[arm], bits[arm]); checks += 1
        forecast = coverage.prediction(data, identity, candidate, fold, seed, ids)
        ade, fde = native_errors(forecast, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
        if verify:
            ma, mf = independent.coordinate_errors(forecast, data['target_eval'][ids], data['valid'][ids])
            independent.close(ma, ade); independent.close(mf, fde)
        cv, cvf = data['baseline_ade'][ids, 1], data['baseline_fde'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']), hard=cv >= design['hard_cut'])
        def metric(error, reference, mask):
            return paired_scene_metrics(error[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        errors = {}
        for arm, use in bits.items():
            errors[arm] = np.where(use, ade, cv)
            risk = risks['hurdle' if arm.startswith('hurdle') else 'product_mse']
            violation = use & (risk[:, 1] > cfg['risk_budget']*risk[:, 0])
            zero = np.isfinite(cv) & (cv == 0)
            row = dict(ADE_vs_CV={s: metric(errors[arm], cv, m) for s, m in masks.items()},
                FDE_vs_CV=metric(np.where(use, fde, cvf), cvf, masks['all']), switch_rate=float(use.mean()),
                selected_rows=int(use.sum()), selected_unknown_ADE=int((use & ~np.isfinite(ade)).sum()),
                selected_unknown_FDE=int((use & ~np.isfinite(fde)).sum()), predicted_risk_violations=int(violation.sum()),
                zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((errors[arm][zero] > 0).sum())),
                decision_sha256=array_hash(ids, use))
            easy = row['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            row['observed_preservation'] = bool(easy is not None and easy >= -2 and not (errors[arm][zero] > 0).any())
            if arm == 'product_original':
                for field in ('ADE_vs_CV', 'FDE_vs_CV', 'switch_rate', 'zero_CV', 'decision_sha256'):
                    if row[field] != prior['views'][key+'_hurdle_original'][field]:
                        raise ValueError('Control metrics changed: '+key+' '+field)
                controls += 1
            if verify:
                for s, m in masks.items():
                    independent.check_metric(np.where(use, ma, cv), cv, sites, m, row['ADE_vs_CV'][s], cfg); reductions += 1
                independent.check_metric(np.where(use, mf, cvf), cvf, sites, masks['all'], row['FDE_vs_CV'], cfg); reductions += 1
            views[key+'_'+arm] = row
        decompositions[key] = {s: support_decomposition(errors, cv, sites, mask=m,
            resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed']) for s, m in masks.items()}
        if verify:
            for s, m in masks.items():
                independent.check_decomposition({arm: np.where(use, ma, cv) for arm, use in bits.items()},
                                                 cv, sites, m, decompositions[key][s], cfg)
        event = key.rsplit('_', 1)[1]
        if candidate == 'neural':
            pair[(fold, seed, event)] = errors
        else:
            neural = pair.pop((fold, seed, event))
            for arm in ('product_original', 'hurdle_original'):
                versus[f'fold{fold}_seed{seed}_{event}_{ARM_NAMES[arm]}'] = {
                    s: metric(neural[arm], errors[arm], m) for s, m in masks.items()}
        beat('group_evaluated', group=key, verify=verify)
    assert len(views) == 216 and controls == 36 and len(versus) == 36 and not pair
    assert_identity(identity)
    result = dict(identity=identity, result_source='fresh_run_36_supported_pair_torch_heads_and_metrics',
        internal_helper_aliases=cfg['internal_helper_aliases'], arm_names=ARM_NAMES,
        decision_manifest_sha256=digest(PRIVATE/'decisions_complete.json'), views=views,
        decomposition=decompositions, neural_vs_damping=versus, parent_controls_reproduced=controls,
        new_heads=36, new_updates=72000, forecasts_unchanged=True, reserved_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC/'analysis.json', result)
    if verify:
        immutable_json(PUBLIC/'verification.json', dict(identity=identity, all_passed=True,
            analysis_sha256=digest(PUBLIC/'analysis.json'), metric_views=216, scalar_sorting_checks=checks,
            separate_coordinate_reductions=reductions, separate_decompositions=108, parent_controls=controls))
    beat('evaluation_complete', views=216, verify=verify)



def support_audit(ctx):
    """Read fitting targets only; do not open held-out outcome arrays."""
    cfg, previous, data, identity, _ = ctx
    rows = {}
    for name, candidate, fold, seed, design in parent.jobs(previous, data, identity['parent_identity']):
        a = parent.assemble(candidate, fold, seed, design, data, identity['parent_identity'])
        fitting = design['train_ids']
        sites = data['sites'][fitting]
        for event in cfg['events']:
            y, pr = parent.old.task_data(a, design, data, event)
            known = pr['known']
            groups = [np.flatnonzero(known & (sites == site)) for site in sorted(set(sites))]
            rng = torch.Generator().manual_seed(seed+7919)
            target = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
            stats = []
            for _ in range(cfg['support_audit_batches']):
                ids = draw_batch(groups, cfg['head_training']['batch_size'], rng)
                p = torch.ones_like(target[ids])
                _, old = old_ranking_loss(p, target[ids], sites[ids],
                    epsilon=cfg['rank_log_epsilon_in_training_cost_units'])
                _, new = ranking_loss(p, target[ids], sites[ids],
                    epsilon=cfg['rank_log_epsilon_in_training_cost_units'])
                stats.append(dict(old_pairs=old['rank_pairs'], new_pairs=new['rank_pairs'],
                                  supported_rows=new['rank_supported_rows']))
            rows[name+'_'+event] = dict(batches=len(stats),
                old_pairs=sum(r['old_pairs'] for r in stats),
                new_pairs=sum(r['new_pairs'] for r in stats),
                old_zero_batches=sum(r['old_pairs'] == 0 for r in stats),
                new_zero_batches=sum(r['new_pairs'] == 0 for r in stats),
                known_training_rows=int(known.sum()),
                supported_training_rows=int((known & (np.nansum(y, axis=1) > 0)).sum()),
                stats=stats)
        beat('fitting_support_audited', group=name)
    assert len(rows) == 36
    immutable_json(PUBLIC/'fitting_support.json', dict(identity=identity, fitting_only=True,
        no_outcome_selection=True, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'audit', 'pilot', 'train', 'resume', 'replay', 'decide', 'evaluate', 'verify'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()) or (args.pilot and (args.evaluate or args.verify or args.decide)):
        parser.error('Explicit phases; no pilot outcome readout')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        started = time.monotonic(); ctx = load()
        beat('verified_start', architecture=platform.machine(), threads=4, workers=0)
        if args.audit:
            support_audit(ctx)
        if args.pilot or args.train:
            train(ctx, pilot=args.pilot, resume=args.resume)
        if args.replay or args.verify:
            replay(ctx)
        if args.decide or args.verify:
            decide(ctx)
        if args.evaluate or args.verify:
            evaluate(ctx, verify=args.verify)
        beat('phase_complete', seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
