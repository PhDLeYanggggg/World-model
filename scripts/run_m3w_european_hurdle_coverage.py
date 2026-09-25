"""Compare frozen occurrence/severity risk rankings at matched locality counts."""
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
    raise RuntimeError('Native arm64 environment required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_hurdle_risk as parent
from scripts.report_m3w_european_hurdle_risk import verify as verified_parent
from scripts.run_m3w_native_forecast import immutable_json, json_write, array_hash
from scripts.run_m3w_native_nested import write_arrays
from src.evaluation.m3w_hurdle_coverage import ARMS, matched_choices, decompose
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.world_model.m3w_european_source_forecast import baseline_numpy

PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/european_hurdle_coverage_v1'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/european_hurdle_coverage_v1'
CONFIG = 'configs/m3w_european_hurdle_coverage_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_hurdle_coverage.py',
         'src/evaluation/m3w_hurdle_coverage.py', 'tests/test_m3w_hurdle_coverage.py',
         'outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/registration.md')
digest = parent.old.digest


def beat(state, **kwargs):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kwargs)
    json_write(PRIVATE / 'heartbeat.json', row)
    with (PRIVATE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    parent.assert_identity(identity['parent_identity'])
    for path, sha in identity['bindings'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Frozen coverage binding changed: ' + path)


def load():
    cfg = json.loads((ROOT / CONFIG).read_text())
    reg, previous, data, pid, originals = parent.load()
    old, _ = verified_parent()
    if (old['identity'] != pid or digest(parent.PUBLIC / 'analysis.json') != cfg['parent_analysis_sha256']
            or cfg['arms'] != list(ARMS) or any(cfg[k] != reg[k] for k in ('seeds', 'candidates', 'events', 'risk_budget'))
            or any(cfg[k] for k in ('new_training', 'threshold_refit', 'calibration_refit', 'reserved_roles_opened',
                                  'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Verified frozen source-only matched-count contract required')
    files = (*FILES, str((parent.PUBLIC / 'analysis.json').relative_to(ROOT)),
             str((parent.PUBLIC / 'verification.json').relative_to(ROOT)),
             str((parent.PUBLIC / 'completion_checks.json').relative_to(ROOT)))
    identity = dict(parent_identity=pid, bindings={p: digest(ROOT / p) for p in files},
                    result_role='opened_development_offline_ranking_diagnostic')
    assert_identity(identity)
    immutable_json(PRIVATE / 'identity.json', identity)
    return cfg, previous, data, identity, originals, old


def jobs(ctx):
    cfg, previous, data, identity, originals, _ = ctx
    for name, candidate, fold, seed, design in parent.jobs(previous, data, identity['parent_identity']):
        ids = design['held_ids']
        utility = parent.old.scores(originals[name + '_utility'], ids)
        moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
        for event in cfg['events']:
            risks = {arm: parent.old.scores(parent.checked(name + '_' + event + '_' + arm,
                     identity['parent_identity']), ids) for arm in ('product_mse', 'hurdle')}
            yield name + '_' + event, candidate, fold, seed, design, utility[:, 0] - utility[:, 1], moving, risks


def decide(ctx, *, pilot=False):
    cfg, _, data, identity, _, old = ctx
    archives = []
    for key, candidate, fold, seed, design, utility, moving, risks in jobs(ctx):
        if shutil.disk_usage(PRIVATE).free < 10 * 1024 ** 3:
            raise OSError('Below10GiB; preserve completed archives')
        ids = design['held_ids']
        choices, counts = matched_choices(utility, risks['product_mse'], risks['hurdle'], moving,
                                         data['sites'][ids], ids, budget=cfg['risk_budget'])
        for new, previous in (('product_original', 'product_mse'), ('hurdle_original', 'hurdle')):
            if array_hash(ids, choices[new]) != old['views'][key + '_' + previous]['decision_sha256']:
                raise ValueError('Original parent policy changed')
        path = PRIVATE / 'decisions' / (key + '.npz')
        write_arrays(path, dict(ids=ids, **choices))
        archives.append(dict(group=key, path=str(path.relative_to(ROOT)), sha256=digest(path),
                             counts=counts, choice_hashes={k: array_hash(ids, v) for k, v in choices.items()}))
        beat('decisions_frozen', group=key, rows=len(ids), pilot=pilot)
        if pilot:
            return
    if len(archives) != cfg['groups']:
        raise ValueError('Incomplete registered decision matrix')
    assert_identity(identity)
    immutable_json(PRIVATE / 'decisions_complete.json', dict(identity=identity, archives=archives,
                   outcome_labels_used=False, future_masks_used=False, trained_heads=0))


def read_decisions(identity):
    result = json.loads((PRIVATE / 'decisions_complete.json').read_text())
    if result['identity'] != identity or len(result['archives']) != 36:
        raise ValueError('All outcome-blind decisions must precede evaluation')
    archives = {}
    for ref in result['archives']:
        if digest(ROOT / ref['path']) != ref['sha256']:
            raise ValueError('Changed decision archive')
        archives[ref['group']] = ref
    return archives


def prediction(data, identity, candidate, fold, seed, ids):
    if candidate == 'neural':
        producers = identity['parent_identity']['geometric_identity']['old_identity']['producer_identity']
        ref = producers['frozen_final_producers'][f'single{fold}_seed{seed}']['prediction']
        relative = parent.old.read_predictions(ref, ids)
    else:
        relative = baseline_numpy(data['history'][ids], 3) - data['origin'][ids, None]
    return relative.astype(float) + data['origin'][ids, None]


def evaluate(ctx, *, verify=False):
    cfg, _, data, identity, _, old = ctx
    archives = read_decisions(identity)
    views, components, controls = {}, {}, 0
    for key, candidate, fold, seed, design, utility, moving, risks in jobs(ctx):
        ids = design['held_ids']; sites = data['sites'][ids]; roster = sorted(set(sites))
        with np.load(ROOT / archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            choices = {name: z[name].copy() for name in ARMS}
        cv, cvf = np.asarray(data['baseline_ade'][ids, 1]), np.asarray(data['baseline_fde'][ids, 1])
        ade, fde = native_errors(prediction(data, identity, candidate, fold, seed, ids),
                                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']), hard=cv >= design['hard_cut'])
        def metric(errors, reference, mask):
            return paired_scene_metrics(errors[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        selected_errors = {}
        for arm, use in choices.items():
            selected = np.where(use, ade, cv); selected_errors[arm] = selected
            scorer = 'hurdle' if arm.startswith('hurdle') else 'product_mse'
            risk = risks[scorer]
            predicted_violation = use & (risk[:, 1] > cfg['risk_budget'] * risk[:, 0])
            zero = np.isfinite(cv) & (cv == 0)
            row = dict(ADE_vs_CV={s: metric(selected, cv, m) for s, m in masks.items()},
                FDE_vs_CV=metric(np.where(use, fde, cvf), cvf, masks['all']),
                switch_rate=float(use.mean()), selected_rows=int(use.sum()),
                selected_unknown_ADE=int((use & ~np.isfinite(ade)).sum()),
                selected_unknown_FDE=int((use & ~np.isfinite(fde)).sum()),
                predicted_risk_violations=int(predicted_violation.sum()),
                zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((selected[zero] > 0).sum())),
                decision_sha256=array_hash(ids, use), by_locality={})
            for site in roster:
                ix = sites == site; known = ix & np.isfinite(ade)
                chosen = known & use
                row['by_locality'][str(site)] = dict(rows=int(ix.sum()), selected=int((ix & use).sum()),
                    selected_known_ADE=int(chosen.sum()), predicted_risk_violations=int((ix & predicted_violation).sum()),
                    selected_positive_harm_mean=float(np.maximum(ade[chosen]-cv[chosen], 0).mean()) if chosen.any() else None,
                    selected_net_gain_mean=float((cv[chosen]-ade[chosen]).mean()) if chosen.any() else None)
            row['observed_preservation'] = bool(row['ADE_vs_CV']['easy']['worst_scene_gain_percent'] is not None
                and row['ADE_vs_CV']['easy']['worst_scene_gain_percent'] >= -2 and not (selected[zero] > 0).any())
            if arm in ('product_original', 'hurdle_original'):
                original = old['views'][key + ('_product_mse' if arm == 'product_original' else '_hurdle')]
                for field in ('ADE_vs_CV', 'FDE_vs_CV', 'switch_rate', 'zero_CV', 'decision_sha256'):
                    if row[field] != original[field]:
                        raise ValueError('Parent control metric changed: ' + key + ' ' + field)
                controls += 1
            views[key + '_' + arm] = row
        components[key] = {s: decompose(selected_errors, cv, sites, mask=m,
                           resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed']) for s, m in masks.items()}
        beat('group_evaluated', group=key, verify=verify)
    if len(views) != 144 or len(components) != 36 or controls != 72:
        raise ValueError('Incomplete registered metric matrix')
    assert_identity(identity)
    result = dict(identity=identity, result_source='fresh_run_matched_count_diagnostic_cached_verified_scores_no_training',
        decision_manifest_sha256=digest(PRIVATE / 'decisions_complete.json'), archives=list(archives.values()),
        views=views, decomposition=components, parent_control_views_reproduced=controls,
        new_training=False, deployment_changed=False, reserved_roles_opened=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC / 'analysis.json', result)
    if verify:
        immutable_json(PUBLIC / 'replay.json', dict(identity=identity, all_passed=True,
                       analysis_sha256=digest(PUBLIC / 'analysis.json'), full_metric_views=144, parent_controls=72))
    beat('evaluation_complete', views=len(views), verify=verify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'pilot', 'decide', 'evaluate', 'verify'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('Explicit phase required')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        started = time.monotonic(); ctx = load()
        beat('verified_start', threads=4, workers=0, architecture=platform.machine())
        if args.pilot or args.decide or args.verify:
            decide(ctx, pilot=args.pilot)
        if args.evaluate or args.verify:
            if args.pilot:
                raise ValueError('No pilot outcome readout')
            evaluate(ctx, verify=args.verify)
        beat('phase_complete', seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
