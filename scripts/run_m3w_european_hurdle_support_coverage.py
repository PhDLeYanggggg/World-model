"""Amended matched-count diagnosis retaining both full support strata."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_hurdle_coverage as base
from src.evaluation.m3w_hurdle_support_coverage import ARMS, support_choices, support_decomposition
import numpy as np

torch, parent = base.torch, base.parent
digest, immutable_json, json_write, array_hash = base.digest, base.immutable_json, base.json_write, base.array_hash
jobs, prediction = base.jobs, base.prediction
PUBLIC, PRIVATE = base.PUBLIC / 'support_v2', base.PRIVATE / 'support_v2'
CONFIG = 'configs/m3w_european_hurdle_support_coverage_v2.json'
FILES = (CONFIG, 'scripts/run_m3w_european_hurdle_support_coverage.py',
         'scripts/verify_m3w_european_hurdle_coverage.py',
         'src/evaluation/m3w_hurdle_support_coverage.py', 'tests/test_m3w_hurdle_support_coverage.py',
         'outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/amendment_support_v2.md',
         'outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/causal_support_mismatch.json')


def beat(state, **kwargs):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kwargs)
    json_write(PRIVATE / 'heartbeat.json', row)
    with (PRIVATE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def assert_identity(identity):
    base.assert_identity(identity['original_coverage_identity'])
    for file, sha in identity['bindings'].items():
        if digest(ROOT / file) != sha:
            raise ValueError('Changed amended binding: ' + file)


def load():
    cfg, previous, data, oid, originals, old = base.load()
    amendment = json.loads((ROOT / CONFIG).read_text())
    if amendment['arms'] != list(ARMS) or amendment['views'] != 216 or amendment['groups'] != 36:
        raise ValueError('Full and common support anchors required')
    cfg = dict(cfg, **amendment)
    identity = dict(parent_identity=oid['parent_identity'], original_coverage_identity=oid,
                    bindings={file: digest(ROOT / file) for file in FILES})
    assert_identity(identity)
    immutable_json(PRIVATE / 'identity.json', identity)
    return cfg, previous, data, identity, originals, old


def decide(ctx, *, pilot=False):
    cfg, _, data, identity, _, old = ctx
    records = []
    for key, candidate, fold, seed, design, utility, moving, risks in jobs(ctx):
        if base.shutil.disk_usage(PRIVATE).free < 10 * 1024 ** 3:
            raise OSError('Below10GiB; preserve completed archives')
        ids = design['held_ids']
        choices, counts = support_choices(utility, risks['product_mse'], risks['hurdle'], moving,
                                          data['sites'][ids], ids, budget=cfg['risk_budget'])
        for arm, previous in (('product_original', 'product_mse'), ('hurdle_original', 'hurdle')):
            if array_hash(ids, choices[arm]) != old['views'][key + '_' + previous]['decision_sha256']:
                raise ValueError('Full original policy changed')
        path = PRIVATE / 'decisions' / (key + '.npz')
        base.write_arrays(path, dict(ids=ids, **choices))
        records.append(dict(group=key, path=str(path.relative_to(ROOT)), sha256=digest(path), counts=counts,
                            choice_hashes={name: array_hash(ids, use) for name, use in choices.items()}))
        beat('decisions_frozen', group=key, rows=len(ids), pilot=pilot)
        if pilot:
            return
    assert len(records) == 36
    assert_identity(identity)
    immutable_json(PRIVATE / 'decisions_complete.json', dict(identity=identity, archives=records,
                   outcome_labels_used=False, future_masks_used=False, trained_heads=0))


def read_decisions(identity):
    result = json.loads((PRIVATE / 'decisions_complete.json').read_text())
    assert result['identity'] == identity and len(result['archives']) == 36
    for ref in result['archives']:
        if digest(ROOT / ref['path']) != ref['sha256']:
            raise ValueError('Changed decision archive')
    return {r['group']: r for r in result['archives']}


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
        ade, fde = base.native_errors(prediction(data, identity, candidate, fold, seed, ids),
                                     data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']), hard=cv >= design['hard_cut'])
        def metric(cost, reference, mask):
            return base.paired_scene_metrics(cost[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
        errors = {}
        for arm, use in choices.items():
            selected = np.where(use, ade, cv); errors[arm] = selected
            risk = risks['hurdle' if arm.startswith('hurdle') else 'product_mse']
            violation = use & (risk[:, 1] > cfg['risk_budget'] * risk[:, 0])
            zero = np.isfinite(cv) & (cv == 0)
            row = dict(ADE_vs_CV={s: metric(selected, cv, m) for s, m in masks.items()},
                FDE_vs_CV=metric(np.where(use, fde, cvf), cvf, masks['all']), switch_rate=float(use.mean()),
                selected_rows=int(use.sum()), selected_unknown_ADE=int((use & ~np.isfinite(ade)).sum()),
                selected_unknown_FDE=int((use & ~np.isfinite(fde)).sum()), predicted_risk_violations=int(violation.sum()),
                zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((selected[zero] > 0).sum())),
                decision_sha256=array_hash(ids, use), by_locality={})
            for site in roster:
                ix = sites == site; chosen = ix & np.isfinite(ade) & use
                row['by_locality'][str(site)] = dict(rows=int(ix.sum()), selected=int((ix & use).sum()),
                    selected_known_ADE=int(chosen.sum()), predicted_risk_violations=int((ix & violation).sum()),
                    selected_positive_harm_mean=float(np.maximum(ade[chosen]-cv[chosen], 0).mean()) if chosen.any() else None,
                    selected_net_gain_mean=float((cv[chosen]-ade[chosen]).mean()) if chosen.any() else None)
            easy = row['ADE_vs_CV']['easy']['worst_scene_gain_percent']
            row['observed_preservation'] = bool(easy is not None and easy >= -2 and not (selected[zero] > 0).any())
            if arm.endswith('_original'):
                prior = old['views'][key + ('_product_mse' if arm.startswith('product') else '_hurdle')]
                for field in ('ADE_vs_CV', 'FDE_vs_CV', 'switch_rate', 'zero_CV', 'decision_sha256'):
                    if row[field] != prior[field]:
                        raise ValueError('Parent control metric changed: ' + key + ' ' + field)
                controls += 1
            views[key + '_' + arm] = row
        components[key] = {s: support_decomposition(errors, cv, sites, mask=m,
                           resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed']) for s, m in masks.items()}
        beat('group_evaluated', group=key, verify=verify)
    assert len(views) == 216 and len(components) == 36 and controls == 72
    assert_identity(identity)
    result = dict(identity=identity, result_source='fresh_run_support_and_matched_count_diagnostic_no_training',
        decision_manifest_sha256=digest(PRIVATE / 'decisions_complete.json'), archives=list(archives.values()),
        views=views, decomposition=components, parent_control_views_reproduced=controls,
        new_training=False, deployment_changed=False, reserved_roles_opened=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC / 'analysis.json', result)
    if verify:
        immutable_json(PUBLIC / 'replay.json', dict(identity=identity, all_passed=True,
            analysis_sha256=digest(PUBLIC / 'analysis.json'), full_metric_views=216, parent_controls=72))
    beat('evaluation_complete', views=len(views), verify=verify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepare', 'pilot', 'decide', 'evaluate', 'verify'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if not any(vars(args).values()) or (args.pilot and (args.evaluate or args.verify)):
        parser.error('Explicit phase required; no pilot readout')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        started = time.monotonic(); ctx = load()
        beat('verified_start', threads=4, workers=0, architecture=base.platform.machine())
        if args.pilot or args.decide or args.verify:
            decide(ctx, pilot=args.pilot)
        if args.evaluate or args.verify:
            evaluate(ctx, verify=args.verify)
        beat('phase_complete', seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
