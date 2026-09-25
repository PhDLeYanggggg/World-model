"""Frozen neural opportunity over an equally protected damping floor, not deployment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_cross_moment as parent
from scripts.report_m3w_european_cross_moment import verify as verified_parent
from scripts.verify_m3w_floor_accounting import verify_ledger
from src.evaluation.m3w_floor_opportunity import floor_ledger, annotation_subsets, SUBSETS
from src.evaluation.m3w_opportunity_diagnosis import causal_reasons
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_floor_opportunity_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_floor_opportunity_v1'
CONFIG = 'configs/m3w_european_floor_opportunity_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_floor_opportunity.py',
         'scripts/verify_m3w_floor_accounting.py', 'src/evaluation/m3w_floor_opportunity.py',
         'tests/test_m3w_floor_opportunity.py', 'tests/test_m3w_floor_protocol.py',
         'outputs/publication_readiness_2026_09/european_floor_opportunity_v1/registration.md')
digest, immutable_json = parent.digest, parent.immutable_json


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    parent.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def validate(cfg):
    if (cfg['modes'] != ['batch', 'fitting'] or cfg['seeds'] != [17, 29, 43]
            or cfg['events'] != ['all', 'easy'] or tuple(cfg['subsets']) != SUBSETS
            or cfg['groups_per_mode'] != 18 or cfg['bootstrap_resamples'] != 3000
            or cfg['bootstrap_seed'] != 39271 or cfg['risk_budget'] != .02
            or cfg['actions'] != ['original_neural', 'rebased_neural', 'floor_neural_oracle', 'union_oracle']
            or any(cfg[k] for k in ('new_training', 'threshold_refit', 'calibration_refit',
                'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Frozen both-mode diagnostic only')


def load(mode):
    cfg = json.loads((ROOT/CONFIG).read_text()); validate(cfg)
    parent.configure(mode)
    ctx = parent.load()
    old = verified_parent()
    if digest(parent.PUBLIC/'analysis.json') != cfg['prior_analysis_sha256'][mode]:
        raise ValueError('Wrong frozen parent analysis')
    identity = dict(mode=mode, bindings={p: digest(ROOT/p) for p in FILES},
        prior_analysis_sha256=cfg['prior_analysis_sha256'][mode],
        prior_identity_sha256=digest(parent.PRIVATE/'identity.json'),
        prior_decisions_sha256=digest(parent.PRIVATE/'decisions_complete.json'))
    immutable_json(PRIVATE/mode/'identity.json', identity)
    return cfg, ctx, old, identity


def choices(na, nf, da, df, cv, cvf, ns, ds):
    d, fd = np.where(ds, da, cv), np.where(ds, df, cvf)
    oracle_n = na < d
    oracle_cv = cv < np.minimum(d, na)
    aa = dict(original_neural=np.where(ns, na, cv), rebased_neural=np.where(ns, na, d),
              floor_neural_oracle=np.minimum(d, na), union_oracle=np.minimum(cv, np.minimum(d, na)))
    ff = dict(original_neural=np.where(ns, nf, cvf), rebased_neural=np.where(ns, nf, fd),
              floor_neural_oracle=np.where(oracle_n, nf, fd),
              union_oracle=np.where(oracle_cv, cvf, np.where(oracle_n, nf, fd)))
    return d, fd, aa, ff


def run(mode, *, pilot=False, resume=False):
    cfg, ctx, old, identity = load(mode)
    data, lineage = ctx[2], ctx[3]
    archives = parent.read_decisions(lineage)
    pairs = {}; done = []; t0 = time.monotonic()
    for key, candidate, fold, seed, design, utility, moving, risks in parent.jobs(ctx):
        ids = design['held_ids']; event = key.rsplit('_', 1)[1]
        group = f'fold{fold}_seed{seed}_{event}'
        why = causal_reasons(utility, risks['hurdle'], moving, np.ones(len(ids), bool), budget=cfg['risk_budget'])
        use = why == 5
        with np.load(ROOT/archives[key]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            np.testing.assert_array_equal(use, z['hurdle_original'])
        if candidate == 'neural':
            pairs[group] = (ids, design, why, use, key)
            continue
        ni, nd, nw, ns, nk = pairs.pop(group)
        np.testing.assert_array_equal(ids, ni)
        assert design['easy_cut'] == nd['easy_cut'] and design['hard_cut'] == nd['hard_cut']
        destination = PRIVATE/mode/'groups'/f'{group}.json'
        if destination.exists():
            if not resume:
                raise ValueError('Existing group requires --resume')
            r = json.loads(destination.read_text())
            assert r['identity'] == identity and r['checks']['all_passed']
            done.append(group); continue
        if shutil.disk_usage(PRIVATE).free < 10*1024**3:
            raise OSError('Below10GiB; preserve work')
        sites = data['sites'][ids]; roster = sorted(set(sites)); assert len(roster) == 8
        valid, target = data['valid'][ids], data['target_eval'][ids]
        costs = {}
        for c in ('neural', 'damping097'):
            p = parent.coverage.prediction(data, lineage, c, fold, seed, ids)
            ade, fde = native_errors(p, target, valid, np.ones(len(ids)))
            independent = parent.independent.coordinate_errors(p, target, valid)
            parent.independent.close(ade, independent[0]); parent.independent.close(fde, independent[1])
            costs[c] = ade, fde
        na, nf = costs['neural']; da, df = costs['damping097']
        cv, cvf = data['baseline_ade'][ids, 1], data['baseline_fde'][ids, 1]
        d, fd, aa, ff = choices(na, nf, da, df, cv, cvf, ns, use)
        masks = annotation_subsets(valid, cv, easy_cut=design['easy_cut'], hard_cut=design['hard_cut'])
        reductions = 0
        def metric(error, reference, mask):
            nonlocal reductions
            value = paired_scene_metrics(error[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
            parent.independent.check_metric(error, reference, sites, mask, value, cfg)
            reductions += 1
            return value
        controls = 0
        for ck, ade, fde in ((nk, aa['original_neural'], ff['original_neural']), (key, d, fd)):
            v = old['views'][ck+'_hurdle_original']
            for sub in ('all', 'easy', 'hard'):
                assert metric(ade, cv, masks[sub]) == v['ADE_vs_CV'][sub]; controls += 1
            assert metric(fde, cvf, masks['all']) == v['FDE_vs_CV']; controls += 1
        metrics = {}
        zero = np.isfinite(cv) & (cv == 0)
        for action in cfg['actions']:
            metrics[action] = dict(
                ADE_vs_floor={s: metric(aa[action], d, m) for s, m in masks.items()},
                FDE_vs_floor={s: metric(ff[action], fd, m) for s, m in masks.items()},
                ADE_vs_CV={s: metric(aa[action], cv, masks[s]) for s in ('all', 'easy', 'hard')},
                zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((aa[action][zero] > 0).sum())))
        for s in ('all', 'easy', 'hard'):
            assert metrics['original_neural']['ADE_vs_floor'][s] == old['neural_vs_damping'][group+'_ranked_original'][s]
            controls += 1
        ledgers = {s: floor_ledger(cv, d, na, nw, sites, expected_scenes=roster, subset=m,
            resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed']) for s, m in masks.items()}
        for s, m in masks.items():
            verify_ledger(cv, d, na, nw, sites, m, ledgers[s], cfg)
        rebase_vs_original = {s: metric(aa['rebased_neural'], aa['original_neural'], m) for s, m in masks.items()}
        support = {site: dict(indexed_rows=int((sites == site).sum()),
            label_count_histogram=np.bincount(valid[sites == site].sum(1), minlength=13).tolist(),
            zero_cv_rows=int(((sites == site) & zero).sum()),
            zero_cv_label_counts=valid[(sites == site) & zero].sum(1).tolist(),
            zero_cv_endpoint_rows=int(valid[(sites == site) & zero, -1].sum())) for site in roster}
        r = dict(identity=identity, group=group, result_source='fresh_run_diagnostic_from_cached_verified_frozen_predictions',
            indexed_rows=len(ids), locality_roster=roster, support=support,
            neural_decision_sha256=parent.array_hash(ids, ns), floor_decision_sha256=parent.array_hash(ids, use),
            neural_switch_rate=float(ns.mean()), floor_switch_rate=float(use.mean()),
            rebase_changes_rows=int((~ns & use).sum()), metrics=metrics, ledgers=ledgers,
            rebase_vs_original=rebase_vs_original,
            checks=dict(all_passed=True, coordinate_arrays=4, decision_arrays=2,
                        exact_parent_metrics=controls, metric_reductions=reductions, independent_ledgers=8),
            deployment_changed=False, reserved_roles_opened=False, future_input=False)
        immutable_json(destination, r); done.append(group)
        beat('group_complete', mode=mode, group=group, seconds=time.monotonic()-t0, checks=r['checks'])
        if pilot:
            return
    assert not pairs and len(done) == 18
    parent.assert_identity(lineage)
    refs = {g: dict(path=str((PRIVATE/mode/'groups'/f'{g}.json').relative_to(ROOT)),
                   sha256=digest(PRIVATE/mode/'groups'/f'{g}.json')) for g in done}
    immutable_json(PRIVATE/mode/'complete.json', dict(identity=identity, groups=refs, all_passed=True))
    beat('mode_complete', mode=mode, groups=len(done), seconds=time.monotonic()-t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('batch', 'fitting'), required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.torch.set_num_threads(4); parent.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('started', mode=args.mode, pilot=args.pilot, resume=args.resume)
        run(args.mode, pilot=args.pilot, resume=args.resume)
        beat('phase_complete', mode=args.mode, pilot=args.pilot)


if __name__ == '__main__':
    main()
