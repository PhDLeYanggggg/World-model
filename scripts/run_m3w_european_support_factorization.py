"""Frozen-source support factorization with current-frame matched decisions."""
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
from scripts import run_m3w_european_causal_abstention as prior
from src.world_model.m3w_support_factorization import decisions, independent_verify, partition_ledger
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np
import torch

BASE = prior.parent
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_support_factorization_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_support_factorization_v1'
CONFIG = 'configs/m3w_european_support_factorization_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_support_factorization.py', 'src/world_model/m3w_support_factorization.py',
         'tests/test_m3w_support_factorization.py', 'tests/test_m3w_support_factorization_protocol.py',
         'outputs/publication_readiness_2026_09/european_support_factorization_v1/registration.md')
digest, immutable_json, artifact, array_hash = prior.digest, prior.immutable_json, prior.artifact, prior.array_hash
OLD_NAMES = dict(stop='stop', joint='combined', joint_risk='combined_risk', joint_random='combined_random')


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    BASE.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def validate(cfg):
    for k, v in dict(modes=['batch', 'fitting'], parents=['cv_targets', 'floor_both'],
        guards=['history', 'disagreement', 'separate', 'joint'], controls=['risk', 'random'],
        seeds=[17, 29, 43], events=['all', 'easy'], groups=36, views=936,
        bootstrap_resamples=3000, bootstrap_seed=39271, support_quantiles=[.01, .99],
        minimum_fitting_rows=32, minimum_source_count=2).items():
        if cfg[k] != v: raise ValueError('Changed registered support comparison: '+k)
    for k in ('threshold_refit', 'new_neural_training', 'producer_retraining', 'reserved_roles_opened',
              'deployment_changed', 'stage5c_executed', 'smc_enabled'):
        if cfg[k]: raise ValueError('Fixed-source development comparison only')


def load(mode):
    cfg = json.loads((ROOT/CONFIG).read_text()); validate(cfg)
    prior.ensure_both_frozen()
    acfg, pcfg, ctx, pid, aid = prior.load(mode)
    if digest(prior.PUBLIC/'summary_metrics.json') != cfg['prior_summary_sha256']:
        raise ValueError('Bound preceding evidence changed')
    identity = dict(mode=mode, bindings={f: digest(ROOT/f) for f in FILES},
        prior_identity=artifact(prior.PRIVATE/mode/'identity.json'),
        prior_decisions=artifact(prior.PRIVATE/mode/'decisions_complete.json'),
        prior_evaluation=artifact(prior.PRIVATE/mode/'evaluation_complete.json'),
        prior_summary_sha256=cfg['prior_summary_sha256'])
    immutable_json(PRIVATE/mode/'identity.json', identity)
    return cfg, acfg, pcfg, ctx, pid, identity


def causal_group(acfg, data, design, a, d, mode, name):
    x, states, fitted, query = prior.inputs(acfg, data, design, a, d)
    path = prior.PRIVATE/mode/'decisions'/(name+'.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt['fitted_support'] != fitted or receipt['inference_sha256'] != array_hash(design['held_ids'], x, states, query):
        raise ValueError('Prior fitting-only boxes or causal rows changed')
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(design['held_ids'], z['ids'])
        old = {k: z[k].copy() for k in receipt['choices_sha256']}
    return x, states, fitted, query, old


def decide(mode):
    cfg, acfg, pcfg, ctx, pid, identity = load(mode); data = ctx[2]; refs = []
    for name, fold, seed, event, design, a, d, _, envelope, floor, provenance in BASE.groups(pcfg, ctx, pid):
        if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below 10GiB; retain decisions')
        ids = design['held_ids']; x, states, fitted, query, old = causal_group(acfg, data, design, a, d, mode, name)
        out = {}; reasons = {}
        for family in cfg['parents']:
            _, risk = prior.original_and_risk(mode, name, family, ids, pid)
            choices, reason = decisions(old[family+'__stop'], x, states, fitted, query, risk, ids, seed)
            for new, previous in OLD_NAMES.items(): np.testing.assert_array_equal(choices[new], old[family+'__'+previous])
            out.update({family+'__'+k: v for k, v in choices.items()}); reasons[family+'__reason'] = reason
        path = PRIVATE/mode/'decisions'/(name+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z['ids'])
                for k, v in {**out, **reasons}.items(): np.testing.assert_array_equal(v, z[k])
        else: np.savez(path, ids=ids, query=query, **out, **reasons)
        receipt = dict(identity=identity, group=name, artifact=artifact(path),
            causal_sha256=array_hash(ids, x, states, query), fitted_support=fitted,
            old_control_arrays_exact=8, choices_sha256={k: array_hash(ids, v) for k, v in out.items()})
        immutable_json(path.with_suffix('.json'), receipt); refs.append(artifact(path.with_suffix('.json')))
        beat('decisions_frozen', mode=mode, group=name, policies=len(out), rows=len(ids))
    assert len(refs) == 18
    immutable_json(PRIVATE/mode/'decisions_complete.json', dict(identity=identity, groups=refs, all_passed=True,
        new_policy_outcomes_read=False, support_thresholds_refitted=False))


def ensure_both_frozen():
    prior.ensure_both_frozen()
    for mode in ('batch', 'fitting'):
        r = json.loads((PRIVATE/mode/'decisions_complete.json').read_text())
        if not r['all_passed'] or len(r['groups']) != 18: raise ValueError('Both decision banks must be complete')
        for f, sha in r['identity']['bindings'].items():
            if digest(ROOT/f) != sha: raise ValueError('Frozen binding changed')
        for key in ('prior_identity', 'prior_decisions', 'prior_evaluation'):
            if artifact(ROOT/r['identity'][key]['path']) != r['identity'][key]: raise ValueError('Prior bank changed')
        for ref in r['groups']:
            if artifact(ROOT/ref['path']) != ref: raise ValueError('Changed decision receipt')
            rr = json.loads((ROOT/ref['path']).read_text())
            if rr['identity'] != r['identity'] or artifact(ROOT/rr['artifact']['path']) != rr['artifact']:
                raise ValueError('Changed decision bank')


def evaluate(mode, resume=False):
    ensure_both_frozen(); cfg, acfg, pcfg, ctx, pid, identity = load(mode); data = ctx[2]; refs = []
    for name, fold, seed, event, design, a, d, _, envelope, floor, provenance in BASE.groups(pcfg, ctx, pid):
        path = PRIVATE/mode/'evaluation'/(name+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing evaluation requires --resume')
            r = json.loads(path.read_text()); assert r['verified'] and r['identity'] == identity
            refs.append(artifact(path)); continue
        ids = design['held_ids']; x, states, fitted, query, old = causal_group(acfg, data, design, a, d, mode, name)
        archive = PRIVATE/mode/'decisions'/(name+'.npz'); receipt = json.loads(archive.with_suffix('.json').read_text())
        assert receipt['fitted_support'] == fitted and receipt['causal_sha256'] == array_hash(ids, x, states, query)
        with np.load(archive, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); np.testing.assert_array_equal(query, z['query'])
            choices = {k: z[k].copy() for k in receipt['choices_sha256']}
            reasons = {p: z[p+'__reason'].copy() for p in cfg['parents']}
        replay_count = 0
        for family in cfg['parents']:
            _, risk = prior.original_and_risk(mode, name, family, ids, pid)
            group = {k.split('__')[1]: v for k, v in choices.items() if k.startswith(family+'__')}
            replay_count += independent_verify(group, reasons[family], old[family+'__stop'], x, states, fitted, query, risk, ids, seed)
            for new, previous in OLD_NAMES.items(): np.testing.assert_array_equal(group[new], old[family+'__'+previous])
        # No new outcome read until both banks freeze and causal decisions independently replay.
        sites = data['sites'][ids]; roster = sorted(set(sites)); errors = []
        for prediction in (d[ids].astype(float)+data['origin'][ids, None], a['p'][ids].astype(float)+data['origin'][ids, None]):
            pair = native_errors(prediction, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            alternate = BASE.cross.independent.coordinate_errors(prediction, data['target_eval'][ids], data['valid'][ids])
            for v, w in zip(pair, alternate): BASE.cross.independent.close(v, w)
            errors.append(pair)
        (da, df), (na, nf) = errors; cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']),
                     hard=cv >= design['hard_cut'], complete=data['valid'][ids].all(1))
        checks = 0
        def metric(error, reference, mask):
            nonlocal checks
            v = paired_scene_metrics(error[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
            BASE.cross.independent.check_metric(error, reference, sites, mask, v, cfg); checks += 1
            return v
        selected = {k: np.where(v, na, da) for k, v in choices.items()}; views = {}; old_checks = 0
        oldpath = prior.PRIVATE/mode/'evaluation'/(name+'.json')
        done = json.loads((prior.PRIVATE/mode/'evaluation_complete.json').read_text())
        if artifact(oldpath) not in done['groups']: raise ValueError('Unbound prior metrics')
        previous = json.loads(oldpath.read_text())['views']
        for key, use in choices.items():
            family, guard = key.split('__'); err = selected[key]
            v = dict(ADE_vs_floor={s: metric(err, da, m) for s, m in masks.items()},
                ADE_vs_stop={s: metric(err, selected[family+'__stop'], m) for s, m in masks.items()},
                FDE_vs_floor=metric(np.where(use, nf, df), df, masks['all']),
                easy_vs_CV=metric(err, cv, masks['easy']), switched_rows=int(use.sum()), switch_rate=float(use.mean()),
                unknown_ADE_switches=int((use & ~np.isfinite(na)).sum()),
                zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err > 0)).sum())))
            if guard in cfg['guards']:
                v['matched_controls'] = {c: {s: metric(err, selected[key+'_'+c], m) for s, m in masks.items()} for c in cfg['controls']}
            if guard in ('history', 'disagreement', 'separate'):
                v['ADE_vs_joint'] = {s: metric(err, selected[family+'__joint'], m) for s, m in masks.items()}
            if guard in OLD_NAMES:
                p = previous[family+'__'+OLD_NAMES[guard]]
                assert v['ADE_vs_floor'] == p['ADE_vs_floor']
                assert v['FDE_vs_floor'] == p['FDE_vs_floor'] and v['easy_vs_CV'] == p['ADE_vs_CV_easy']
                old_checks += 6
            views[key] = v
        partitions = {p: {s: partition_ledger(reasons[p], na, da, sites, m) for s, m in masks.items()} for p in cfg['parents']}
        for p, subsets in partitions.items():
            for s, localities in subsets.items():
                ledger = prior.removal_ledger(choices[p+'__joint'], choices[p+'__stop'], na, da, sites, {s: masks[s]})[s]
                for source, entry in localities.items():
                    categories = list(entry['categories'].values())[1:5]
                    for k, oldkey in (('harm', 'avoided_harm'), ('benefit', 'lost_benefit')):
                        np.testing.assert_allclose(sum(c[k] for c in categories), ledger[source][oldkey], rtol=1e-12, atol=1e-9)
        r = dict(identity=identity, group=name, verified=True, views=views, partitions=partitions,
            saved_decisions_verified=replay_count, old_decision_arrays_exact=8, independent_coordinate_arrays=4,
            independent_metric_reductions=checks, old_metrics_exact=old_checks, partition_equalities=128,
            result_source='fresh_run_fixed_box_factorization', new_neural_training=False, deployment_changed=False)
        immutable_json(path, r); refs.append(artifact(path))
        beat('group_evaluated', mode=mode, group=name, policies=len(views), metric_checks=checks)
    assert len(refs) == 18
    immutable_json(PRIVATE/mode/'evaluation_complete.json', dict(identity=identity, groups=refs, all_passed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('batch', 'fitting'), required=True)
    parser.add_argument('--phase', choices=('prepare', 'decide', 'evaluate'), required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('phase_started', mode=args.mode, phase=args.phase)
        if args.phase == 'prepare': load(args.mode)
        elif args.phase == 'decide': decide(args.mode)
        else: evaluate(args.mode, args.resume)
        beat('phase_complete', mode=args.mode, phase=args.phase)


if __name__ == '__main__': main()
