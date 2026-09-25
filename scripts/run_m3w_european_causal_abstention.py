"""Freeze then evaluate past-only abstention and same-frame intervention controls."""
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
from scripts import run_m3w_european_floor_relative as parent
from src.world_model.m3w_causal_abstention import causal_features, fit_support, query_ids, variants, verify_variants
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_causal_abstention_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_causal_abstention_v1'
CONFIG = 'configs/m3w_european_causal_abstention_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_causal_abstention.py', 'src/world_model/m3w_causal_abstention.py',
         'tests/test_m3w_causal_abstention.py', 'tests/test_m3w_causal_abstention_protocol.py',
         'outputs/publication_readiness_2026_09/european_causal_abstention_v1/registration.md')
digest, immutable_json, array_hash, artifact = parent.digest, parent.immutable_json, parent.array_hash, parent.artifact


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def validate(cfg):
    for k, v in dict(modes=['batch', 'fitting'], parents=['cv_targets', 'floor_both'], seeds=[17, 29, 43],
                     events=['all', 'easy'], groups=36, views=720, guards=['stop', 'support', 'combined'],
                     matched_controls=['risk', 'random'], quantiles=[.01, .99], minimum_rows=32,
                     minimum_sources=2, query='recording_and_observed_current_frame',
                     bootstrap_resamples=3000, bootstrap_seed=39271).items():
        if cfg[k] != v: raise ValueError('Changed fixed comparison: '+k)
    for k in ('new_neural_training', 'threshold_search', 'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'):
        if cfg[k]: raise ValueError('Development-only fixed abstention comparison')


def load(mode):
    cfg = json.loads((ROOT/CONFIG).read_text()); validate(cfg)
    parent.ensure_both_frozen()
    pcfg, ctx, previous, pid = parent.load(mode)
    if digest(parent.PUBLIC/'summary_metrics.json') != cfg['prior_summary_sha256']:
        raise ValueError('Prior reported state changed')
    identity = dict(mode=mode, bindings={f: digest(ROOT/f) for f in FILES},
                    parent_identity=artifact(parent.PRIVATE/mode/'identity.json'),
                    parent_decisions=artifact(parent.PRIVATE/mode/'decisions_complete.json'),
                    parent_evaluation=artifact(parent.PRIVATE/mode/'evaluation_complete.json'),
                    prior_summary_sha256=cfg['prior_summary_sha256'])
    immutable_json(PRIVATE/mode/'identity.json', identity)
    return cfg, pcfg, ctx, pid, identity


def inputs(cfg, data, design, a, d):
    train, ids = design['train_ids'], design['held_ids']
    if set(data['sites'][train]) & set(data['sites'][ids]):
        raise ValueError('Fitting and outer localities overlap')
    x, states = causal_features(data['history'], a['p'], d)
    # Label availability restricts FITTING support only; held labels never gate a query.
    known = np.isfinite(data['baseline_ade'][train, 1])
    fitted = fit_support(x[train], states[train], data['sites'][train], known,
        minimum_rows=cfg['minimum_rows'], quantiles=tuple(cfg['quantiles']))
    q = query_ids(data['recordings'][ids], data['frames'][ids])
    return x[ids], states[ids], fitted, q


def original_and_risk(mode, name, key, ids, pid):
    with np.load(parent.PRIVATE/mode/'decisions'/(name+'.npz'), allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids']); original = z[key].copy()
    reference = 'cv' if key == 'cv_targets' else 'floor'
    risk = parent.head_scores(mode, name+'_'+reference+'_risk', ids, pid)
    return original, risk


def decide(mode):
    cfg, pcfg, ctx, pid, identity = load(mode); data = ctx[2]; records = []
    for name, fold, seed, event, design, a, d, _, envelope, floor, provenance in parent.groups(pcfg, ctx, pid):
        if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below 10GiB; retain saved decisions')
        ids = design['held_ids']; x, states, fitted, q = inputs(cfg, data, design, a, d)
        out = {}
        for key in cfg['parents']:
            original, risk = original_and_risk(mode, name, key, ids, pid)
            out.update({key+'__'+k: v for k, v in variants(original, x, states, fitted, q, risk, ids, seed).items()})
        path = PRIVATE/mode/'decisions'/(name+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(ids, z['ids'])
                for key, value in out.items(): np.testing.assert_array_equal(value, z[key])
        else: np.savez(path, ids=ids, query=q, state=states, **out)
        receipt = dict(identity=identity, group=name, fitted_support=fitted, artifact=artifact(path),
            inference_sha256=array_hash(ids, x, states, q), choices_sha256={k: array_hash(ids, v) for k, v in out.items()},
            fitting_ids_sha256=array_hash(design['train_ids']), outer_ids_sha256=array_hash(ids),
            fitting_sources=sorted(set(data['sites'][design['train_ids']])), outer_sources=sorted(set(data['sites'][ids])))
        immutable_json(path.with_suffix('.json'), receipt); records.append(artifact(path.with_suffix('.json')))
        beat('decisions_frozen', mode=mode, group=name, rows=len(ids), indexed_queries=int(q.max()+1))
    assert len(records) == 18
    immutable_json(PRIVATE/mode/'decisions_complete.json', dict(identity=identity, groups=records,
        all_passed=True, new_policy_outcomes_read=False, result_source='fresh_run_causal_support_and_decisions'))


def ensure_both_frozen():
    parent.ensure_both_frozen()
    for mode in ('batch', 'fitting'):
        r = json.loads((PRIVATE/mode/'decisions_complete.json').read_text())
        if not r['all_passed'] or len(r['groups']) != 18: raise ValueError('Both complete banks required')
        for f, sha in r['identity']['bindings'].items():
            if digest(ROOT/f) != sha: raise ValueError('Frozen source binding changed')
        for k in ('parent_identity', 'parent_decisions', 'parent_evaluation'):
            if artifact(ROOT/r['identity'][k]['path']) != r['identity'][k]: raise ValueError('Parent bank changed')
        for ref in r['groups']:
            if artifact(ROOT/ref['path']) != ref: raise ValueError('Decision receipt changed')
            receipt = json.loads((ROOT/ref['path']).read_text())
            if receipt['identity'] != r['identity'] or artifact(ROOT/receipt['artifact']['path']) != receipt['artifact']:
                raise ValueError('Decisions changed')


def removal_ledger(use, original, neural_error, floor_error, sites, masks):
    removed = original & ~use; delta = neural_error-floor_error
    out = {}
    for subset, mask in masks.items():
        rows = {}
        for site in sorted(set(sites)):
            known = mask & (sites == site) & np.isfinite(delta) & np.isfinite(floor_error)
            cut = known & removed; denom = float(floor_error[known].sum())
            avoided = float(np.maximum(delta[cut], 0).sum()); lost = float(np.maximum(-delta[cut], 0).sum())
            rows[site] = dict(removed_indexed=int((mask & removed & (sites == site)).sum()),
                removed_known=int(cut.sum()), avoided_harm=avoided, lost_benefit=lost,
                floor_error_sum=denom, gain_change_pp=100*(avoided-lost)/denom if denom > 0 else None)
        out[subset] = rows
    return out


def evaluate(mode, resume=False):
    ensure_both_frozen(); cfg, pcfg, ctx, pid, identity = load(mode); data = ctx[2]; refs = []
    for name, fold, seed, event, design, a, d, _, envelope, floor, provenance in parent.groups(pcfg, ctx, pid):
        path = PRIVATE/mode/'evaluation'/(name+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing result requires --resume')
            row = json.loads(path.read_text()); assert row['identity'] == identity and row['verified']
            refs.append(artifact(path)); continue
        ids = design['held_ids']; x, states, fitted, q = inputs(cfg, data, design, a, d)
        archive = PRIVATE/mode/'decisions'/(name+'.npz')
        receipt = json.loads(archive.with_suffix('.json').read_text())
        assert receipt['fitted_support'] == fitted
        assert receipt['inference_sha256'] == array_hash(ids, x, states, q)
        with np.load(archive, allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); np.testing.assert_array_equal(q, z['query']); np.testing.assert_array_equal(states, z['state'])
            choices = {k: z[k].copy() for k in receipt['choices_sha256']}
        replay_count = 0
        for key in cfg['parents']:
            original, risk = original_and_risk(mode, name, key, ids, pid)
            sub = {k.split('__')[1]: v for k, v in choices.items() if k.startswith(key+'__')}
            replay_count += verify_variants(sub, original, x, states, fitted, q, risk, ids, seed)
        # First new policy outcome access occurs only after both frozen banks and causal replay.
        sites = data['sites'][ids]; roster = sorted(set(sites)); errors = []
        for prediction in (d[ids].astype(float)+data['origin'][ids, None], a['p'][ids].astype(float)+data['origin'][ids, None]):
            pair = native_errors(prediction, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            alt = parent.cross.independent.coordinate_errors(prediction, data['target_eval'][ids], data['valid'][ids])
            for v, w in zip(pair, alt): parent.cross.independent.close(v, w)
            errors.append(pair)
        (da, df), (na, nf) = errors; cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']),
                     hard=cv >= design['hard_cut'], complete=data['valid'][ids].all(1))
        metrics_checked = 0
        def metric(error, reference, mask):
            nonlocal metrics_checked
            v = paired_scene_metrics(error[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
            parent.cross.independent.check_metric(error, reference, sites, mask, v, cfg)
            metrics_checked += 1
            return v
        selected = {k: np.where(use, na, da) for k, use in choices.items()}; views = {}; old_matches = 0
        prior_path = parent.PRIVATE/mode/'evaluation'/(name+'.json')
        prior_manifest = json.loads((parent.PRIVATE/mode/'evaluation_complete.json').read_text())
        if artifact(prior_path) not in prior_manifest['groups']: raise ValueError('Unbound preceding result')
        prior = json.loads(prior_path.read_text())['views']
        for key, use in choices.items():
            policy, variant = key.split('__'); original = choices[policy+'__original']; err = selected[key]
            view = dict(ADE_vs_floor={s: metric(err, da, m) for s, m in masks.items()},
                ADE_vs_original={s: metric(err, selected[policy+'__original'], masks[s]) for s in ('all', 'easy', 'hard')},
                ADE_vs_CV_easy=metric(err, cv, masks['easy']),
                FDE_vs_floor=metric(np.where(use, nf, df), df, masks['all']),
                switch_rate=float(use.mean()), switched_rows=int(use.sum()),
                unknown_ADE_switches=int((use & ~np.isfinite(na)).sum()),
                stopped_switches=int((use & (states == 1)).sum()),
                zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err > 0)).sum())),
                removal=removal_ledger(use, original, na, da, sites, masks))
            if variant in cfg['guards']:
                view['matched_controls'] = {control: {s: metric(err, selected[key+'_'+control], masks[s])
                    for s in ('all', 'easy', 'hard', 'complete')} for control in cfg['matched_controls']}
            if variant == 'original':
                assert view['ADE_vs_floor'] == prior[policy]['ADE_vs_floor']; old_matches += 4
                assert view['FDE_vs_floor'] == prior[policy]['FDE_vs_floor']; old_matches += 1
                assert view['ADE_vs_CV_easy'] == prior[policy]['ADE_vs_CV']['easy']; old_matches += 1
            views[key] = view
        sizes = np.bincount(q); eligible = {p: np.bincount(q[choices[p+'__original']], minlength=len(sizes)) for p in cfg['parents']}
        row = dict(identity=identity, group=name, views=views, verified=True, saved_decisions_verified=replay_count,
            independent_coordinate_arrays=4, independent_metric_reductions=metrics_checked,
            original_metrics_exact=old_matches, result_source='fresh_run_fixed_forecast_guard_evaluation',
            no_new_neural_training=True, deployment_changed=False,
            queries=dict(indexed=len(sizes), multi_agent=int((sizes >= 2).sum()),
                         multi_original_switch={p: int((v >= 2).sum()) for p, v in eligible.items()}))
        immutable_json(path, row); refs.append(artifact(path))
        beat('group_evaluated', mode=mode, group=name, policies=len(views), metric_checks=metrics_checked)
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
