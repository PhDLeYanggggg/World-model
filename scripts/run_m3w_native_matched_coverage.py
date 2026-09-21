"""Frozen-score matched-count research control; not calibration or deployment."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_gain_harm import load as load_parent
from scripts.run_m3w_native_forecast import array_hash, assert_current, immutable_json, json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_matched_coverage import matched_policies, paired_scene_contrast
import numpy as np
import torch

CONFIG = 'configs/m3w_native_matched_coverage_v1.json'
CODE = ('scripts/run_m3w_native_matched_coverage.py',
        'src/evaluation/m3w_native_matched_coverage.py', 'tests/test_m3w_native_matched_coverage.py')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    parent, data, _, identity, views = load_parent()
    if file_digest(ROOT/reg['parent_analysis']) != reg['parent_analysis_sha256']:
        raise ValueError('Changed parent score analysis')
    analysis = json.loads((ROOT/reg['parent_analysis']).read_text())
    v = json.loads((ROOT/parent['reports']/'independent_verification.json').read_text())
    if (analysis['identity'] != identity or not v['all_checks_passed']
            or v['analysis_sha256'] != reg['parent_analysis_sha256']
            or reg['budgets'] != ['positive_gain', 'harm_fraction_0p1']
            or reg['policies'] != ['asym_rule', 'mse_ratio', 'mse_net_gain', 'asym_net_gain', 'ridge_raw_net_gain', 'hash_control']
            or reg['primary_contrast'] != ['asym_rule', 'mse_ratio']
            or any(reg[k] for k in ('training', 'threshold_search', 'deployment', 'new_closed_role_readout'))):
        raise ValueError('Only verified, fixed source-only matched-coverage design permitted')
    bindings = dict(identity['source_bindings'])
    for path in (CONFIG, reg['registration'], reg['parent_analysis'], *CODE,
                 parent['reports']+'/verification.json', parent['reports']+'/independent_verification.json'):
        bindings[path] = file_digest(ROOT/path)
    for r in analysis['score_archives'] + analysis['training']:
        path, sha = (r['path'], r['sha256']) if 'path' in r else (r['checkpoint'], r['checkpoint_sha256'])
        if file_digest(ROOT/path) != sha:
            raise ValueError('Changed frozen head/score archive')
        bindings[path] = sha
    return reg, parent, data, views, analysis, dict(source_bindings=bindings, parent_identity=identity,
        scope=reg['scope'], numpy=np.__version__, torch=torch.__version__)


def evaluate(reg, parent, data, views, old, identity, beat):
    n = len(data['sites'])
    cv, cf = native_errors(data['geometry'][:, 332:356].reshape(-1, 12, 2), data['target'], data['valid'], data['scale'])
    complete = data['valid'].all(1)
    zero = complete & (cv == 0)
    masks = dict(all=np.ones(n, bool), complete=complete, zero_CV=zero,
        hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    arrays, receipts, random_views = {}, [], []
    anchor_checks = 0
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        sb = next(r for r in old['score_archives'] if r['view'] == key)
        with np.load(ROOT/sb['path'], allow_pickle=False) as z:
            ids = z['ids'].copy()
            scores = {name:z[name].copy() for name in ('underharm4', 'mse', 'ridge_raw')}
        record = meta['outer_prediction']['prediction']
        with np.load(ROOT/record['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids'])
            pred = z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        same = np.all(pred == baseline, axis=(1, 2))
        ne, nf = native_errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        known = np.isfinite(ne)
        harm = np.maximum(ne-cv[ids], 0)
        r = next(r for r in old['training'] if r['view'] == key and r['arm'] == 'ridge')
        pr = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks['hard'][ids] = cv[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv[ids] > 0) & (cv[ids] <= pr['positive_easy_cut'])
        for budget in reg['budgets']:
            policies = matched_policies(scores, same, ids, site=site, seed=seed, budget=budget)
            k = int(policies['asym_rule'].sum())
            if list(policies) != reg['policies'] or any(int(u.sum()) != k for u in policies.values()):
                raise ValueError('Changed policy roster or unequal intervention budget')
            p_random = k/int((~same).sum()) if (~same).any() else 0.
            for policy, use in policies.items():
                chosen_ade, chosen_fde = np.where(use, ne, cv[ids]), np.where(use, nf, cf[ids])
                index = budget, policy, seed
                if index not in arrays:
                    arrays[index] = dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan), use=np.zeros(n, bool), seen=np.zeros(n, bool))
                storage = arrays[index]
                if storage['seen'][ids].any():
                    raise ValueError('Repeated held query')
                storage['ade'][ids], storage['fde'][ids] = chosen_ade, chosen_fde
                storage['use'][ids], storage['seen'][ids] = use, True
                selected = use & known
                zl = zero[ids]
                source_ade = old['summary']['underharm4'][budget]['seeds'][str(seed)]['ADE']['by_scene'][site]
                if policy == 'asym_rule':
                    np.testing.assert_allclose(chosen_ade[known].mean(), source_ade['model_error'], rtol=1e-12, atol=1e-12)
                    anchor_checks += 1
                receipts.append(dict(view=key, site=site, seed=seed, budget=budget, policy=policy,
                    indexed_rows=len(ids), eligible_rows=int((~same).sum()), selected_rows=k,
                    selected_ids_sha256=array_hash(ids[use]), known_selected_rows=int(selected.sum()),
                    unknown_selected_rows=int((use & ~known).sum()),
                    realized_mean_selected_harm=float(harm[selected].mean()) if selected.any() else None,
                    realized_selected_harm_sum=float(harm[selected].sum()),
                    zero_CV_selected_rows=int((use & zl).sum()), zero_CV_harmed_rows=int((chosen_ade[zl] > 0).sum()),
                    zero_CV_max_harm=float(chosen_ade[zl].max()) if zl.any() else None))
            # Uniform sampling of exactly K has marginal inclusion K/N. Evaluate
            # expected costs, not a fractional path or a tail guarantee.
            exp_ade = cv[ids]+p_random*np.where(~same, ne-cv[ids], 0)
            exp_fde = cf[ids]+p_random*np.where(~same, nf-cf[ids], 0)
            zero_harmed_eligible = zero[ids] & ~same & (ne > 0)
            groups = {}
            for name, whole in masks.items():
                use = whole[ids] & known
                ref = float(cv[ids][use].mean()) if use.any() else None
                val = float(exp_ade[use].mean()) if use.any() else None
                groups[name] = dict(rows=int(use.sum()), expected_ADE=val, reference_ADE=ref,
                    gain_percent=100*(1-val/ref) if ref is not None and ref > 0 else None)
            random_views.append(dict(view=key, budget=budget, inclusion_probability=p_random,
                expected_unknown_selected_rows=float(p_random*(~same & ~known).sum()),
                expected_zero_CV_harmed_rows=float(p_random*zero_harmed_eligible.sum()),
                expected_zero_CV_added_ADE_sum=float(p_random*ne[zero_harmed_eligible].sum()),
                expected_FDE=float(np.nanmean(exp_fde)), groups=groups,
                status='analytic_uniform_exact_K_expectation_not_realized_policy_tail_or_safety_certificate'))
        beat(state='matched_view_evaluated', view=key, rows=len(ids))
    def measure(a, b, mask):
        return paired_scene_metrics(a[mask], b[mask], data['sites'][mask], expected_scenes=parent['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    summary, table = {}, []
    for budget in reg['budgets']:
        summary[budget] = {}
        for policy in reg['policies']:
            seed_results = {}
            for seed in parent['seeds']:
                a = arrays[budget, policy, seed]
                if not a['seen'].all():
                    raise ValueError('Missing held-scene queries')
                seed_results[str(seed)] = dict(ADE=measure(a['ade'], cv, masks['all']), FDE=measure(a['fde'], cf, masks['all']),
                    subsets={k:measure(a['ade'], cv, mask) for k,mask in masks.items() if k != 'all'},
                    intervention_rows=int(a['use'].sum()), intervention_rate=float(a['use'].mean()),
                    zero_CV_harmed_rows=int((a['ade'][zero] > 0).sum()),
                    zero_CV_max_absolute_harm=float(a['ade'][zero].max()),
                    unknown_selected_rows=int((a['use'] & ~np.isfinite(cv)).sum()))
            mean_ade = np.mean([arrays[budget, policy, s]['ade'] for s in parent['seeds']], axis=0)
            r = dict(seeds=seed_results, ADE=measure(mean_ade, cv, masks['all']),
                subsets={k:measure(mean_ade, cv, mask) for k,mask in masks.items() if k != 'all'})
            summary[budget][policy] = r
            table.append(dict(budget=budget, policy=policy, ADE_gain_percent=r['ADE']['equal_scene_gain_percent'],
                CI_low=r['ADE']['scene_bootstrap_ci95'][0], CI_high=r['ADE']['scene_bootstrap_ci95'][1],
                worst_scene_gain_percent=r['ADE']['worst_scene_gain_percent'], hard_gain_percent=r['subsets']['hard']['equal_scene_gain_percent'],
                positive_easy_degradation_percent=-r['subsets']['positive_easy']['equal_scene_gain_percent'],
                switch_percent=100*np.mean([v['intervention_rate'] for v in seed_results.values()]),
                zero_CV_harmed_query_seed_instances=sum(v['zero_CV_harmed_rows'] for v in seed_results.values()),
                zero_CV_max_absolute_harm=max(v['zero_CV_max_absolute_harm'] for v in seed_results.values()),
                unknown_selected_query_seed_instances=sum(v['unknown_selected_rows'] for v in seed_results.values())))
    contrasts = {}
    for budget in reg['budgets']:
        contrasts[budget] = {}
        anchor = summary[budget]['asym_rule']
        for policy in reg['policies'][1:]:
            other = summary[budget][policy]
            contrasts[budget][policy] = {}
            for subset in ('all', 'hard', 'positive_easy'):
                first = anchor['ADE'] if subset == 'all' else anchor['subsets'][subset]
                second = other['ADE'] if subset == 'all' else other['subsets'][subset]
                aa = [first['by_scene'][s]['gain_percent'] for s in parent['sites']]
                bb = [second['by_scene'][s]['gain_percent'] for s in parent['sites']]
                contrasts[budget][policy][subset] = paired_scene_contrast(aa, bb,
                    resamples=reg['bootstrap_resamples'], seed=reg['bootstrap_seed'])
    return dict(identity=identity, result_source='fresh_run_matched_coverage_arithmetic_cached_verified_forecasts_and_cost_heads',
        summary=summary, paired_contrasts=contrasts, table=table, policy_receipts=receipts,
        uniform_random_expectations=random_views, anchor_reproductions=anchor_checks,
        exact_count_checks=len(receipts), source_query_rows=n, seed_prediction_instances=n*len(parent['seeds']),
        training=False, threshold_search=False, independent_confirmation=False, risk_calibration=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, parent, data, views, old, identity = load()
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    if args.audit_only:
        beat(state='preflight_pass', bindings=len(identity['source_bindings']), views=len(views), rows=len(data['sites'])); return
    if args.verify and not (public/'analysis.json').exists():
        raise ValueError('Verify cannot create missing analysis')
    immutable_json(root/'identity.json', identity)
    result = evaluate(reg, parent, data, views, old, identity, beat)
    assert_current(identity)
    immutable_json(public/'analysis.json', result)
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(result['table'][0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(result['table'])
    path = public/'results.csv'
    if path.exists():
        if path.read_bytes() != buf.getvalue().encode():
            raise ValueError('Changed existing result table')
    else:
        if args.verify:
            raise ValueError('Missing table in replay')
        path.write_bytes(buf.getvalue().encode())
    if args.verify:
        immutable_json(public/'verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            anchor_reproductions=result['anchor_reproductions'], exact_count_checks=result['exact_count_checks'],
            selected_id_hashes_recomputed=len(result['policy_receipts']), all_checks_passed=True, new_training=False))
    beat(state='verified_complete' if args.verify else 'completed', policies=len(result['policy_receipts']), training=False)


if __name__ == '__main__':
    main()
