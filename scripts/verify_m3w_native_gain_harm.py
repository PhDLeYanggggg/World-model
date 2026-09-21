"""Independent arithmetic and post-fit failure slices; no fitting or policy search."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_gain_harm import load
from scripts.run_m3w_native_forecast import immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
import numpy as np
import torch


def errors(pred, target, valid, scale):
    """Reduce twelve masked scalar distances without the training metric helper."""
    total = np.zeros(len(pred), dtype=np.float64)
    count = np.zeros(len(pred), dtype=np.int64)
    endpoint = np.full(len(pred), np.nan)
    for t in range(pred.shape[1]):
        use = valid[:, t]
        dx = pred[use, t, 0].astype(float)-target[use, t, 0].astype(float)
        dy = pred[use, t, 1].astype(float)-target[use, t, 1].astype(float)
        distance = np.sqrt(dx*dx+dy*dy)*scale[use]
        total[use] += distance
        count[use] += 1
        if t == pred.shape[1]-1:
            endpoint[use] = distance
    mean = np.full(len(pred), np.nan)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, endpoint


def check_mean(model, baseline, mask, reference):
    use = mask & np.isfinite(model) & np.isfinite(baseline)
    assert int(use.sum()) == reference['rows']
    if not use.any():
        assert reference['model_error'] is None and reference['reference_error'] is None
        return
    mm, rr = float(model[use].mean()), float(baseline[use].mean())
    np.testing.assert_allclose(mm, reference['model_error'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rr, reference['reference_error'], rtol=1e-12, atol=1e-12)
    if rr == 0:
        assert reference['gain_percent'] is None
    else:
        np.testing.assert_allclose(100*(1-mm/rr), reference['gain_percent'], rtol=1e-10, atol=1e-10)


def history_summary(g, scale, mask):
    if not mask.any():
        return None
    xy = g[mask, :16].astype(float).reshape(-1, 8, 2)
    dt = np.diff(g[mask, 16:24].astype(float), axis=1)
    velocity = np.diff(xy, axis=1)/dt[..., None]
    change = np.linalg.norm(np.diff(velocity, axis=1), axis=-1).max(1)
    distance = np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(1)*scale[mask]
    neighbors = g[mask, 230:294].reshape(-1, 8, 8).any(2).sum(1)
    return dict(rows=int(mask.sum()), stationary_history_rows=int(np.all(xy == xy[:, :1], axis=(1, 2)).sum()),
        normalized_velocity_change_max=float(change.max()),
        normalized_velocity_change_median=float(np.median(change)),
        native_history_path_median=float(np.median(distance)),
        visible_neighbor_count_mean=float(neighbors.mean()))


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, data, _, identity, views = load()
    public, private = ROOT/reg['reports'], ROOT/reg['output']
    a = json.loads((public/'analysis.json').read_text())
    v = json.loads((public/'verification.json').read_text())
    assert a['identity'] == identity and v['all_checks_passed']
    assert v['analysis_sha256'] == file_digest(public/'analysis.json')
    baseline = data['geometry'][:, 332:356].reshape(-1, 12, 2)
    cv, cf = errors(baseline, data['target'], data['valid'], data['scale'])
    complete = data['valid'].all(1)
    checks, checked_rows = 0, 0
    slices, outputs = [], []
    under = {k:0 for k in a['summary']}
    for key, meta in views.items():
        site, seed = meta['outer_site'], meta['seed']
        binding = next(b for b in a['score_archives'] if b['view'] == key)
        assert file_digest(ROOT/binding['path']) == binding['sha256']
        forecast = meta['outer_prediction']['prediction']
        assert file_digest(ROOT/forecast['path']) == forecast['sha256']
        with np.load(ROOT/forecast['path'], allow_pickle=False) as z:
            ids, pred = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        ne, nf = errors(pred, data['target'][ids], data['valid'][ids], data['scale'][ids])
        checked_rows += len(ids)
        ref, rf = cv[ids], cf[ids]
        known = np.isfinite(ref)
        truth_harm = np.maximum(ne-ref, 0)
        truth_benefit = np.maximum(ref-ne, 0)
        same = np.all(pred == baseline[ids], axis=(1, 2))
        t = next(t for t in a['training'] if t['view'] == key and t['arm'] == 'ridge')
        assert file_digest(ROOT/t['checkpoint']) == t['checkpoint_sha256']
        pr = torch.load(ROOT/t['checkpoint'], map_location='cpu', weights_only=False)['preprocess']
        masks = dict(ADE=np.ones(len(ids), bool), complete_ADE=complete[ids],
            positive_easy_q25_diagnostic=(ref > 0) & (ref <= pr['positive_easy_cut']),
            hard_q75_diagnostic=ref >= pr['hard_cut'], zero_CV_complete=complete[ids] & (ref == 0))
        with np.load(ROOT/binding['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            for kind in a['summary']:
                p = z[kind]
                positive = (p[:, 0] > p[:, 1]) & ~same
                decisions = dict(positive_gain=positive, harm_fraction_0p1=positive & (p[:, 1] <= .1*p[:, 0]))
                c = next(c['conditional'] for c in a['conditional'] if c['view'] == key and c['kind'] == kind)
                np.testing.assert_allclose(np.mean((p[known, 1]-truth_harm[known])**2), c['harm_mse'], rtol=1e-12)
                for rule, use in decisions.items():
                    ade, fde = np.where(use, ne, ref), np.where(use, nf, rf)
                    r = a['summary'][kind][rule]['seeds'][str(seed)]
                    for name, mask in masks.items():
                        check_mean(ade, ref, mask, r[name]['by_scene'][site]); checks += 1
                    check_mean(fde, rf, np.ones(len(ids), bool), r['FDE']['by_scene'][site]); checks += 1
                    group = c['groups'][rule]
                    selected = use & known
                    assert int(use.sum()) == group['indexed_rows']
                    assert int(selected.sum()) == group['supported_rows']
                    assert int((use & ~known).sum()) == group['unknown_rows']
                    if selected.any():
                        np.testing.assert_allclose(truth_harm[selected].mean(), group['realized_harm'], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(p[selected, 1].mean(), group['predicted_harm'], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose((truth_benefit[selected]-truth_harm[selected]).mean(), group['realized_gain'], rtol=1e-12, atol=1e-12)
                        if rule == 'harm_fraction_0p1':
                            under[kind] += int(group['predicted_harm'] < group['realized_harm'])
                    zero = masks['zero_CV_complete']
                    harmed = zero & (ade > 0)
                    record = dict(view=key, kind=kind, rule=rule, indexed_rows=len(ids),
                        selected_rows=int(use.sum()), selected_unknown_rows=int((use & ~known).sum()),
                        zero_CV_complete_rows=int(zero.sum()), zero_CV_selected_rows=int((zero & use).sum()),
                        zero_CV_harmed_rows=int(harmed.sum()), zero_CV_max_absolute_harm=float(ade[zero].max()) if zero.any() else None,
                        harmed_query_ids_sha256=hashlib.sha256(ids[harmed].astype('<i8').tobytes()).hexdigest(),
                        harmed_history=history_summary(data['geometry'][ids], data['scale'][ids], harmed))
                    slices.append(record)
        print(json.dumps(dict(view=key, independent_arithmetic='passed', rows=len(ids))), flush=True)
    for kind, policies in a['summary'].items():
        for rule, summary in policies.items():
            seeds = summary['seeds']
            outputs.append(dict(kind=kind, rule=rule,
                ADE_gain_percent=summary['mean_seed_ADE']['equal_scene_gain_percent'],
                ADE_CI_low=summary['mean_seed_ADE']['scene_bootstrap_ci95'][0],
                ADE_CI_high=summary['mean_seed_ADE']['scene_bootstrap_ci95'][1],
                hard_gain_percent=summary['mean_seed_hard']['equal_scene_gain_percent'],
                positive_easy_degradation_percent=-summary['mean_seed_positive_easy_diagnostic']['equal_scene_gain_percent'],
                mean_switch_percent=100*float(np.mean([s['intervention_rate'] for s in seeds.values()])),
                zero_safe_seed_count=sum(s['zero_CV_empirical_no_added_error'] for s in seeds.values()),
                worst_zero_absolute_harm=max(s['zero_CV_max_absolute_harm'] for s in seeds.values()),
                sum_seed_unknown_selected_rows=sum(s['unknown_selected_rows'] for s in seeds.values())))
    independent = dict(result_source='fresh_run_independent_arithmetic_on_cached_verified_predictions',
        analysis_sha256=file_digest(public/'analysis.json'), replay_sha256=file_digest(public/'verification.json'),
        verifier_sha256=file_digest(Path(__file__)), bindings_checked=len(identity['source_bindings']),
        forecast_rows_independently_reduced=checked_rows, scene_metric_reductions_checked=checks,
        fixed_policy_scene_slices=len(slices), strict_rule_group_mean_harm_underpredictions=under,
        all_checks_passed=True, new_training=False, new_threshold_selection=False,
        independent_confirmation=False, deployment=False)
    immutable_json(public/'independent_verification.json', independent)
    immutable_json(public/'failure_slices.json', dict(provenance=independent, slices=slices))
    csv_path = public/'results.csv'
    import io
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(outputs[0])); writer.writeheader(); writer.writerows(outputs)
    content = buf.getvalue()
    if csv_path.exists():
        assert csv_path.read_bytes() == content.encode()
    else:
        csv_path.write_bytes(content.encode())
    print(json.dumps(independent, indent=2))


if __name__ == '__main__':
    main()
