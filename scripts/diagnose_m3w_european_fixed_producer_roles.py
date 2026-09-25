"""Post-readout accounting of changed actions; no new policy or fit."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_fixed_producer_roles as run
from scripts.report_m3w_european_floor_relative import dump
from src.evaluation.m3w_native_metrics import native_errors
import numpy as np


def switch_accounting(candidate, floor, new, old):
    n, d = np.asarray(candidate, float), np.asarray(floor, float)
    new, old = np.asarray(new), np.asarray(old)
    if (n.ndim != 1 or d.shape != n.shape or new.shape != n.shape or old.shape != n.shape
            or new.dtype != bool or old.dtype != bool or not np.isfinite(n).all()
            or not np.isfinite(d).all() or np.any(n < 0) or np.any(d < 0) or len(n) == 0):
        raise ValueError('Supported paired nonnegative errors and frozen boolean choices required')
    added, removed = new & ~old, old & ~new
    diff = n-d; new_error = np.where(new, n, d); old_error = np.where(old, n, d)
    terms = dict(added_harm=float(np.maximum(diff[added], 0).sum()/len(n)),
        added_benefit=float(np.maximum(-diff[added], 0).sum()/len(n)),
        removed_lost_benefit=float(np.maximum(-diff[removed], 0).sum()/len(n)),
        removed_avoided_harm=float(np.maximum(diff[removed], 0).sum()/len(n)))
    reconstructed = terms['added_harm']-terms['added_benefit']+terms['removed_lost_benefit']-terms['removed_avoided_harm']
    actual = float((new_error-old_error).mean())
    np.testing.assert_allclose(reconstructed, actual, rtol=1e-10, atol=1e-10)
    reference = float(old_error.mean())
    return dict(rows=len(n), added_rows=int(added.sum()), removed_rows=int(removed.sum()),
        old_error_mean=reference, new_error_mean=float(new_error.mean()), mean_change=actual,
        native_pixel_terms=terms, percent_terms={k: 100*v/reference for k, v in terms.items()} if reference > 0 else None,
        net_degradation_percent=100*actual/reference if reference > 0 else None)


def main():
    run.ensure_frozen(); cfg, bcfg, ctx, bid, identity = run.load(); data = ctx[2]; output = {}
    for g in run.groups(bcfg, ctx, bid, identity):
        ids = g['roles']['readout']; sites = data['sites'][ids]; cv = data['baseline_ade'][ids, 1]
        def error(p):
            return native_errors(p.astype(float)+data['origin'][ids, None], data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
        ne, de = error(g['a']['p'][ids]), error(g['d'][ids])
        with np.load(run.PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); new, old = z['producer_matched'].copy(), z['old_stop'].copy()
        result = json.loads((run.PRIVATE/'evaluation'/(g['name']+'.json')).read_text())
        masks = dict(all=np.isfinite(cv), easy=(cv > 0) & (cv <= g['design']['easy_cut']), hard=cv >= g['design']['hard_cut'])
        records = {}
        for subset, mask in masks.items():
            localities = {}; terms = []
            for site in sorted(set(sites)):
                use = mask & (sites == site)
                if not use.any():
                    localities[site] = dict(status='no_supported_labels'); continue
                r = switch_accounting(ne[use], de[use], new[use], old[use]); localities[site] = r
                expected = result['views']['producer_matched']['ADE_vs_old_stop4'][subset]['by_scene'][site]
                assert expected['rows'] == r['rows']
                np.testing.assert_allclose([expected['model_error'], expected['reference_error']], [r['new_error_mean'], r['old_error_mean']], rtol=1e-10, atol=1e-10)
                if r['percent_terms'] is not None: terms.append(r['percent_terms'])
            averaged = {k: float(np.mean([x[k] for x in terms])) for k in terms[0]} if len(terms) == 4 else None
            net = (averaged['added_harm']-averaged['added_benefit']+averaged['removed_lost_benefit']-averaged['removed_avoided_harm']) if averaged is not None else None
            expected = result['contrasts']['old_stop']['ADE'][subset]
            if net is not None: np.testing.assert_allclose(net, -expected['equal_scene_gain_percent'], rtol=1e-9, atol=1e-9)
            records[subset] = dict(localities=localities, equal_locality_percent_terms=averaged,
                net_degradation_percent=net, comparison_gain_ci=expected['scene_bootstrap_ci95'])
        output[g['name']] = records
    negative = {group: r['all'] for group, r in output.items() if r['all']['comparison_gain_ci'] is not None and r['all']['comparison_gain_ci'][1] < 0}
    report = dict(status='fresh_run_posthoc_frozen_choice_accounting_not_new_policy',
        units='image_pixel_errors_percentage_changes_vs_old_stop', no_training=True, no_threshold_selection=True,
        groups=output, negative_all_CI_groups=list(negative),
        negative_group_terms={g: r['equal_locality_percent_terms'] for g, r in negative.items()},
        evaluation_receipt=run.artifact(run.PRIVATE/'evaluation_complete.json'))
    dump(run.PUBLIC/'changed_action_accounting.json', report)
    print(json.dumps(dict(groups=len(output), negative_groups=report['negative_group_terms'])), flush=True)


if __name__ == '__main__': main()
