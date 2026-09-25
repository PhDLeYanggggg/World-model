"""Account for frozen rejected switches and motion/neural contributions."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_dual_event_bridge as run
from scripts import build_m3w_european_selection_data as adapter
from scripts.report_m3w_european_floor_relative import dump
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.evaluation.m3w_native_metrics import native_errors


def main():
    cfg, _, bank, pid, identity = run.registration()
    training = run.checked_training(identity)
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text())
    assert done['identity'] == identity and done['all_passed']
    data = adapter.load(run.parent, pid); y = adapter.load(run.parent, pid, labels=True)
    sites = data['sites']; n = len(sites); origin = data['origin'][:, None]
    def error(p): return native_errors(p+origin, y['target_eval'], y['valid'], np.ones(n))[0]
    cv = native_errors(baseline_numpy(data['history'], 1), y['target_eval'], y['valid'], np.ones(n))[0]
    groups = {}
    for g in training['groups']:
        name = g['group']; seed = int(name.split('_seed')[1].split('_')[0])
        _, env, pair, old, _ = run.selection_pair(data, bank, g['producer'], seed, g['controller'])
        re, ce = error(pair['easy']), error(pair['all'])
        with np.load(run.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
            dec = {k: z[k].copy() for k in ('all_risk_only', 'dual_risk', 'easy_risk_only', 'utility_only')}
        assert not np.any(dec['dual_risk'] & ~dec['all_risk_only'])
        rejected = dec['all_risk_only'] & ~dec['dual_risk']
        event = (cv > 0)&(cv <= g['easy_cut'])
        labels = np.isfinite(cv)
        category = np.where(old['all']['old_stop'] & ~old['easy']['old_stop'], 'to_neural',
            np.where(old['easy']['old_stop'] & ~old['all']['old_stop'], 'from_neural',
            np.where(~old['easy']['old_stop'] & ~old['all']['old_stop'], 'motion_to_motion', 'same_neural')))
        row = dict(rejected_by_easy_risk=int(rejected.sum()), rejected_by_easy_risk_unknown=int((rejected & ~labels).sum()),
                   event_rows=int((event & labels).sum()), by_locality={}, intervention_attribution={})
        for site in sorted(set(sites)):
            pop = sites == site; denied = rejected & pop & labels
            delta = ce-re
            parts = dict(rows=int(denied.sum()),
                positive_gain=float(np.maximum(-delta[denied], 0).sum()),
                positive_harm=float(np.maximum(delta[denied], 0).sum()),
                easy_rows=int((denied & event).sum()),
                easy_gain=float(np.maximum(-delta[denied & event], 0).sum()),
                easy_harm=float(np.maximum(delta[denied & event], 0).sum()))
            er_all = np.where(dec['all_risk_only'], ce, re)
            er_dual = np.where(dec['dual_risk'], ce, re)
            observed = float((er_dual[pop & labels]-er_all[pop & labels]).sum())
            expected = parts['positive_gain']-parts['positive_harm']
            np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=1e-10)
            parts['dual_added_ADE_sum'] = observed
            row['by_locality'][site] = parts
        for arm in ('all_risk_only', 'dual_risk'):
            cats = {}
            for kind in ('to_neural', 'from_neural', 'motion_to_motion', 'same_neural'):
                use = dec[arm] & (env > 0) & (category == kind)
                good = use & labels
                cats[kind] = dict(interventions=int(use.sum()), supported=int(good.sum()),
                    net_ADE_reduction_sum=float((re[good]-ce[good]).sum()),
                    positive_harm_sum=float(np.maximum(ce[good]-re[good], 0).sum()))
            assert sum(v['interventions'] for v in cats.values()) == int((dec[arm] & (env > 0)).sum())
            row['intervention_attribution'][arm] = cats
        groups[name] = row
    result = dict(result_source='fresh_run_diagnostic_of_frozen_decisions', groups=groups,
        no_new_fitting=True, thresholds_changed=False, not_an_additional_policy_search=True,
        pooled_sums_not_primary_metric=True, localities_repeated_across_groups=True)
    dump(run.PUBLIC/'diagnostic_accounting.json', result)
    agg = {}
    for arm in ('all_risk_only', 'dual_risk'):
        agg[arm] = {k: dict(interventions=sum(r['intervention_attribution'][arm][k]['interventions'] for r in groups.values()),
            net_ADE_reduction_sum=sum(r['intervention_attribution'][arm][k]['net_ADE_reduction_sum'] for r in groups.values()))
            for k in ('to_neural', 'from_neural', 'motion_to_motion', 'same_neural')}
    risk = {}
    public_rows = [json.loads(p.read_text()) for p in (run.PUBLIC/'groups').glob('*.json')]
    for arm in ('all_risk_only', 'dual_risk', 'easy_risk_only'):
        risk[arm] = {}
        for event in ('all', 'easy'):
            ratios = [v['realized_ratio'] for r in public_rows for v in r['views'][arm]['risk_reliability'][event].values()]
            known = [v for v in ratios if v is not None]
            risk[arm][event] = dict(maximum=max(known), above_002=sum(v > .02 for v in known),
                defined=len(known), undefined=len(ratios)-len(known), not_independent_repetitions=True)
    summary = dict(attribution=agg, risk=risk,
        denied_rows_repeated=sum(r['rejected_by_easy_risk'] for r in groups.values()),
        locality_views_with_net_loss_from_easy_veto=sum(v['dual_added_ADE_sum'] > 0 for r in groups.values() for v in r['by_locality'].values()),
        total_locality_views=108)
    dump(run.PUBLIC/'diagnostic_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
