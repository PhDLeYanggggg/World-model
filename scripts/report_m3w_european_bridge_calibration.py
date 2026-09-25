"""Calibration transport, moment diagnostics and finite-scene feasibility."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_bridge_calibration as run
from scripts import build_m3w_european_selection_data as adapter
from src.evaluation.m3w_bridge_risk_calibration import zero_violation_scene_upper, hb_zero_loss_floor


def summarize(rows, seeds):
    if len(rows) != 18 or len(seeds) != 6: raise ValueError('All registered groups required')
    views = {}
    for name in sorted(rows[0]['views']):
        values = [r['views'][name] for r in rows]
        def span(v): return [min(v), max(v)] if all(x is not None for x in v) else None
        risks = [local['actual']['events'] for v in values for local in v['per_locality'].values()]
        easy = [-v['ADE_vs_CV_easy']['worst_scene_gain_percent'] for v in values]
        views[name] = dict(all_gain_vs_raw=span([v['ADE_vs_raw']['all']['equal_scene_gain_percent'] for v in values]),
            hard_gain_vs_raw=span([v['ADE_vs_raw']['hard']['equal_scene_gain_percent'] for v in values]),
            all_gain_vs_full_neural=span([v['ADE_vs_full_neural']['all']['equal_scene_gain_percent'] for v in values]),
            all_gain_vs_training_selected=span([v['ADE_vs_training_selected']['equal_scene_gain_percent'] for v in values]),
            worst_easy_degradation=span([max(0., x) for x in easy]),
            easy_pass=sum(e <= 2 and v['zero_CV_harm'] == 0 for e,v in zip(easy, values)),
            complete_risk_pass=sum(v['observed_risk_feasible'] for v in values),
            calibration_complete_risk_pass=sum(v['calibration_evidence']['feasible'] for v in values),
            abstention_maps=sum(v['calibration_rule']['abstain'] for v in values),
            all_positive_harm_violations=sum(not v['all']['passes'] for v in risks),
            easy_positive_harm_violations=sum(not v['easy']['passes'] for v in risks),
            undefined_event_views=sum(not v['all']['supported'] or not v['easy']['supported'] for v in risks),
            switch_rate=span([v['switch_rate'] for v in values]), locality_views_dependent=True)
    contrasts = {}
    for key in next(iter(seeds.values())):
        vals = [g[key] for g in seeds.values()]
        if any(v.get('status') == 'not_estimable' for v in vals):
            contrasts[key] = dict(status='not_estimable'); continue
        contrasts[key] = dict(point_range=[min(v['gain_percent'] for v in vals), max(v['gain_percent'] for v in vals)],
            positive_CI=sum(v['CI'][0] > 0 for v in vals), negative_CI=sum(v['CI'][1] < 0 for v in vals),
            overlapping_CI=sum(v['CI'][0] <= 0 <= v['CI'][1] for v in vals))
    return dict(views=views, contrasts=contrasts)


def feasibility():
    return dict(scope='hypothetical_iid_locality_sensitivity_not_actual_harm_ratio_certificate',
        zero_violation_upper=[dict(n=n, delta=d, bound=zero_violation_scene_upper(n, d))
            for n in (4, 6, 12, 24, 60, 150) for d in (.1, .05, .01)],
        best_case_HB=[dict(n=n, hypothetical_bounded_loss_target=.02, p_value_floor=hb_zero_loss_floor(n, .02))
            for n in (4, 6, 12, 24, 60, 150)],
        minimum_hypothetical_zero_loss_scenes=[dict(delta=d, n=int(np.ceil(np.log(d)/np.log(.98)))) for d in (.1, .05, .01)],
        actual_harm_ratio_known_bounded_0_1=False, confidence_level_adopted=False,
        risk_tolerance_changed=False, independent_calibration_opened=False)


def moment_transport(rows, bank, pid):
    data = adapter.load(run.parent, pid); labels = adapter.load(run.parent, pid, labels=True)
    known = labels['valid'].any(1); sites = data['sites']; records = {}
    for g in rows:
        group = g['group']; out = {}
        for pair in ('full', 'motion_only'):
            values = run.attribution.scores(pair, group)
            for family in ('neural', 'ridge'):
                moments = values[1 if family == 'neural' else 3]
                filename = group+'_'+pair+'_'+family
                mapping = json.loads((run.PRIVATE/'maps'/(filename+'.json')).read_text())
                with np.load(run.PRIVATE/'decisions'/(filename+'.npz'), allow_pickle=False) as z:
                    for kind in ('none', 'population_rescale', 'selected_risk_grid'):
                        key = pair+'__'+family+'__'+kind; use = z[kind]; rule = mapping['fitted']['rules'][kind]
                        mult = np.array([rule['denominator_multiplier'], rule['harm_multiplier']]) if kind == 'population_rescale' else np.ones(2)
                        by_site = {}
                        for site in sorted(set(sites)):
                            pop = sites == site; supported = pop & known
                            truth = g['views'][key]['per_locality'][site]['actual']['events']['all']
                            den, harm = truth['reference_mass'], truth['positive_harm']
                            sd = float(moments[supported, 0].sum()); sh = float(moments[supported & use, 1].sum())
                            by_site[site] = dict(supported_rows=int(supported.sum()), unknown_rows=int((pop & ~known).sum()),
                                raw_predicted_ref_mass_supported=sd, raw_predicted_harm_supported=sh,
                                raw_ratio_supported=sh/sd if sd > 0 else None,
                                adjusted_ratio_supported=sh*mult[1]/(sd*mult[0]) if sd > 0 else None,
                                actual_ratio_supported=truth['ratio'], actual_ref_mass=den, actual_positive_harm=harm,
                                raw_ref_bias=sd/den if den > 0 else None, raw_selected_harm_bias=sh/harm if harm > 0 else None,
                                adjusted_selected_harm_bias=sh*mult[1]/harm if harm > 0 else None,
                                whole_index_predicted_ratio=g['views'][key]['per_locality'][site]['predicted_ratio'])
                        out[key] = by_site
        records[group] = out
    return dict(groups=records, role='posthoc_diagnostic_not_used_for_thresholds',
        matched_support_for_prediction_truth_comparison=True,
        unknown_futures_kept_in_inference=True, thresholds_changed=False)


def fmt(v): return 'undefined' if v is None else f'{v[0]:+.6f}% to {v[1]:+.6f}%'


def plot_transport(aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [k for k in sorted(aggregate['views']) if not k.endswith('__reference')]
    y = np.arange(len(names)); values = [aggregate['views'][k] for k in names]
    with plt.rc_context({'font.size': 9, 'svg.hashsalt': 'm3w-bridge-calibration'}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        left.barh(y-.16, [v['calibration_complete_risk_pass'] for v in values], .3,
                  color='#23856d', label='Four-source calibration C')
        left.barh(y+.16, [v['complete_risk_pass'] for v in values], .3,
                  color='#bb4962', label='Six-locality development readout')
        left.set_yticks(y, [k.replace('__', ' / ') for k in names]); left.invert_yaxis()
        left.set_xlim(0, 19); left.set_xticks([0, 6, 12, 18])
        left.set_xlabel('Settings passing all observed constraints (of 18)')
        left.set_title('Empirical calibration does not establish transport')
        left.legend(loc='lower left', bbox_to_anchor=(0, -0.16), frameon=False)
        for i, v in enumerate(values):
            lo, hi = v['all_gain_vs_raw']
            right.plot([lo, hi], [i, i], color='#335a98', linewidth=3)
            right.scatter([lo, hi], [i, i], color='#335a98', s=12)
        right.axvline(0, color='#555555', linewidth=.8)
        right.set_xlabel('All-ADE gain over the same raw rule (%)')
        right.set_title('Range across all 18 settings, not a confidence interval')
        for ax in (left, right):
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', color='#e6e6e6'); ax.set_axisbelow(True)
        fig.suptitle('Fixed reference-aligned calibration: neural and ridge controls', fontsize=13)
        fig.text(.02, .025, 'Opened development only. Settings share localities; they are not independent trials. '
                 'No deployment or risk certificate.', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .96))
        fig.savefig(run.PUBLIC/'calibration_transport.svg', metadata={'Date': None})
        fig.savefig(run.PRIVATE/'calibration_transport_preview.png', dpi=130)
        plt.close(fig)


def main():
    cfg, bank, pid, identity = run.registration()
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed']
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    seeds = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text()); aggregate = summarize(rows, seeds)
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', aggregate)
    plot_transport(aggregate)
    gate = dict(source_C_exclusion_verified=True, empirical_maps_fitted=True, decisions_frozen_before_readout=True,
        calibrated_full_neural_easy_preserved=aggregate['views']['full__neural__selected_risk_grid']['easy_pass'] == 18,
        calibrated_full_neural_observed_harm_pass=aggregate['views']['full__neural__selected_risk_grid']['complete_risk_pass'] == 18,
        calibrated_full_neural_beats_raw_all_six=aggregate['contrasts']['full__neural__selected_risk_grid_vs_none__all']['positive_CI'] == 6,
        independent_reserved_calibration=False, independent_confirmation=False, certified_risk_control=False,
        new_neural_dynamics=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json', gate)
    run.immutable_json(run.PUBLIC/'finite_scene_feasibility.json', feasibility())
    run.immutable_json(run.PUBLIC/'moment_transport.json', moment_transport(rows, bank, pid))
    maps = run.checked_calibration(identity)['maps']; loostats = {}
    for ref in maps:
        m = json.loads((ROOT/ref['path']).read_text()); key = m['pair']+'__'+m['family']
        short = dict(group=m['group']['group'], pair=m['pair'], family=m['family'],
            calibration_localities=m['fitted']['fit_localities'], fitted=m['fitted'],
            leave_one_source_C=m['leave_one_source_C'], private_source=ref)
        run.immutable_json(run.PUBLIC/'maps'/Path(ref['path']).name, short)
        vals = loostats.setdefault(key, [])
        vals.extend(dict(abstain=v['rule']['abstain'], feasible=v['held_out']['feasible'],
            gain=v['held_out']['equal_locality_gain_percent']) for v in m['leave_one_source_C'].values())
    loo = {k:dict(views=len(v), feasible=sum(x['feasible'] for x in v), abstain=sum(x['abstain'] for x in v),
        not_independent=True, used_to_select_final_threshold=False) for k,v in loostats.items()}
    run.immutable_json(run.PUBLIC/'leave_one_source_C_summary.json', loo)
    lines = ['# Aligned Source-C Calibration Results', '', '## Material Passport',
        'Fresh source-C inference and calibration; cached_verified models; fresh frozen development readout.',
        'Three seeds,72 maps,288 policy views. Six reused opened model-selection localities, not confirmation.', '',
        '| Pair / scorer / rule | All gain vs own raw | Hard gain vs own raw | All gain vs full neural raw | Worst easy degradation | Easy passes | Complete observed risk passes | C risk passes | Abstention maps |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in aggregate['views'].items():
        lines.append(f"| {k} | {fmt(v['all_gain_vs_raw'])} | {fmt(v['hard_gain_vs_raw'])} | {fmt(v['all_gain_vs_full_neural'])} | {fmt(v['worst_easy_degradation'])} | {v['easy_pass']}/18 | {v['complete_risk_pass']}/18 | {v['calibration_complete_risk_pass']}/18 | {v['abstention_maps']}/18 |")
    lines += ['', 'Complete observed risk means both positive-harm/R-error events, net CV-easy <=2%,',
        'and no CV-zero harm. Missing support cannot pass. Structural fallback is not positive gain.',
        'Ranges contain all source-role/seed groups, not chosen winners. Predicted ratios in group',
        'JSON use all indexed rows and raw moments. moment_transport.json additionally reports',
        'matched-label-support raw/adjusted moments so unknown future support is not silently mixed.', '',
        '## Three-Seed All-ADE Contrasts', '',
        'Seedwise gains averaged within locality, then3,000 paired locality resamples.',
        'No row independence or multiplicity-adjusted discovery claim.', '',
        '| Producer/controller | Contrast | Gain | 95% locality CI |', '|---|---|---:|---:|']
    for group, vals in seeds.items():
        for key,v in vals.items():
            if key.endswith('__all'):
                lines.append(f"| {group} | {key} | {v['gain_percent']:+.6f}% | {fmt(v['CI'])} |")
    lines += ['', '## Within-C Stability Diagnostic', '',
        '| Pair/scorer | Three-source fit, held-C feasible | Abstentions |', '|---|---:|---:|']
    for k,v in loo.items(): lines.append(f"| {k} | {v['feasible']}/{v['views']} | {v['abstain']} |")
    lines += ['', 'These views overlap and never select the final four-C map. Twelve reserved calibration',
        'and six confirmation localities remain closed. No deployment, metric/seconds/physical-safety,',
        'true3D/foundation/human-gold claim, Stage5C or SMC.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    (run.PUBLIC/'world_model_gate.md').write_text('# Aligned Calibration Gates\n\nNo empirical pass is a certificate.\n\n'+
        '\n'.join(f'- {k}: {str(v).lower()}' for k,v in gate.items())+'\n')
    queue = json.loads((run.PRIVATE/'create_queue.json').read_text())
    events = [json.loads(v) for v in (run.PRIVATE/'events.jsonl').read_text().splitlines()]
    run.immutable_json(run.PUBLIC/'compute_receipt.json', dict(runtime='native_arm64_torch_cpu', compute_threads=4,
        interop_threads=1, dataloader_workers=0, new_neural_updates=0, fitted_calibration_maps=72,
        source_scoring_pids=sorted({v['pid'] for v in events if v['state'] == 'source_C_scoring'}),
        cached_scoring_models=144, new_source_scoring=True, checkpoint_weights_unchanged=True,
        create_queue_returncode=queue['response']['returncode'], create_checked_at=queue['completed_utc'],
        create_jobs_submitted=0, remote_modified=False, remote_project_inventory='not_run',
        placement='Data and checkpoints local; bounded CPU inference/calibration workload'))
    print(json.dumps(dict(aggregate=aggregate, gates=gate, loo=loo), indent=2))


if __name__ == '__main__': main()
