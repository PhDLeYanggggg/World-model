"""Fixed matched projection contrasts; no favorable threshold or arm selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.report_m3w_european_temporal_support import contrasts, summarize
from src.world_model.m3w_temporal_support import ARMS


def comparisons():
    result = []
    for arm in ARMS:
        for mode in ('envelope', 'envelope_coupled'):
            result.append([arm+'_'+mode, arm+'_frozen_cap'])
    for mode in ('frozen_cap', 'envelope_coupled'):
        for control in ('score_only', 'old_summary', 'ordered_history'):
            result.append(['history_neighbors_'+mode, control+'_'+mode])
    return result


def gates(summary):
    full = summary['full']
    cap = full['history_neighbors_envelope_coupled_vs_history_neighbors_frozen_cap']
    context = [full['history_neighbors_envelope_coupled_vs_'+a+'_envelope_coupled']
               for a in ('score_only', 'old_summary')]
    def primary(v):
        p = v['primary']
        return p['positive'] >= 2 and p['negative'] == p['not_estimable'] == 0
    cap_ok = primary(cap); context_ok = all(primary(v) for v in context)
    guards = all(v[k]['negative'] == v[k]['not_estimable'] == 0
                 for v in [cap, *context] for k in ('top10_pp', 'coverage_log', 'all_harm'))
    return dict(attribution_complete=True, original_predictions_exact=True,
        cap_relaxation_primary_screen=cap_ok, context_under_relaxed_cap_screen=context_ok,
        joint_cost_protection_screen=guards, joint_conditional_followup_justified=cap_ok and context_ok and guards,
        ceiling_floor_proves_conditional_bias=False, new_neural_training=False,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)


def distribution(values):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError('Finite complete diagnostics required')
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))


def main():
    from scripts import run_m3w_european_cap_attribution as run
    cfg, _ = run.registration(); cfg['comparisons'] = comparisons()
    doc = json.loads((run.PRIVATE/'readout.json').read_text()); rows = doc['rows']
    assert len(rows) == 432 and all(r['prior_scores_exact'] for r in rows)
    cs = contrasts(rows, cfg); summary = summarize(cs); gs = gates(summary)
    diagnostic = {}
    for pair in ('full', 'motion_only'):
        rr = [r for r in rows if r['input']['pair'] == pair]; diagnostic[pair] = {}
        for arm in ARMS:
            ds = [r['diagnostics'][arm] for r in rr]
            assert all(d['status'] == 'measured' for d in ds)
            diagnostic[pair][arm] = {key:distribution([d[key] for d in ds]) for key in ds[0]
                if key not in ('status', 'target_used_only_for_analysis')}
    run.immutable_json(run.PUBLIC/'aggregate_metrics.json', dict(contrasts=cs, summary=summary,
        dependent_view_diagnostics=diagnostic, detailed_metrics=run.artifact(run.PRIVATE/'readout.json')))
    run.immutable_json(run.PUBLIC/'gates.json', gs)
    lines = ['# Frozen-Cap Attribution', '', '## Material Passport',
        'Fresh deterministic attribution of previously frozen fitting-development readouts; zero new fits or neural updates.',
        'Registered after the parent negative result, before new projection scoring. Not independent confirmation.', '',
        '## Paired Contrasts', '',
        '| Family / contrast | Metric | Positive / negative / overlap / missing | Point range |', '|---|---|---|---|']
    for pair, table in summary.items():
        for name, metrics in table.items():
            for metric, v in metrics.items():
                lines.append(f"| {pair} / {name} | {metric} | {v['positive']} / {v['negative']} / {v['overlap']} / {v['not_estimable']} | {v['point_range']} |")
    lines += ['', 'Primary is expected easy-harm MSE gain, not trajectory ADE/FDE or easy degradation.',
        'Three excluded-outer contexts and three seeds are averaged inside scoring locality. 3000 paired four-locality resamples.',
        'Six assignments overlap. Intervals are exploratory, unadjusted, with few independent localities. Missing guards are retained.', '',
        '## Algebra and Intervention Frequency', '',
        'Ranges below summarize dependent views, not independent effect estimates. Positive cap-relaxation benefit means lower realized MSE.',
        'The signed score is only an algebra diagnostic. Nonnegative and uncoupled envelope scores may violate the nested harm ordering.',
        'Coupling raises predicted all-harm to at least predicted easy-harm; it does not train a better all-harm model or certify safety.', '',
        '| Family / arm | Quantity | Minimum / median / maximum |', '|---|---|---|']
    for pair, arms in diagnostic.items():
        for arm, quantities in arms.items():
            for key, value in quantities.items(): lines.append(f'| {pair} / {arm} | {key} | {value} |')
    lines += ['', 'The target-derived pointwise floor cannot establish bias in a conditional mean.',
        'Projection changes are label-free. Analysis labels establish retrospective cost differences, never inference thresholds.',
        'The causal disagreement envelope follows matched masked ADE and the triangle inequality; it is not a physical safety bound.', '',
        '## Gates', '', *['- '+k+': '+str(v).lower() for k,v in gs.items()], '',
        'Detector pixels and obs8/pred12 native annotation steps only. No seconds/metric/true3D/foundation or physical-safety claim.',
        'Independent selection, calibration and confirmation remain closed. Stage5C/SMC stay disabled.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary, gates=gs), indent=2))


if __name__ == '__main__': main()
