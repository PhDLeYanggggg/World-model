"""Explain frozen benefit/harm changes without changing a decision or primary."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_risk_priority_policy as run
from scripts.analyze_m3w_easy_hurdle import exchange
from scripts.report_m3w_easy_risk_priority import BOUNDARIES, interpret, put, value


def main():
    cfg = json.loads(run.CONFIG.read_text()); ident = run.identity()
    summary_path = run.PUBLIC/'summary.json'; summary = json.loads(summary_path.read_text())
    assert summary['identity'] == ident
    replay = json.loads((run.PUBLIC/'evaluation_replay.json').read_text())
    assert replay['exact'] and replay['summary'] == run.base.artifact(summary_path)
    ref = replay['details']; assert run.base.artifact(ROOT/ref['path']) == ref
    details = json.loads((ROOT/ref['path']).read_text())
    index = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['rows']}
    sites = sorted({r['site'] for r in details['rows']}); assert len(sites) == 12
    comparisons = {}
    for new, old in run.CONTRASTS:
        rows = [dict(group=r['group'], site=r['site'],
                     metric=exchange(r['metric'], index[r['group'], r['site'], old]))
                for r in details['rows'] if r['policy'] == new]
        assert len(rows) == 216
        comparisons[new+'_vs_'+old] = {key: run.base.inter.paired_localities(rows, sites, key,
            cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for key in rows[0]['metric']}
    training = json.loads((run.PUBLIC/'training_summary.json').read_text())
    for arm in run.repair.ARMS:
        losses = [r for r in details['losses'] if r['arm'] == arm]
        assert len(losses) == 108 and all(r['step'] == 2000 for r in losses)
        for term, expected in training['loss'][arm].items():
            for phase, key in [('first', 'first_mean'), ('last', 'final_mean')]:
                np.testing.assert_allclose(np.mean([r[phase]['monitor'][term] for r in losses]),
                                           expected[key], rtol=1e-12, atol=1e-12)
    support = json.loads((run.PUBLIC/'structural_support.json').read_text())['support']
    result = dict(result_source='fresh_run_posthoc_accounting_of_frozen_development_readout',
        summary=run.base.artifact(summary_path), details=ref, comparisons=comparisons,
        fitting_monitors_match_training_receipt=True, independent_localities=12,
        repeated_role_views_not_independent=True, primary_changed=False, thresholds_changed=False,
        new_training=False, full_floor_denominator_is_diagnostic=True, **interpret(summary))
    run.immutable(run.PUBLIC/'posthoc_accounting.json', result)
    lines = ['# Risk-Priority Repair: Benefit, Harm and Remaining Gaps', '',
        'This descriptive decomposition uses the frozen readout. No policy, threshold, '
        'checkpoint, primary comparison or gate is selected from it.', '',
        '## Predictive Result', '', f"Verdict: **{result['verdict']}**. No deployment change.", '',
        '| Comparison | Lost benefit pp | Avoided positive harm pp | Net gain/full floor pp |',
        '|---|---:|---:|---:|']
    for name, metrics in comparisons.items():
        lines.append('| '+name+' | '+' | '.join(value(metrics[k]) for k in
            ('lost_benefit_pp', 'harm_reduction_pp', 'net_gain_full_floor_pp'))+' |')
    lines += ['', 'For each view, error = floor error - benefit + positive harm. '
        'Net gain therefore equals harm reduction minus lost benefit. These columns '
        'share the full-floor denominator; neither they nor a changed intervention '
        'volume can replace the registered matched-query ADE or selected-risk results.', '',
        '## Training Mechanism Check', '',
        f"The auxiliary cap was active on{training['cap_active_updates']:,}/216,000 updates; "
        f"mean alpha={training['mean_alpha_over_updates']:.7g}. Minimum pre-optimizer risk "
        f"projection={training['minimum_pre_optimizer_risk_projection']:.7g}. Only "
        f"{training['repaired_lower_final_direct_risk_heads']}/108 repaired heads have lower final "
        'direct-risk fitting loss than the control. Fitting monitors from the restored '
        'checkpoints agree with the CREATE training receipt. Bounding raw gradient '
        'norms is not a guarantee about AdamW steps, useful admissions or held risk.', '',
        'The supervised monitor is the unweighted reporting sum, not the scalar '
        'objective implied by a time-varying detached cap. Compare the same monitor '
        'component between arms; do not rank these training rules by total loss.', '',
        '| Arm | Held-development Brier | Signed-risk MSE | Signed bias |', '|---|---:|---:|---:|']
    for arm, metrics in summary['quality'].items():
        lines.append('| '+arm+' | '+' | '.join(value(metrics[k]) for k in ('Brier', 'signed_MSE', 'signed_bias'))+' |')
    lines += ['', 'The single changed factor supports a controlled comparison of this '
        'fixed cap, not a universal claim about gradient balancing. A lower composed '
        'risk loss, if present in a slice, need not improve utility-preserving joint '
        'allocation. Probability, conditional-moment bias and ranking need separate '
        'evidence; their causal mechanism is not identified by these scores alone.', '',
        '## Structural Support and Generalization', '',
        f"At least{support['forced_undefined_selected_risk_views']} dependent views are "
        'forced to abstain by unchanged anchors. Their undefined selected risk is '
        'retained. The every-view-defined-risk screen cannot be repaired by auxiliary '
        'gradient reweighting alone. Predicted feasible constraints are also not a '
        'certificate about realized harm. Independent calibration and confirmation '
        'remain missing; no eligibility or risk-definition change is authorized by '
        'this descriptive result.', '', BOUNDARIES]
    put('failure_analysis.md', '\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=result['verdict'], comparisons=len(comparisons), changed_policy=False)))


if __name__ == '__main__':
    main()
