"""Publish completed matched-estimator evidence without model or policy selection."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_multiaction_forest as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_multiaction_cost_readout import interval


def build():
    public = run.PUBLIC
    summary = json.loads((public/'summary.json').read_text())
    verified = json.loads((public/'verification.json').read_text())
    if not verified['exact_replay'] or verified['summary_sha256'] != run.sha(public/'summary.json'):
        raise ValueError('Separate exact inference replay required before reporting')
    groups = list(aggregate_groups(run.NAME).values())
    cfg = json.loads(run.parent.parent.CONFIG.read_text())
    held = [g for g in groups if g['scope'] == 'inner_held']
    details = {}
    for name in held[0]['result']['policies']:
        details[name] = {metric:interval(held, lambda r,n=name,m=metric:r['policies'][n][m], cfg)
            for metric in ('known_FDE_gain_floor_percent','known_regret_over_cost_scale')}
    fitting = json.loads((public/'training_freeze.json').read_text())
    rows = [json.loads((ROOT/ref['path']).read_text()) for ref in fitting['fits']]
    if len(rows) != 54 or sum(r['trees'] for r in rows) != 6912:
        raise ValueError('Complete fixed training roster required')
    held_summary = summary['scopes']['inner_held']
    diff = held_summary['forest_minus_neural']['physical_neural']['full_known_pp']
    lower = held_summary['forest_minus_neural']['physical_neural']['full_lower_pp']
    gates = dict(all54forests_and6912trees_complete=True, all_current_inputs_and_samplers_matched=True,
        exact144group_replay=True,
        forest_minus_neural_known_CI_positive=diff['CI95'] is not None and diff['CI95'][0] > 0,
        forest_minus_neural_completion_lower_CI_positive=lower['CI95'] is not None and lower['CI95'][0] > 0,
        neural_increment_completion_CI_positive=summary['neural_increment_conditional_CI_positive'],
        full_required_recording_2percent_budget_supported=summary['primary_full_roster_budget_supported'],
        independent_confirmation=False, deployment_promoted=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    return dict(result_source='fresh_generated_from_verified_frozen_readout',
        summary_sha256=run.sha(public/'summary.json'), verification_sha256=run.sha(public/'verification.json'),
        training_freeze_sha256=run.sha(public/'training_freeze.json'), primary_fits=54, primary_trees=6912,
        total_primary_fit_seconds=sum(r['seconds'] for r in rows), training=fitting,
        gates=gates, held=held_summary, held_FDE_and_regret=details,
        inference_label_filter=False, three_forecasting_training_seeds=False,
        primary_neural_comparator='frozen_static_scene_cost_heads',
        primary_estimator_contrast='fixed_mixed_bank_forest_minus_static_neural_full_and_matched',
        inherited_summary_bank_contrast='mixed_minus_physical_within_forests_not_the_estimator_contrast',
        estimator_objective_compute_identical=False)


def show(value):
    return 'not_estimable' if value is None else f'{value:+.6f}'


def ci(value):
    return 'not_estimable' if value['CI95'] is None else '['+', '.join(show(x) for x in value['CI95'])+']'


def render(value):
    held = value['held']; gates = value['gates']; train = value['training']
    lines = ['# Matched Conventional Cost Estimator: Verified Readout', '',
        'Result source: fresh ExtraTrees training and inference; frozen neural forecasts,',
        'source packets and static-scene neural controls are cached_verified. SOURCE',
        'inner-held is exposed development, not independent confirmation or whole-pipeline OOF.', '',
        '## What Was Actually Trained', '',
        f"All 54 registered forests / 6,912 primary trees complete (PID {train['pid']}, exit 0).",
        f"Training process wall time {train['seconds']:.6f}s; sum of primary fit times {value['total_primary_fit_seconds']:.6f}s;",
        f"peak RSS {train['peak_RSS_bytes']:,}B; all private forest checkpoints {train['private_checkpoint_bytes']:,}B.",
        'No new trajectory forecaster or Torch optimizer updates. The 128-tree refit and',
        'all 144 inference groups replay exactly. Three controller seeds are not three',
        'independently trained forecasting pipelines.', '',
        'Both estimator families receive the same 398 causal/static fields, seven action',
        'identities and causal envelope information, fitting-only normalization and the',
        'original query-draw multiplicities. Standard forest quadratic regression with',
        'post-fit projection is not the neural underharm/ranking loss or decoder; this',
        'is an estimator/objective control, not a matched-FLOP one-factor ablation.', '',
        'Inherited calibration MSE retains the parent all-seven-action definition,',
        'including inactive-bank predictions. It is not active-bank policy error;',
        'the separate fitting-region mechanism check uses legal bank actions.', '',
        '## Primary Fixed Comparisons', '',
        '| Bank | Forest known ADE gain vs floor (%) | Static neural gain (%) | Forest minus neural (pp), 95% CI | Completion-lower contrast (pp), 95% CI | Full record-risk support |',
        '|---|---:|---:|---|---|---|']
    for arm in ('physical_only','physical_neural','neural_only'):
        contrast = held['forest_minus_neural'][arm]; safety = held['safety'][arm]
        lines.append(f"| {arm} | {show(held['accuracy'][arm]['point'])} | {show(held['accuracy'][arm+'_neural_parent']['point'])} | "
            f"{show(contrast['full_known_pp']['point'])}, {ci(contrast['full_known_pp'])} | "
            f"{show(contrast['full_lower_pp']['point'])}, {ci(contrast['full_lower_pp'])} | "
            f"{safety['budget_record_views']}/{safety['required_record_views']} |")
    neural = held['contrasts']['physical_neural_minus_physical_only']
    lines.extend(['',f"Within forests, mixed minus physical known ADE: {show(neural['full_known_pp']['point'])}pp, CI {ci(neural['full_known_pp'])}.",
        f"Its completion-lower contrast: {show(neural['full_lower_pp']['point'])}pp, CI {ci(neural['full_lower_pp'])}.",
        'Intervals use 3,000 locality resamples after averaging repeated context/controller',
        'seed views; these are conditional development intervals. Count-matched contrasts',
        'are utility-only. They do not repair recording-risk failures or undefined support.', '',
        'Completion contrasts divide worst-case net utility differences by the fixed',
        'known floor-error mass. They are not error-reduction percentages for a fully',
        'observed future population with an unknown denominator.', '',
        '## Known-Outcome Hard/Easy Supplement', '',
        '| Subset | Repeated known row views | Empty required views | Forest mixed ADE gain (%) | Neural mixed ADE gain (%) |',
        '|---|---:|---:|---:|---:|'])
    for subset, row in held['subsets'].items():
        lines.append(f"| {subset} | {row['known_row_views']} | {row['empty_known_views']} | "
            f"{show(row['policies']['physical_neural']['ADE_gain_percent'])} | {show(row['policies']['physical_neural_neural_parent']['ADE_gain_percent'])} |")
    risk = held['safety']['physical_neural']
    lines.extend(['', 'These ratios pool known error mass over repeated views, not independent samples',
        'or the primary locality macro. Empty required hard views remain not_estimable.',
        'Hard/easy cuts were fixed on fitting data. Unknown outcomes are not imputed.', '',
        '## Safety And Interpretation', '',
        f"Mixed forest: {risk['selected']:,} selected row views, {risk['selected_unknown']:,} unknown selected; "
        f"{risk['known_all_risk_violations']} known all-risk violations, {risk['known_easy_risk_violations']} known easy-risk violations, "
        f"{risk['undefined_easy_record_views']} undefined easy recording views.", '',
        f"Full required recording all/easy 2% contract supported: {gates['full_required_recording_2percent_budget_supported']}.",
        f"Forest beats static neural on known paired CI: {gates['forest_minus_neural_known_CI_positive']}.",
        f"Forest completion-lower paired CI positive: {gates['forest_minus_neural_completion_lower_CI_positive']}.",
        f"Neural candidate increment completion CI positive: {gates['neural_increment_completion_CI_positive']}.",
        'Do not replace an unknown-completion failure with a known-only win. No policy,',
        'seed, checkpoint or 2x/4x sensitivity was chosen after this readout. There is',
        'no deployment promotion, independent confirmation, or submission-ready claim.', '',
        '## Scope And Next Scientific Decision', '',
        'EuropeanSquares SOURCE uses oblique webcam detector-track box midpoint pixels,',
        'obs8/pred12/stride12 raw frames. It is not SDD human-gold, metric/seconds,',
        'true 3D or a foundation model. No future endpoints, central velocity, test goals,',
        'test normalization or future-support masks enter inference. Independent',
        'selection/calibration/confirmation and DroneCrowd remain closed. No Stage5C or SMC.', '',
        'Use this result to decide whether cost-estimator capacity is the limiting factor.',
        'If a stronger conventional estimator still fails the full recording contract,',
        'the next experiment must address identifiable conditional risk or support, not',
        'repeat another neural architecture/threshold sweep on this exposed readout.', ''])
    return '\n'.join(lines)


def main():
    value = build()
    run.once(run.PUBLIC/'evidence.json', value)
    text = render(value); path = run.PUBLIC/'readout.md'
    if path.exists() and path.read_text() != text:raise ValueError('Immutable published report changed')
    path.write_text(text)
    run.once(run.PUBLIC/'gates.json', value['gates'])
    gate_text = '# Matched Cost Estimator Evidence Gates\n\n'+''.join(f'- {key}: `{result}`\n' for key,result in value['gates'].items())
    (run.PUBLIC/'gates.md').write_text(gate_text)
    print(json.dumps(value['gates']))


if __name__ == '__main__':main()
