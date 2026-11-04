"""Report fixed learned-control contrasts without promoting prediction to safety."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_hurdle_strong_reference as study
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_hurdle_reference import loss_interval


def positive(row):
    return row['CI95'] is not None and row['CI95'][0] > 0


def main():
    source, public = study.source, study.PUBLIC
    analysis = source.read(public / 'analysis.json')
    verification = source.read(public / 'verification.json')
    if (not verification['all_arrays_and_two_estimand_metrics_exact']
            or verification['analysis_sha256'] != source.sha(public / 'analysis.json')):
        raise ValueError('Finish exact learned-control replay before reporting')
    effects = analysis['effects']
    decisions = dict(
        C_selected_beats_old_neural_on_conditional_known_MSE=positive(effects['conditional_known']['C_selected']),
        C_selected_beats_old_neural_on_original_bounded_MSE=positive(effects['original_bounded_label']['C_selected']),
        old_neural_better_on_both_estimands=all(
            effects[e]['C_selected']['CI95'] is not None
            and effects[e]['C_selected']['CI95'][1] < 0 for e in effects),
        post_readout_diagnostic=True, targets_and_capacities_not_matched=True,
        new_training=False, new_decisions=False, trajectory_lift_tested=False,
        original_full_risk_repaired=False, deployment_changed=False,
        independent_confirmation=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    source.once(public / 'decision.json', source.stable(decisions))
    records = [dict(site=site, scores=scores)
               for ref in analysis['receipts']
               for site, scores in source.read(source.checked(ref))['by_source'].items()]
    sites = sorted({r['site'] for r in records})
    cfg = source.read(study.run.lineage.parent.CONFIG)
    absolute = {e: {model: source.group_interval(records,
        lambda r, e=e, model=model: r['scores'][e][model]['MSE'], sites, cfg)
        for model in ('prior', 'direct', 'hurdle', 'C_selected', 'old_neural')}
        for e in effects}
    source.once(public / 'summary.json', source.stable(dict(absolute_MSE=absolute,
        effects=effects, decision=decisions, source_groups=23, source_group_views=138)))
    lines = ['# Comparison With The Existing Learned Neural Reference', '',
             'Fresh replay, no new fitting or decisions. All108new reference arrays and18old neural',
             'cost arrays match their completed evaluation hashes; both estimands and all23supported',
             'source groups replay exactly. Three actual seeds; 3000source-group bootstrap resamples',
             'after producer/seed averaging. These are exposed-development diagnostic intervals,',
             'not independent confirmation or multiplicity-adjusted claims.', '',
             '|Estimand|New reference minus old neural: MSE reduction [95%CI]|',
             '|---|---:|']
    for estimand, models in effects.items():
        for model, effect in models.items():
            lines.append(f'|{estimand}: {model}|{loss_interval(effect)}|')
    lines.extend(['', 'Positive means lower normalized ER prediction MSE, not an ADE/FDE percentage.', '',
        'C-selected improvement over the learned control is established on known labels: '
        + str(decisions['C_selected_beats_old_neural_on_conditional_known_MSE']) + '.',
        'C-selected improvement over the learned control under original bounded labels: '
        + str(decisions['C_selected_beats_old_neural_on_original_bounded_MSE']) + '.',
        'Old neural reference is better on both targets with intervals excluding zero: '
        + str(decisions['old_neural_better_on_both_estimands']) + '.', '',
        '**Conditional-known** scores observed eligible rows. **Original-bounded-label** keeps all',
        'eligible rows and the old conservative zero reference contribution for wholly unknown',
        'labels. That zero is not evidence of a negative physical future event. No future label',
        'availability enters inference. The inherited neural R/ER already shares its reference',
        'across all seven candidates; reference sharing is not a new repair.', '',
        'The new heads train on known-label ER only. The old neural head fits multiple bounded',
        'moments, including unknown-reference-zero contributions. These are actual operational',
        'comparators, not matched input, architecture, target or capacity ablations. A loss contrast',
        'cannot identify which difference causes it or justify substituting a conditional mean',
        'into the bounded deployment decoder.', '',
        'C-only selection remains16direct/2hurdle. No held-source model or threshold reselection.',
        'All prior support failures, original2%positive-harm/support and signed easy-net conditions',
        'remain. The original24group primary is not_estimable. Reserved Moving20groups, DUT and',
        'DroneCrowd confirmation stay closed. No new policy, trajectory gain, record-risk guarantee',
        'or submission readiness. No metric/seconds/true3D/foundation/human-gold claim; Stage5C/SMC off.', ''])
    lines.extend(['## Absolute Prediction Losses', '',
        '|Estimand|Model|Source-group mean normalized MSE [95%CI]|', '|---|---|---:|'])
    for estimand, models in absolute.items():
        for model, result in models.items():
            lines.append(f'|{estimand}|{model}|{loss_interval(result)}|')
    text_once(public / 'readout.md', '\n'.join(lines) + '\n')
    per_source = ['# All Sources, Both Reference Targets', '',
                  '|Source|Conditional selected-vs-neural MSE reduction|Bounded selected-vs-neural MSE reduction|',
                  '|---|---:|---:|']
    sites = sorted(effects['conditional_known']['C_selected']['by_source_group'])
    for site in sites:
        values = [effects[estimand]['C_selected']['by_source_group'][site]
                  for estimand in ('conditional_known', 'original_bounded_label')]
        per_source.append('|' + site + '|' + '|'.join(f'{v:+.8g}' for v in values) + '|')
    text_once(public / 'per_source.md', '\n'.join(per_source) + '\n')
    text_once(public / 'gates.md', '''# Execution Is Not A Deployment Pass

|Check|Status|Scope|
|---|---|---|
|216 forests, three seeds,108OOF arrays|complete|Registered internal source-excluded reference study|
|108new arrays and18old neural arrays exact replay|pass|Same H population, both estimands|
|C-selected prediction versus constant prior|positive|Normalized known-reference MSE, not trajectory|
|C-selected prediction versus existing learned reference|failed|Both fixed target intervals favor the old output|
|Two-part head versus direct regression|failed|The fixed grouped interval favors direct|
|Event Brier benefit|not established|Interval crosses zero|
|Record-risk repair|not_run by these heads|Existing full-risk failures are not recertified|
|Trajectory or intervention benefit|not_run|No decisions changed|
|Independent confirmation|not_run|All23supported groups are exposed development|
|Original24group primary|not_estimable|Unsupported group retained|
|Deployment/submission readiness|false|Prediction-only study cannot clear full contract|
|Stage5C / SMC|false / false|No execution|

There is no total pass count mixing execution and scientific success. Existing
2%positive-harm/support, easy-net and empty-control requirements are unchanged.
Source-time, physical metric and independent-data claims remain unsupported.
''')
    print(source.json.dumps(dict(decision=decisions,
        effects={e: {m: dict(point=r['point'], CI95=r['CI95']) for m, r in models.items()}
                 for e, models in effects.items()}), sort_keys=True))


if __name__ == '__main__':
    main()
