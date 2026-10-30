"""Report frozen causal-anchor controls and a separately registered post-hoc audit."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_position_anchor as run
from scripts import diagnose_m3w_position_anchor as diagnosis
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def checked(study, folder, flags):
    complete, summary, replay = [json.loads((folder/n).read_text())
        for n in ('complete.json', 'summary.json', 'verification.json')]
    reg = json.loads((folder/'registration.json').read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Scientific registration changed: '+rel)
    if (any(not replay[k] for k in flags) or replay['summary_sha256'] != run.sha(folder/'summary.json')
            or replay['complete_sha256'] != run.sha(folder/'complete.json')
            or complete['summary_sha256'] != run.sha(folder/'summary.json')):
        raise ValueError('Complete separate replay required')
    bundle = verify(ROOT, folder); groups = aggregate_groups(study)
    rows = [groups[Path(ref['path']).name] for ref in complete['groups']]
    if len(rows) != 24 or bundle['groups'] != 24: raise ValueError('Complete24 required')
    if any(row['registration_sha256'] != run.sha(folder/'registration.json') for row in rows):
        raise ValueError('Group registration differs')
    return dict(complete=complete, summary=summary, verification=replay, registration=reg,
        replay_runtime=json.loads((folder/'verification_runtime.json').read_text()), rows=rows, bundle=bundle)


def validate_summary(rows, summary):
    canonical = run.evaluation.summarize(rows, json.loads(run.REG.read_text())['contexts'],
        json.loads(run.CONFIG.read_text()))
    # The public groups establish record identities, not private row-set uniqueness.
    records = {(r['source'], rec) for r in rows for rec in r['result']['recordings']}
    views = sum(len(r['result']['recordings']) for r in rows)
    if summary['unique_recordings'] != len(records) or summary['record_context_views'] != views:
        raise ValueError('Recording metadata differs')
    if summary['unique_target_rows'] != 149510:
        raise ValueError('Frozen runner private row-set assertion differs')
    canonical.update(unique_target_rows=summary['unique_target_rows'], unique_recordings=len(records), record_context_views=views)
    if canonical != summary: raise ValueError('Canonical aggregate reduction differs')


def build():
    value = checked(run.NAME, run.PUBLIC, ('all24groups_exact', 'all48_raw_controls_exact', 'all24_identity_controls_exact'))
    validate_summary(value['rows'], value['summary'])
    child = checked(diagnosis.NAME, diagnosis.PUBLIC, ('all24groups_exact', 'all24_parent_controls_exact'))
    for key, path in (('parent_registration_sha256', run.REG), ('parent_summary_sha256', run.PUBLIC/'summary.json'),
                      ('parent_complete_sha256', run.PUBLIC/'complete.json'), ('parent_verification_sha256', run.PUBLIC/'verification.json')):
        if child['registration'][key] != run.sha(path): raise ValueError('Diagnostic parent binding differs')
    if diagnosis.summarize(child['rows']) != child['summary']: raise ValueError('Diagnostic reduction differs')
    value['mechanism'] = child
    return value


def fmt(v):
    if v['point'] is None or v['CI95'] is None: return 'not_estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"


def render(value):
    s = value['summary']; c = value['complete']; replay = value['replay_runtime']
    lines = ['# Causal position anchor: complete negative control', '',
        '## Evidence Roles', '',
        'fresh_run:24 full past-only forecast/error groups and a separate complete24-group replay.',
        'cached_verified:original inputs,fit partitions,floor,physical predictions and producer43 identity.',
        'not_run:no new neural training,independent calibration/confirmation,new RGB acquisition or deployment.',
        'The separate error-geometry diagnosis is post-hoc,not another primary result.', '',
        f"Readout PID{c['pid']}:{c['seconds']:.6f}s,peakRSS{c['peak_RSS_bytes']}B.",
        f"Replay PID{replay['pid']}:{replay['seconds']:.6f}s,peakRSS{replay['peak_RSS_bytes']}B.",
        'All24 groups,24 row/support controls and48 original raw path/ADE/FDE controls agree exactly.',
        f"{s['unique_target_rows']:,} unique target rows,{s['unique_recordings']} recordings,{s['record_context_views']} reused record/context views;12 exposed localities.",
        'Unique row count is a private ID-set assertion in both full runs,bound by replay receipts;it is not inferred from public row sums.', '',
        '## Registered Comparison', '',
        'Hold OLS4/8 velocity fixed. Raw=current last observation+t*v;fitted=history mean+(k-1)/2*v+t*v.',
        'Only the current-position anchor changes. Both use past observations only. Raw trajectories and costs reproduce originals exactly.',
        'Keep the original causal eligibility and floor fallback,including each path exact-no-op exclusion.',
        'This is not a matched-count selector. Counts happen to agree here;that is not the experimental constraint.',
        'Primary:OLS8 fitted-minus-raw completion-lower utility. OLS4 is prespecified secondary,not a replacement winner.', '',
        '|Fitted minus raw|Known utility pp [nominal95% CI]|Completion-lower pp [nominal95% CI]|',
        '|---|---:|---:|']
    for k in ('8', '4'):
        lines.append(f"|OLS{k}|{fmt(s['effects'][k]['known_pp'])}|{fmt(s['effects'][k]['lower_pp'])}|")
    lines += ['', 'Both effects are negative in every one of the12 exposed localities. No locality is discarded.', '',
        '|Candidate|Gated ADE gain over floor (%)|Ungated forecast ADE gain (%)|Full-roster FDE gain|Original risk support|Known all/easy violations|',
        '|---|---:|---:|---|---:|---:|']
    for a in run.evaluation.NAMES:
        r = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{fmt(s['ungated_accuracy'][a])}|{fmt(s['FDE'][a])}|"
            f"{r['supported_record_views']}/{r['required_record_views']}|{r['known_all_risk_violations']}/{r['known_easy_risk_violations']}|")
    lines += ['', 'Each candidate makes298,325 interventions over reused row/context views,including6,280 with wholly unknown futures.',
        'All4 candidates have0 undefined easy-risk recording views;their failures are actual risk failures,not empty-control passes.',
        'Original2% all/easy positive-harm/reference inequalities and easy-degradation requirements are unchanged.',
        'Benefit does not cancel positive harm in the risk numerator. Gated here means original eligibility,not a calibrated safe selector.',
        'Missing endpoints make the full-roster FDE aggregate not_estimable. Available endpoint errors remain in all group receipts and tail tables.', '',
        '## Supplementary Hard/Easy Results', '',
        'These are pooled known error masses over reused row/context views,not the locality-macro primary or independent samples.',
        '|Candidate|Hard ADE gain (%)|Easy ADE gain (%)|', '|---|---:|---:|']
    for a in run.evaluation.NAMES:
        lines.append(f"|{a}|{s['subsets']['hard']['policies'][a]['ADE_gain_percent']:+.6f}|"
            f"{s['subsets']['easy']['policies'][a]['ADE_gain_percent']:+.6f}|")
    lines += ['', 'All148 recording/context tail views,each for4 candidates,are retained in [record_tail_table.csv](record_tail_table.csv).',
        'All24 source/context pair-path proxies are in [pair_proxy_table.csv](pair_proxy_table.csv).',
        'Tail p95 uses observed native local-image pixel errors within each record,not pooled physical safety.',
        'Pair-distance contraction is a prediction-only,query-target proxy. It is not collision frequency,all-visible-agent coverage or physical safety.', '',
        '## Fitting Checks', '']
    lines += [f"- {key}: {'pass' if passed else 'fail'}" for key, passed in s['fitting_gates'].items()]
    lines += ['', f"{sum(s['fitting_gates'].values())}/{len(s['fitting_gates'])} checks pass. Advancement=False;deployment unchanged.", '',
        '## Statistical And Claim Limits', '',
        'Nominal3,000-draw paired locality bootstrap averages reused contexts within each of12 exposed SOURCE localities first.',
        'Producer43 identifies the same fixed source forecaster;these deterministic predictions are not three new forecasting seeds.',
        'This is not independent confirmation or whole-pipeline out-of-fold evaluation. Intervals are not corrected for the long adaptive research history.',
        'All unknown labels and failed recording views remain. Completion bounds do not fill missing interior labels.',
        'A failed blind fitted-position control does not prove every selectively chosen fitted prediction must fail.',
        'It does reject this registered simple state-estimation repair on this fitting cohort. Raw anchoring was a modeling choice,not a demonstrated implementation bug.',
        'Obs8/pred12/rawstride12,oblique detector-silver/local-image pixels,not SDDt50,metric,seconds,human gold,true3D or foundation success.',
        'Independent calibration/confirmation,Stage5C and SMC remain closed. No stronger deployable model or submission-ready claim.', '',
        '[Post-hoc mechanism](mechanism/readout.md),[failure analysis](failure_analysis.md),[figure](figures/position_anchor_result.svg),',
        '[model/data card](model_data_card.md),[statistical review](statistical_review.md),[reproduction](reproduction_zh.md).', '']
    return '\n'.join(lines)


def render_mechanism(value):
    v = value['mechanism']; s = v['summary']
    lines = ['# Position-shift error geometry (post-hoc)', '',
        'This arithmetic was defined after the parent negative result and registered before its own values.',
        'No models,policies,thresholds or scientific primary change. It is not ADE and does not train an estimator.',
        f"Fresh24-group accounting:{v['complete']['seconds']:.6f}s;separate full replay:{v['replay_runtime']['seconds']:.6f}s.",
        'All24 original parent results/history controls and all24 diagnostic groups agree exactly.', '',
        'For raw error r and fitted-minus-raw position shift d,',
        '`||r+d||^2 - ||r||^2 = 2*r.dot(d) + ||d||^2`.',
        'Each term uses frozen causal row scale squared,available-step means within rows,known-row means within views,',
        'then context averaging within source and macro averaging over12 sources. Unknown rows are not zero.', '',
        '|Window|Squared-error change|Cross term|Squared shift|Known row views|Unknown row views|Increased-error known row views|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k, w in s['windows'].items():
        m = w['moments']; lines.append(f"|{k}|{m['direct']['mean']:.9f}|{m['cross']['mean']:.9f}|{m['squared_shift']['mean']:.9f}|"
            f"{w['known_row_views']}|{w['unknown_row_views']}|{w['squared_error_increased_row_views']}|")
    lines += ['', 'All12 source means of both cross term and direct error change are positive for both windows.',
        'The shift tends to reinforce the existing error on these observed labels,not merely add its nonnegative squared length.',
        'This is error accounting,not a causal decomposition of measurement noise or motion. The noise/process split is not identified.', '',
        '## Past-Only Geometry', '',
        'For an exact quadratic history with second difference a,the fitted-current minus last-position shift is',
        '`-a*(k-1)*(k-2)/12`: -0.5*a forOLS4 and -3.5*a forOLS8.',
        'This establishes a possible lag mechanism under that synthetic model,not evidence that the real data have constant acceleration.',
        'On real gated predictions the shift opposes the last past second difference in:', '']
    for k, w in s['windows'].items():
        lines.append(f"- OLS{k}:{w['opposite_past_acceleration_row_views']}/{w['active_past_acceleration_row_views']} active reused row views;")
    lines += ['', 'Past second differences in detector tracks are not verified physical acceleration. Neither their direction nor future label errors identify true motion versus observation noise.',
        'Per-step support and arithmetic terms remain in all24 group receipts. Different masks prevent treating per-step differences as matched horizon effects.',
        'Floating identity residual checks scale with floating-point arithmetic only;they do not relax a risk inequality.',
        'Wholly unknown labels remain absent. This diagnostic uses future labels only in error accounting,never as model inputs.',
        'No optimizer updates,no independent evidence,no deployment,no Stage5C,no SMC. Parent conclusion remains negative.', '']
    return '\n'.join(lines)


def tables(value):
    p = run.PUBLIC; tail_rows = []; pair_rows = []
    for row in value['rows']:
        r = row['result']; prefix = dict(context=row['context'], source=row['source'])
        for record, candidates in sorted(r['record_tails'].items()):
            for a, outcomes in candidates.items():
                item = dict(prefix, recording=record, candidate=a)
                for metric, values in outcomes.items():
                    item.update({metric+'_'+key: v for key, v in values.items()})
                tail_rows.append(item)
        q = r['pair_proxy']
        pair_rows.append(dict(prefix, pair_requested_step_views=q['pair_requested_step_views'],
            **q['mean_positive_distance_contraction_over_frozen_scale']))
    for filename, rows in (('record_tail_table.csv', tail_rows), ('pair_proxy_table.csv', pair_rows)):
        with (p/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    return len(tail_rows), len(pair_rows)


def documents(value):
    p = run.PUBLIC; s = value['summary']
    (p/'readout.md').write_text(render(value)); (p/'mechanism/readout.md').write_text(render_mechanism(value))
    (p/'failure_analysis.md').write_text('\n'.join([
        '# Failure analysis: position estimation is not repaired by ordinary intercept smoothing', '',
        '## Established', '',
        'The registered OLS8 fitted-position repair worsens the primary completion-lower utility by '+fmt(s['effects']['8']['lower_pp'])+' pp.',
        'OLS4 also worsens. The direction is negative in all12 exposed source localities,not only the worst outlier.',
        'Both raw and fitted direct candidates fail all148 original recording-risk views. A large known-hard pooled benefit for rawOLS4 does not rescue easy harm.',
        'The raw predictors,costs,row/support identities and entire separate replay are exact. No implementation discrepancy was detected by those controls.', '',
        '## Mechanism Supported Only As Error Geometry', '',
        'The post-hoc cross term is positive in every source for both windows. The fitted anchor reinforces existing observed error on average.',
        'The longer linear fit can lag changing motion in a synthetic quadratic example. Detector noise,real turns,association errors and process mismatch remain competing explanations.',
        'Measurement noise was not independently measured. A negative control is not a verified noise model or proof that all state estimation is futile.', '',
        '## What Is Not Fixed', '',
        'Forecast quality,source-selection risk and independent-scene calibration are distinct. This intercept control supplies no new learned neural dynamics.',
        'Blindly replacing a floor by a single OLS arm is not a learned selective switch. Selective opportunity could remain,but was not validated here.',
        'Missing full-roster endpoints,source exposure,one forecast-producer identity,record-level safety and weak neural marginal value are still research gaps.', '',
        '## Action', '',
        'Keep original raw-position forecasts and the previous deployment unchanged. Do not tune a new window,clip or risk tolerance on these outcomes.',
        'The next method change should address an independently testable representation/forecast or positive-harm estimation deficit,not rename this failed control.',
        'Register a new hypothesis and training-only selection before its own outcomes;retain all original rows and risk requirements.',
        'No independent roles,Stage5C or SMC are opened. This is exposed SOURCE detector-silver local-image research,not metric/seconds/true3D/foundation success.', '']))
    (p/'model_data_card.md').write_text('\n'.join([
        '# Causal-anchor control model/data card', '',
        'Model:closed-form ordinary least-squares velocity over last4/8 of8 observed2D positions.',
        'Compared state estimate:raw last observation versus causal fitted line evaluated at current observation.',
        'Output:12 deterministic future local-image2D positions. No learned weights,no neural training,no target used for inference.',
        'Candidate inputs:history only. Original past-only eligibility/floor and producer43 identity remain fixed.',
        'No new scene RGB,goal endpoints,central velocity or future support is supplied as input.', '',
        'Data:149,510 unique detector-silver target rows,74 unique recordings,12 exposed SOURCE localities,24 reused fitting views.',
        'Obs8/pred12/rawstride12. Local image coordinates,oblique EuropeanSquares;not the SDD raw-frame t50 benchmark.',
        'No verified homography,scale,seconds,physical safety,true3D or foundation status. Missing labels retain valid masks.',
        'Future labels are error targets only. Unknown-completion envelopes are not imputed trajectories.', '',
        'Result:registered fitted-position repair negative,4/7 fitting checks pass,0/148 supported original record/context risk views for all4 arms.',
        'Independent evidence:new none. External/SDD deployment safety:new not_run. Decision:no deployment change.',
        'Arithmetic diagnosis:post-hoc,error identity only,not a new primary metric or motion/noise identification.',
        'Runtime:native arm64 CPU4/interop1/workers0,no resource probing,multiprocessing or remote scheduler changes.',
        'Reproducibility:registered source/config/test hashes,immutable24group receipts,aggregate ZIP and complete separate prediction/replay.',
        'Public distribution:code,protocols,aggregated metrics and report only;private trajectories/caches/media/weights excluded.',
        'Stage5C execution=false;SMC=false;submission-ready=false.', '']))
    (p/'paper_results_component.md').write_text('\n'.join([
        '# Results component (development control,not a complete manuscript)', '',
        'We tested whether smoothing current position,while holding history-based OLS velocity fixed,improves deterministic trajectory prediction.',
        'In the prespecified eight-observation control,the fitted anchor reduced known-label utility by '+fmt(s['effects']['8']['known_pp'])+' percentage points.',
        'The primary completion-lower contrast was '+fmt(s['effects']['8']['lower_pp'])+' percentage points.',
        'The four-observation secondary control was also negative. Both contrasts were negative in each of12 exposed development localities.',
        'All24 groups were recomputed in a separate process with exact raw-path/cost and summary agreement.',
        'Neither raw nor fitted blind OLS policy met the original2% positive-harm/reference requirements in any of148 required recording/context views.',
        'A post-hoc squared-error identity showed a positive residual/shift cross term in each locality,consistent with error reinforcement but not identifying observation noise or physical motion.', '',
        'Intervals are nominal3,000-draw locality bootstraps conditional on exposed SOURCE development,not independent confirmation or multi-forecaster-seed evidence.',
        'The complete-roster FDE aggregate is not estimable because required endpoint support is missing;available endpoints remain supplementary.',
        'These detector-silver,local-image,past-only controls reject a simple fitted-anchor repair;they do not establish a new neural model or deployment.', '']))
    (p/'statistical_review.md').write_text('\n'.join([
        '# Statistical review', '',
        'Primary was fixed before own outcomes:OLS8 fitted-minus-raw completion-lower utility. OLS4 is secondary,not a selected substitute.',
        'All24 context/source views remain,including all149,510 unique target rows and74 recordings verified in both complete private-ID passes.',
        'Pairing:identical target IDs,masks,past slopes,floor and causal eligibility. It is not a count-matched policy design.',
        'Bootstrap:3,000 draws of12 exposed SOURCE localities after within-locality averaging of reused contexts.',
        'The seed43 label is forecast-producer identity,not an additional randomization factor. No three-forecaster-seed claim.',
        'Overlapping rows,repeated contexts and recording/context views do not create independent sample size.',
        'The nominal interval excludes zero negatively;adaptive experiment history and exposed fitting data prevent confirmatory generalization.',
        'Known utility is a ratio to known floor reference. Completion-lower is a bounded unknown-outcome contrast,not missing-label imputation.',
        'Original all/easy positive-harm/reference risk remains separate from net utility and from any pooled easy average.',
        'All148 record/context risk views fail in each arm;undefined easy count is0. Empty comparisons are never safe zero.',
        'Full-roster FDE is not_estimable;do not average only available records and present that as the required macro result.',
        'Supplementary hard/easy rows use repeated-view pooled masses. Tail and pair-path metrics have separate units and coverage.',
        'Post-hoc error geometry uses normalized squared error,not ADE. The cross term is descriptive,not causal noise/process attribution.',
        'No thresholds,baseline windows or model checkpoints are selected on these own outcomes. No independent calibration or confirmation was consumed.', '']))
    (p/'next_steps.md').write_text('\n'.join([
        '# Next research step', '',
        'Completed:the causal fitted-position control and separately registered error identity both have exact complete replays.',
        'Rejected:blindly replacing raw last-position anchors by ordinary linear fitted anchors as the simple repair.',
        'Unchanged:existing forecast bank/deployment,original2% full-record risk,all missing-label restrictions and closed independent roles.', '',
        'Prioritize a discriminating learned-forecast or risk-estimation hypothesis,not more arbitrary windows or decoder thresholds.',
        'Before new training,check existing attempts so a different name does not disguise the same loss/representation.',
        'A candidate needs real neural lift beyond reoptimized strong physical controls,non-destructive easy behavior,and usable risk estimates.',
        'Only after source evidence supports that mechanism should whole-pipeline isolated calibration and held-out scene evaluation be consumed.',
        'Formal paper gaps remain:three genuine forecast seeds,strong public task-matched controls,independent new-scene evidence and complete anonymous paper/supplement.',
        'No matched image modality was added in this control;do not claim multimodal causality from architecture alone.',
        'This iteration is completed research work,not fulfillment of the full CVPR goal. Stage5C and SMC remain prohibited.', '']))


def plot(value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    s = value['summary']
    for i, a in enumerate(run.evaluation.NAMES):
        v = s['accuracy'][a]
        axes[0].errorbar(v['point'], i, xerr=[[v['point']-v['CI95'][0]], [v['CI95'][1]-v['point']]],
            fmt='o', color='#A13737' if a.endswith('fitted') else '#587A81', capsize=3)
    axes[0].set_yticks(range(4), ('OLS4 / raw', 'OLS4 / fitted', 'OLS8 / raw', 'OLS8 / fitted'))
    axes[0].invert_yaxis(); axes[0].set_title('Gated known-label fitting accuracy'); axes[0].set_xlabel('ADE gain over floor (%)')
    for i, (k, b) in enumerate((('8', 'known_pp'), ('8', 'lower_pp'), ('4', 'known_pp'), ('4', 'lower_pp'))):
        v = s['effects'][k][b]
        axes[1].errorbar(v['point'], i, xerr=[[v['point']-v['CI95'][0]], [v['CI95'][1]-v['point']]],
            fmt='o', color='#A13737' if b == 'lower_pp' else '#587A81', capsize=3)
    axes[1].set_yticks(range(4), ('OLS8 / known', 'OLS8 / lower', 'OLS4 / known', 'OLS4 / lower'))
    axes[1].invert_yaxis(); axes[1].set_title('Fitted minus raw: paired effect'); axes[1].set_xlabel('Utility difference (pp)')
    ms = value['mechanism']['summary']['windows']; cross = [ms[k]['moments']['cross']['mean'] for k in ('4', '8')]
    shift = [ms[k]['moments']['squared_shift']['mean'] for k in ('4', '8')]
    axes[2].barh([0, 1], cross, color='#A13737', label='Cross term')
    axes[2].barh([0, 1], shift, left=cross, color='#D6B455', label='Squared shift')
    axes[2].set_yticks([0, 1], ('OLS4', 'OLS8')); axes[2].invert_yaxis()
    axes[2].set_title('Post-hoc error identity (not ADE)'); axes[2].set_xlabel('Scaled squared-error change'); axes[2].legend(loc='upper right')
    for ax in axes:
        ax.axvline(0, color='#777777', linewidth=.7); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE development: causal fitted-position control is negative', fontsize=11)
    fig.tight_layout(rect=(0, .13, 1, .91))
    fig.text(.5, .035, 'Nominal 3,000-draw intervals over 12 exposed localities. All four policies: 0/148 original risk views supported.', ha='center', fontsize=9)
    folder = run.PUBLIC/'figures'; folder.mkdir(exist_ok=True)
    path = folder/'position_anchor_result.svg'; fig.savefig(path, bbox_inches='tight', metadata={'Date': None})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'position_anchor_result.png', bbox_inches='tight', dpi=160); plt.close(fig)


def main():
    v = build(); documents(v); tails, pairs = tables(v); plot(v)
    run.write(run.PUBLIC/'evidence.json', dict(summary_sha256=run.sha(run.PUBLIC/'summary.json'),
        complete_sha256=run.sha(run.PUBLIC/'complete.json'), verification_sha256=run.sha(run.PUBLIC/'verification.json'),
        mechanism_summary_sha256=run.sha(diagnosis.PUBLIC/'summary.json'),
        mechanism_verification_sha256=run.sha(diagnosis.PUBLIC/'verification.json'),
        report_code_sha256=run.sha(Path(__file__)), report_sha256=run.sha(run.PUBLIC/'readout.md'),
        record_tail_rows=tails, source_proxy_rows=pairs, new_training_updates=0,
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(report_generated=True, record_tail_rows=tails, source_proxy_rows=pairs,
        fitting_checks_passed=sum(v['summary']['fitting_gates'].values()), advancement=False, deployment_changed=False)))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
