"""Read complete coupled-head evidence; never use partial policy results for tuning."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_coupled_benefit_risk as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation.m3w_metric_bundle import verify


def training():
    reg = json.loads(run.REG.read_text()); freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    for rel, digest in reg['bindings'].items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Scientific registration changed')
    rows = []; states = {}
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Fit receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); cp = r['checkpoint']
        if run.sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Trained weights changed')
        state = run.api.parent.read_checkpoint(ROOT/cp['path'])
        if state['step'] != 2000 or state['unknown_rows_with_loss']: raise ValueError('Full label-safe training required')
        states[r['context'], r['seed'], r['mode']] = state; rows.append(r)
    if len(states) != 36 or freeze['registration_sha256'] != run.sha(run.REG): raise ValueError('All36 frozen fits required')
    for c, seed, _ in states:
        run.api.assert_matched(states[c, seed, run.api.ARMS[0]], states[c, seed, run.api.ARMS[1]])
    for a in run.api.ARMS:
        if not json.loads((run.PUBLIC/('first_full_refit_'+a+'.json')).read_text())['scientific_state_exact']:
            raise ValueError('Both complete exact refits required')
    return freeze, rows, states


def training_report():
    freeze, rows, states = training(); stats = {}
    fields = ('total', 'benefit_MSE', 'pointwise_risk', 'selected_query_score')
    for mode in run.api.ARMS:
        selected = [r for r in rows if r['mode'] == mode]
        stats[mode] = {k: dict(initial_median=float(run.np.median([r['first_trace']['monitor'][k] for r in selected])),
            final_median=float(run.np.median([r['final_trace']['monitor'][k] for r in selected])),
            decreased_heads=sum(r['final_trace']['monitor'][k] < r['first_trace']['monitor'][k] for r in selected), required_heads=18) for k in fields}
    lines = ['# Coupled-head training evidence', '', '## Material Passport', '',
        'fresh_run:36 neural fits,72,000 primary updates and two2,000-update complete exact refits.',
        'cached_verified:original input/features,forecasts,normalizers,draw rules and protected model controls.',
        'This report contains fixed fitting monitors,not policy outcomes,validation,test,calibration or deployment evidence.',
        f"PID{freeze['pid']}:{freeze['seconds']:.6f}s from runtime_ready,peakRSS{freeze['peak_RSS_bytes']}B;initial admission is outside that phase timer.",
        'All18 protected model controls are exact;both modes have the same30,857 parameters and draw hashes.', '',
        '|Mode|Fixed fitting monitor|Initial median|Final median|Decreased heads|', '|---|---|---:|---:|---:|']
    for mode in run.api.ARMS:
        for field in fields:
            s = stats[mode][field]; lines.append(f"|{mode}|{field}|{s['initial_median']:.9f}|{s['final_median']:.9f}|{s['decreased_heads']}/18|")
    lines += ['', 'The total is0.5*(benefit+pointwise risk+selected-query score);the protected benefit term is constant with zero gradient.',
        'Do not compare stochastic batch losses as if they were the same fixed examples. Curves below use each head fixed fitting monitor.',
        'No monitor selects a checkpoint or changes settings. All heads complete exactly2,000 updates.',
        '![Fixed fitting monitors](figures/training_monitors.svg)',
        'Unknown outcomes remain context rows and have zero supervised loss. Repeated draws are not independent samples.',
        'This is cost-head training,not a new trajectory forecast/world-dynamics model. No independent roles,Stage5C or SMC.', '']
    run.write(run.PUBLIC/'training_evidence.json', dict(stats=stats, training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        unknown_draws_by_mode={a: sum(r['unknown_rows_drawn'] for r in rows if r['mode'] == a) for a in run.api.ARMS},
        protected_controls_exact=True, policy_outcomes_used=False, independent_confirmation=False))
    (run.PUBLIC/'training_report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6))
    for ax, field in zip(axes, fields):
        for mode, color in zip(run.api.ARMS, ('#557D86', '#AF3D3D')):
            traces = [v['trace'] for (c, seed, a), v in states.items() if a == mode]
            steps = [t['step'] for t in traces[0]]
            if any([r['step'] for r in t] != steps for t in traces): raise ValueError('Monitor step grids differ')
            values = run.np.array([[r['monitor'][field] for r in t] for t in traces])
            ax.plot(steps, run.np.median(values, 0), color=color, label=mode.replace('_benefit', ''))
        ax.set_title(field.replace('_', ' ')); ax.set_xlabel('Actual updates per head'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Fixed exposed-SOURCE fitting monitors,18 heads per arm (not validation)')
    fig.tight_layout(rect=(0, 0, 1, .9)); (run.PUBLIC/'figures').mkdir(exist_ok=True)
    path = run.PUBLIC/'figures/training_monitors.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'training_monitors.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    return stats


def build():
    freeze, _, _ = training(); p = run.PUBLIC
    complete, summary, replay = [json.loads((p/n).read_text()) for n in ('complete.json', 'summary.json', 'verification.json')]
    if (not replay['all72groups_exact'] or not replay['all72_original_controls_exact']
            or replay['summary_sha256'] != run.sha(p/'summary.json') or replay['complete_sha256'] != run.sha(p/'complete.json')
            or complete['summary_sha256'] != run.sha(p/'summary.json')
            or complete['training_freeze_sha256'] != run.sha(p/'training_freeze.json')):
        raise ValueError('Complete separate full policy replay required')
    bundle = verify(ROOT, p); groups = aggregate_groups(run.NAME)
    rows = [groups[Path(r['path']).name] for r in complete['groups']]
    s = run.evaluation.summarize(rows, json.loads(run.REG.read_text())['contexts'], json.loads(run.CONFIG.read_text()))
    record_count = len({(r['source'], rec) for r in rows for rec in r['result']['recordings']})
    s.update(unique_target_rows=summary['unique_target_rows'], unique_recordings=record_count)
    if s != summary or record_count != 74 or summary['unique_target_rows'] != 149510: raise ValueError('Canonical full cohort differs')
    old = aggregate_groups(run.unit.NAME)
    for ref, r in zip(complete['groups'], rows):
        prior = old[Path(ref['path']).name]
        if r['result']['policies']['protected_joint'] != prior['result']['policies']['set_joint']:
            raise ValueError('Old complete policy control changed')
    return dict(summary=summary, complete=complete, replay=replay, rows=rows, bundle=bundle,
        decomposition=decompose(rows, summary),
        training_freeze=freeze, replay_runtime=json.loads((run.PRIVATE/'verification_runtime.json').read_text()))


def decompose(rows, summary):
    if len(rows) != 72: raise ValueError('Complete72-group result required for descriptive decomposition')
    sites = {}
    for row in rows:
        r = row['result']; p = r['policies']; ref = r['contrasts']['joint']['full']['full_reference_mass_counted_once']
        if ref <= 0: raise ValueError('Observed positive fixed reference required;do not fill missing groups')
        v = [100*(p['released_joint']['policy'][k]-p['protected_joint']['policy'][k])/ref
             for k in ('selected_known_benefit_mass', 'selected_known_harm_mass')]
        sites.setdefault(row['source'], []).append(v)
    by_source = {s: dict(zip(('benefit_delta_pp', 'harm_delta_pp'), run.np.mean(v, 0).tolist())) for s, v in sorted(sites.items())}
    effect = summary['effects']['joint']; discrepancies = []
    for s, v in by_source.items():
        v['known_pp'] = effect['known_pp']['by_locality'][s]
        v['lower_pp'] = effect['lower_pp']['by_locality'][s]
        v['unknown_disagreement_penalty_pp'] = v['known_pp']-v['lower_pp']
        discrepancies.append(abs(v['benefit_delta_pp']-v['harm_delta_pp']-v['known_pp']))
    if max(discrepancies) > 1e-8: raise ArithmeticError('Descriptive mass identity changed;not a risk tolerance')
    return dict(by_source=by_source,
        benefit_delta_pp=float(run.np.mean([v['benefit_delta_pp'] for v in by_source.values()])),
        harm_delta_pp=float(run.np.mean([v['harm_delta_pp'] for v in by_source.values()])),
        unknown_disagreement_penalty_pp=effect['known_pp']['point']-effect['lower_pp']['point'],
        known_positive_sources=sum(v['known_pp'] > 0 for v in by_source.values()),
        lower_positive_sources=sum(v['lower_pp'] > 0 for v in by_source.values()),
        source_count=len(by_source), posthoc_descriptive_not_new_primary=True,
        causal_failure_source_identified=False)


def fmt(v):
    if v['point'] is None or v['CI95'] is None: return 'not_estimable'
    return f"{v['point']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]"


def render(v):
    s = v['summary']; lines = ['# Coupled benefit/risk: complete SOURCE development result', '',
        '## Material Passport', '',
        'fresh_run:36 real Torch cost-head fits,72,000 primary updates,two exact full refits,and complete72-group policy readout/replay.',
        'cached_verified:original past inputs,forecasts,floor,normalizers and old protected full-policy choices.',
        'not_run:new trajectory/dynamics training,independent calibration/confirmation,SDD deployment retest.',
        'All72 protected model predictions and full-joint policy controls are exact. Controls are not relabeled independent tests.', '',
        '## Registered Factor', '',
        'Release only the benefit constraint in the selected-set risk recalibrator. Both models have30,857 parameters,',
        'identical initialization/features/draws and2,000 updates. Protected updates reproduce the old model exactly.',
        'Both optimize0.5*(benefit_MSE+pointwise_risk+selected_query_score);benefit_MSE is constant for the protected arm.',
        'Original reference predictions R/easy_R,all observed evaluation targets,strict2% positive-harm/reference budgets and easy preservation remain unchanged.',
        'Primary:released minus protected FULL-joint completion-lower utility. Matched-count is a secondary utility control,not a safety replacement.', '',
        '|Released minus protected|Known pp [95% CI]|Completion-lower pp [95% CI]|', '|---|---:|---:|']
    for mode in ('joint', 'matched'):
        lines.append(f"|{mode}|{fmt(s['effects'][mode]['known_pp'])}|{fmt(s['effects'][mode]['lower_pp'])}|")
    lines += ['', '|Policy|Known ADE gain over floor (%)|Completion-lower pp over floor|Full-roster FDE|Supported original record-risk views|Known easy-risk violations|Undefined easy|Selected / unknown|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        risk = s['safety'][a]
        lines.append(f"|{a}|{fmt(s['accuracy'][a])}|{fmt(s['floor_lower'][a])}|{fmt(s['FDE'][a])}|{risk['supported_record_views']}/{risk['required_record_views']}|"
            f"{risk['known_easy_risk_violations']}|{risk['undefined_easy_record_views']}|{risk['selected']}/{risk['selected_unknown']}|")
    lines += ['', 'Genuine positive-harm/reference risk is not net error;benefit never cancels its numerator.',
        'The released floor-relative lower point is '+fmt(s['floor_lower']['released_joint'])+' pp;its interval still crosses zero.',
        'Subtracting two individual lower-bound points does not give the paired lower bound. It cannot replace the negative registered paired effect.',
        'All149,510 unique target rows,74 recordings,12 exposed localities and444 reused record/context/critic-seed views remain.',
        'Wholly unknown outcomes remain in bounds and are not safe zeros. Empty matched risk is not_estimable.',
        'The floor makes no intervention;its selected-risk denominator is absent,not evidence that the baseline adds harm.',
        'Unavailable required endpoints keep full-roster FDE not_estimable;available errors remain supplementary.', '',
        '## Fitting Checks', '']
    lines += [f"- {k}: {'pass' if b else 'fail'}" for k, b in s['fitting_gates'].items()]
    lines += ['', f"Advancement:{s['fitting_advancement_gate']}. Deployment unchanged,even if an exploratory fitting check passes.", '',
        '## Limits', '',
        'Intervals are nominal3,000-draw locality bootstraps after averaging reused contexts and critic seeds within12 EXPOSED SOURCE localities.',
        'Not independent confirmation,not whole-pipeline OOF,and not adjusted for adaptive prior research. Three critic seeds share forecast producer43.',
        'Any paired cost-head effect includes changes to predicted utility and admissible harm capacity;it is not isolated proof of physical dynamics.',
        'Record tails,hard/easy pooled known masses,and prediction-only pair-distance contraction are in complete aggregate group receipts.',
        'Pair proxies cover query targets,not all visible agents,and do not certify physical collision safety.',
        'A lower fitting monitor does not establish decision calibration,conformal control or deployment.',
        'Obs8/pred12/rawstride12,oblique detector-silver/local-image pixels,not SDDt50,metric,seconds,human gold,true3D or foundation.',
        'No independent role,Stage5C execution or SMC. Not submission-ready.', '',
        '[Policy figure](figures/policy_result.svg),[training curves](training_report.md),[method and primary-source compatibility](method_note.md),',
        '[frozen protocol](protocol.md),[failure analysis](failure_analysis.md),[reproduction](reproduction_zh.md).', '']
    return '\n'.join(lines)


def extra_documents(v):
    p = run.PUBLIC; s = v['summary']; d = v['decomposition']; cfg = json.loads(run.CONFIG.read_text())
    run.write(p/'descriptive_decomposition.json', d)
    with (p/'locality_effects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source', *next(iter(d['by_source'].values()))], lineterminator='\n')
        writer.writeheader(); writer.writerows(dict(source=k, **r) for k, r in d['by_source'].items())
    lines = ['# Failure analysis: coupled cost capacity is not sufficient', '', '## Material Passport', '',
        'The primary comparison was registered before own training/outcomes. This decomposition was designed after the negative complete readout.',
        'It reduces complete aggregate metrics only;no new labels,features,weights,thresholds or primary tests are introduced.',
        'Initial fits/readout were computed2026-10-30. The complete new-process replay is fresh_run on2026-11-01.', '',
        '## What Worked', '',
        'All36 real neural fits finish. All18 protected controls and both complete full-refits are exact.',
        'The released head learns benefit:all18 fixed benefit monitors decrease;the control stays constant.',
        'Median selected-query monitor improves more than the control. These are fixed fitting diagnostics,not calibration.', '',
        '## What Failed', '',
        'Known full-joint utility changes '+fmt(s['effects']['joint']['known_pp'])+' pp.',
        'The primary full-joint completion-lower contrast is '+fmt(s['effects']['joint']['lower_pp'])+' pp.',
        'The matched-count secondary contrast is also negative:'+fmt(s['effects']['matched']['lower_pp'])+' lower pp.',
        f"Only{d['known_positive_sources']}/{d['source_count']} locality known effects and{d['lower_positive_sources']}/{d['source_count']} lower effects are positive.",
        'Original recording-risk support improves only1/444->6/444;known easy violations remain362 and16 easy views undefined.',
        'Fewer interventions and better aggregate regression loss therefore do not meet the original2% requirements.', '',
        '## Descriptive Error Accounting', '',
        f"At the original source-macro aggregation,selected known benefit changes{d['benefit_delta_pp']:+.9f}pp and positive harm changes{d['harm_delta_pp']:+.9f}pp.",
        'Their difference exactly reproduces the known utility contrast up to floating arithmetic. Benefit lost exceeds harm avoided.',
        f"The unknown-disagreement penalty between known and completion-lower effects is{d['unknown_disagreement_penalty_pp']:.9f}pp.",
        'This uses paired forecast disagreement envelopes,not the number of unknown rows. Fewer unknown switches do not imply a smaller uncertainty penalty.',
        'The released floor-relative lower point improves,but its CI crosses zero. Subtracting individual lower bounds is not a valid paired-effect bound;the registered paired result remains negative.',
        'See all12 sources in locality_effects.csv;no small or adverse locality is removed.', '',
        '## Failure Taxonomy And Remaining Uncertainty', '',
        'Optimization:no stall or failure to update. Engineering:one missing output directory was repaired without changing scientific code.',
        'Capacity:the targeted benefit constraint is released,but that is not enough to meet the decision/safety objective.',
        'Decision target:fixed selected-query supervision is evaluated after the learned predictions alter selection;post-selection error remains a plausible cause,not identified by this comparison.',
        'Uncertainty:unknown futures and selected positive-harm/reference ratios remain problematic. Known easy-risk failures show missing labels alone do not explain everything.',
        'Representation/labels:unobserved intent,detector-silver measurement error and inadequate conditional features are not disentangled.',
        'Generality:exposed SOURCE fitting,one forecast-producer identity and no independent scene result remain limitations.', '',
        '## Decision', '',
        'Do not deploy the released model or raise risk tolerance. Retain old forecasts and current deployment.',
        'Do not infer that further coupling or lower regression loss will solve the task. Prioritize a discriminating policy-conditional calibration or forecast-quality repair,after checking existing attempts.',
        'No new trajectory/world-dynamics contribution,independent confirmation,Stage5C or SMC. The full research goal is unfinished.', '']
    (p/'failure_analysis.md').write_text('\n'.join(lines))
    (p/'model_data_card.md').write_text('\n'.join([
        '# Coupled neural cost-head model/data card', '',
        'Task:predict benefit,harm and easy-harm moments for a frozen seven-action bank;select optional baseline/neural interventions.',
        'This is a neural decision component,not new trajectory dynamics,JEPA pretraining,latent rollout or a foundation world model.',
        'Inputs:existing462-field past trajectory/neighbor/detector/scene/rollout context,old predicted moments,causal masks and query grouping.',
        'Future endpoints,central velocity,test-derived goals and future support are not input fields. Loss targets and fitting-only RMS use supervised training labels.',
        'References R/easy_R remain fixed. Released benefit/harm lie in the declared envelope;this alone is not complete physical or probabilistic moment coherence.',
        'Architecture:width64 selected-set critic and scalar benefit branch;30,857 nominal parameters per arm. Protected branch is unused.',
        'Training:36 fits,3 critic seeds,6 reused source contexts,2,000 updates each;two complete exact refits.',
        'Data:149,510 unique detector-silver target rows,74 recordings,12 exposed European SOURCE localities;whole-pipeline OOF=false.',
        'Observation protocol:8 past/12 future steps,raw stride12,oblique local-image coordinates. No verified metric/seconds/human-gold/true3D claim.',
        'Three critic seeds are not three independently trained forecast producers. There is no independent calibration/confirmation result.',
        'Result:registered primary and original recording-risk requirements fail. Deployment unchanged;not CCF-A ready.',
        'Private data,features,weights and per-row choices are excluded from Git. Public aggregate bundles and source/checkpoint hashes support local reproducibility.',
        'Local native arm64 CPU4/interop1/workers0. No new CREATE job,worker multiprocessing or resource probing.',
        'Stage5C execution=false;SMC=false.', '']))
    (p/'statistical_review.md').write_text('\n'.join([
        '# Statistical review', '',
        'Primary:full-joint released-minus-protected completion-lower utility. Matched-count results are secondary utility controls.',
        'All72 required source/context/critic-seed groups remain;complete private passes verify149,510 unique targets and74 recordings.',
        'Nominal3,000 paired resamples use12 exposed localities after averaging reused contexts and critic seeds within each locality.',
        'Repeated/overlapping rows are not independent. The intervals are conditional fitting evidence,not adaptive-search-adjusted confirmation.',
        'Complete separate inference/allocation/scoring replay establishes execution reproducibility,not new statistical replication.',
        'Known-label utility,unknown-completion bounds,positive-harm/reference risk and net easy degradation are separate estimands.',
        'No benefits cancel positive harm in the2% risk numerator. All444 required recording views remain,including failures and undefined easy views.',
        'Empty selected-risk denominators remain not_estimable. A nonintervening baseline is not thereby labeled harmful.',
        'Full-roster endpoint support is incomplete;FDE is not_estimable rather than an average over an easier observed subset.',
        'Fixed fitting monitors select no checkpoint. Their decreases are not population calibration,validation,test or neural dynamics evidence.',
        'The B/H/disagreement decomposition is explicitly post-hoc descriptive arithmetic with no new primary claim.',
        'No causal mechanism for detector noise,latent intent or domain generalization is established.', '']))
    (p/'paper_results_component.md').write_text('\n'.join([
        '# Development-result component (not the full paper)', '',
        'We compared matched selected-set cost critics that either freeze or jointly adjust predicted benefit while preserving both reference moments.',
        'All36 neural fits completed their fixed update budgets,with exact protected-model controls and complete refits.',
        'Although benefit and selected-query fitting monitors improved,the full-policy known utility contrast was '+fmt(s['effects']['joint']['known_pp'])+' percentage points.',
        'The prespecified completion-lower contrast was '+fmt(s['effects']['joint']['lower_pp'])+' percentage points.',
        'Matched-count utility controls were also negative,and complete recording-level risk remained unsupported.',
        'A post-hoc mass decomposition showed that lost observed benefit exceeded avoided harm;paired unknown-outcome disagreement further widened the lower bound.',
        'These exposed-source,detector-silver,local-image findings reject this simple coupled-moment repair as a sufficient solution.',
        'They do not establish neural dynamics improvement,independent scene generalization or deployment safety.', '']))
    (p/'next_steps.md').write_text('\n'.join([
        '# Next research action', '',
        'This iteration is complete:matched real neural fitting,full policy comparison,exact replay and negative-result reporting.',
        'Do not promote the model or retune its weights,thresholds or baseline windows on this result.',
        'Before another training run,compare any proposed repair with the existing unit,current-policy,query,tail,calibration and forecast-quality attempts.',
        'The key unresolved issue is decision/risk quality after intervention selection,not merely regression fit or benefit/harm representability.',
        'A useful next experiment must distinguish conditional information/forecast quality from policy-conditional risk estimation,with fixed complete cohorts and original2% risk.',
        'Remaining formal evidence:three actual forecast seeds,strong public task-matched controls,isolated scene calibration/confirmation and a complete anonymous paper/supplement.',
        'Independent outcomes stay closed until an authorized protocol is met. Stage5C and SMC remain prohibited. The overall goal remains active.', '']))


def policy_plot(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'svg.hashsalt': run.NAME})
    s = v['summary']; fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    policies = ('protected_joint', 'released_joint', 'protected_matched', 'released_matched')
    for i, a in enumerate(policies):
        r = s['accuracy'][a]
        if r['point'] is None: raise ValueError('Do not plot missing accuracy as zero')
        axes[0].errorbar(r['point'], i, xerr=[[r['point']-r['CI95'][0]], [r['CI95'][1]-r['point']]],
            fmt='o', capsize=3, color='#AF3D3D' if a.startswith('released') else '#557D86')
    axes[0].set_yticks(range(4), [a.replace('_', ' / ') for a in policies]); axes[0].invert_yaxis()
    axes[0].set_title('Known-label policy fitting utility'); axes[0].set_xlabel('ADE gain over floor (%)')
    for i, (mode, field) in enumerate((('joint', 'known_pp'), ('joint', 'lower_pp'), ('matched', 'known_pp'), ('matched', 'lower_pp'))):
        r = s['effects'][mode][field]
        axes[1].errorbar(r['point'], i, xerr=[[r['point']-r['CI95'][0]], [r['CI95'][1]-r['point']]],
            fmt='o', capsize=3, color='#AF3D3D' if field == 'lower_pp' else '#557D86')
    axes[1].set_yticks(range(4), ('Full / known', 'Full / lower', 'Matched / known', 'Matched / lower'))
    axes[1].invert_yaxis(); axes[1].set_title('Released minus protected'); axes[1].set_xlabel('Paired utility difference (pp)')
    d = v['decomposition']; contributions = [d['benefit_delta_pp'], -d['harm_delta_pp'], s['effects']['joint']['known_pp']['point']]
    axes[2].barh(range(3), contributions, color=['#AF3D3D', '#557D86', '#666666'])
    axes[2].set_yticks(range(3), ('Benefit change', 'Avoided harm', 'Net known effect')); axes[2].invert_yaxis()
    axes[2].set_title('Post-hoc observed-mass accounting'); axes[2].set_xlabel('Contribution to known utility (pp)')
    for ax in axes:
        ax.axvline(0, color='#777777', linewidth=.7); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Exposed SOURCE development: better cost fitting does not improve this policy', fontsize=11)
    fig.tight_layout(rect=(0, .13, 1, .9))
    fig.text(.5, .035, 'Nominal 3,000 resamples of 12 exposed localities. Released original risk: only 6/444 supported; no deployment.', ha='center', fontsize=9)
    path = run.PUBLIC/'figures/policy_result.svg'; fig.savefig(path, metadata={'Date': None}, bbox_inches='tight')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(run.PRIVATE/'policy_result.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def report():
    training_report(); v = build(); p = run.PUBLIC
    (p/'readout.md').write_text(render(v))
    extra_documents(v); policy_plot(v)
    run.write(p/'verification_runtime.json', v['replay_runtime'])
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), training_freeze_sha256=run.sha(p/'training_freeze.json'),
        verification_runtime_sha256=run.sha(p/'verification_runtime.json'),
        report_code_sha256=run.sha(Path(__file__)), readout_sha256=run.sha(p/'readout.md'),
        independent_confirmation=False, deployment_changed=False, submission_ready=False))
    print(json.dumps(dict(complete_report=True, fitting_gates=v['summary']['fitting_gates'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--training-only', action='store_true'); a = p.parse_args()
    training_report() if a.training_only else report()
