"""Rebuild boundary-guard results from public scalar receipts, without weights."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_boundary_guard as run
from scripts import report_m3w_current_bank_deferral as previous
from src.evaluation.m3w_metric_bundle import verify

cell, interval = previous.cell, previous.interval


def build():
    p = run.PUBLIC; reg = json.loads(run.REG.read_text()); cfg = json.loads(run.CONFIG.read_text())
    for rel, digest in dict(reg['bindings'], **reg['dependencies']).items():
        if run.sha(ROOT/rel) != digest: raise ValueError('Registered science changed')
    for name, digest in reg['parent_files'].items():
        if run.sha(run.prior.PUBLIC/name) != digest: raise ValueError('Parent result changed')
    previous.build(); bundle = verify(ROOT, p)
    freeze = json.loads((p/'training_freeze.json').read_text()); complete = json.loads((p/'complete.json').read_text())
    replay = json.loads((p/'verification.json').read_text())
    if (complete['training_freeze_sha256'] != run.sha(p/'training_freeze.json')
            or complete['summary_sha256'] != run.sha(p/'summary.json')
            or replay['summary_sha256'] != complete['summary_sha256']
            or replay['complete_sha256'] != run.sha(p/'complete.json')
            or not replay['all72_new_cost_inference_and_action_scores_exact']):
        raise ValueError('Complete training, readout and numeric verification required')
    fits = {}
    for ref in freeze['fits']:
        if run.sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('New training receipt changed')
        r = json.loads((ROOT/ref['path']).read_text()); key = r['context'], r['seed'], r['arm']
        if (key in fits or r['step'] != r['budget'] or r['parameters'] != 31119 or r['minimum_query_visits'] < 4
                or r['unknown_rows_with_supervised_labels'] or r['registration_sha256'] != run.sha(run.REG)):
            raise ValueError('Full matched causal training required')
        fits[key] = r
    if set(fits) != {(c, s, a) for c in reg['contexts'] for s in reg['seeds'] for a in run.api.ARMS}:
        raise ValueError('All 36 heads required')
    for context in reg['contexts']:
        for seed in reg['seeds']:
            a, b = (fits[context, seed, arm] for arm in run.api.ARMS)
            for k in ('input_hashes', 'draw_hash', 'budget', 'source_query_counts', 'warm_checkpoint', 'row_draws', 'unknown_rows_drawn'):
                if a[k] != b[k]: raise ValueError('Matched training factor changed')
    refits = [json.loads((p/f'first_full_refit_{a}.json').read_text()) for a in run.api.ARMS]
    if (sum(r['step'] for r in fits.values()) != 208320 or freeze['primary_optimizer_updates'] != 208320
            or sum(r['updates'] for r in refits) != 14336 or freeze['exact_full_refit_updates'] != 14336
            or not all(r['scientific_state_exact'] for r in refits)):
        raise ValueError('Complete primary and exact-refit budgets required')
    groups = run.prior.parent.aggregate_groups(run.NAME); old = run.prior.parent.aggregate_groups(run.prior.NAME)
    rows = [groups[Path(ref['path']).name] for ref in complete['groups']]; records = set()
    for r in rows:
        prior = old[f'{r["context"]}_{r["source"]}_seed{r["seed"]}_fit.json']
        if (r['registration_sha256'] != run.sha(run.REG) or r['ids_sha256'] != prior['ids_sha256']
                or r['training_freeze_sha256'] != run.sha(p/'training_freeze.json')
                or r['parent_complete_sha256'] != run.sha(run.prior.PUBLIC/'complete.json')):
            raise ValueError('Original input/result lineage changed')
        for a, b in (('frozen_tdef_guarded', 'tdef_guarded'), ('frozen_own_joint', 'frozen_own_joint'), ('tdef_raw', 'tdef_raw')):
            if r['action_sha256'][a] != prior['action_sha256'][b] or r['result']['policies'][a] != prior['result']['policies'][b]:
                raise ValueError('Original full control differs')
            for rec, v in r['result']['recordings'].items():
                if v[a] != prior['result']['recordings'][rec][b]: raise ValueError('Original record control differs')
        records.update((r['source'], rec) for rec in r['result']['recordings'])
    summary = run.evaluation.summarize(rows, reg['contexts'], cfg)
    summary.update(unique_target_rows=149510, unique_recordings=len(records))
    if len(records) != 74 or summary != json.loads((p/'summary.json').read_text()):
        raise ValueError('Canonical original-cohort summary differs')
    traces = json.loads((p/'training_traces.json').read_text())
    if traces['training_freeze_sha256'] != run.sha(p/'training_freeze.json') or len(traces['cells']) != 36:
        raise ValueError('Complete frozen training trace export required')
    trace_keys = set()
    for r in traces['cells']:
        key = r['context'], r['seed'], r['arm']
        if key in trace_keys or key not in fits or r['checkpoint_sha256'] != fits[key]['checkpoint']['sha256']:
            raise ValueError('Training curve identity changed')
        trace_keys.add(key)
        if r['trace'][0]['step'] != 0 or r['trace'][-1]['step'] != fits[key]['budget']:
            raise ValueError('Only complete fitting curves may be plotted')
    auxiliary = {}
    for a in run.evaluation.NAMES:
        policies = [r['result']['policies'][a] for r in rows]
        cells = [v[a] for r in rows for v in r['result']['recordings'].values()]
        counts = {str(k): sum(v['action_counts'][str(k)] for v in policies) for k in range(-1, 7)}
        total = sum(counts.values()); easy = [100*(v['selected_known_easy_harm_mass']-v['selected_known_easy_benefit_mass'])/v['full_known_easy_reference_mass'] for v in cells if v['full_known_easy_reference_mass'] > 0]
        auxiliary[a] = dict(action_counts=counts, repeated_row_views=total, switch_rate=1-counts['-1']/total,
            neural_row_rate=counts['6']/total, worst_known_easy_degradation_percent=max(easy) if easy else None,
            easy_views_above2=sum(v > 2+1e-10 for v in easy))
    return dict(summary=summary, training=freeze, fits=list(fits.values()), complete=complete, replay=replay,
        auxiliary=auxiliary, traces=traces, bundle=bundle, pilot=json.loads((p/'pilot.json').read_text()))


def render(v):
    p = run.PUBLIC; s = v['summary']; aux = v['auxiliary']
    lines = ['# Two-Sided Boundary Learning: Complete Result', '', '## Evidence Scope', '',
        'fresh_run: 36 real Torch cost fits, two exact complete refits, all 72 original groups and separate full inference/action-score replay.',
        'cached_verified: original forecasts, TDEF ranking, normalizers, 462 causal fields, own-choice warm weights and complete original controls.',
        'not_run: new trajectory/JEPA/Transformer training, independent roles, deployment or submission.',
        'All results are exposed SOURCE fitting/resubstitution. Cost seeds17/29/43 share forecaster43.', '',
        'Both arms keep the same architecture, initial state, complete draws, fixed-initial-choice query loss and full budget.',
        'Only the boundary arm adds a unit log-ratio BCE on the original all/easy 2% boundary. It does not predict a best-expert class.',
        'The auxiliary reference is detached; reference regression and the shared encoder still train. The outputs are not calibrated event probabilities.', '',
        '|Arm|Fits|Updates|Initial anchor objective|Final anchor objective|Initial boundary BCE|Final boundary BCE|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.api.ARMS:
        rows = [r for r in v['fits'] if r['arm'] == arm]
        mean = lambda trace, key: sum(r[trace]['monitor'][key] for r in rows)/len(rows)
        lines.append(f'|{arm}|{len(rows)}|{sum(r["step"] for r in rows)}|{cell(mean("first_trace","anchor_objective"))}|{cell(mean("final_trace","anchor_objective"))}|{cell(mean("first_trace","boundary_objective"))}|{cell(mean("final_trace","boundary_objective"))}|')
    lines += ['', 'Monitor values are fitting losses, not held-out performance or checkpoint selection. No unknown observed-cost targets are supplied.', '',
        '## Full Policy And Required Safety', '',
        '|Policy|Known ADE gain % [conditional 95% CI]|Full-roster FDE gain|Completion-lower utility pp|Risk-supported recording views|Known all/easy violations|Undefined easy support|Worst known easy degradation %|Switch rate|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in run.evaluation.NAMES:
        r = s['safety'][a]; x = aux[a]
        lines.append(f'|{a}|{interval(s["accuracy"][a])}|{interval(s["FDE"][a])}|{interval(s["floor_lower"][a])}|{r["supported_record_views"]}/{r["required_record_views"]}|{r["known_all_risk_violations"]}/{r["known_easy_risk_violations"]}|{r["undefined_easy_record_views"]}|{cell(x["worst_known_easy_degradation_percent"])}|{cell(x["switch_rate"])}|')
    lines += ['', 'Floor-only has no interventions; an undefined selected-risk field is not evidence that the floor causes harm, nor a learned intervention success.', '',
        '## Registered Contrasts', '', 'Primary: boundary minus moment FULL completion-lower utility. Secondary or matched-count results cannot replace it.', '',
        '|Contrast|Scope|Known pp [conditional 95% CI]|Completion-lower pp [conditional 95% CI]|', '|---|---|---:|---:|']
    for name, effects in s['effects'].items():
        for mode, x in effects.items(): lines.append(f'|{name}|{mode}|{interval(x["known_pp"])}|{interval(x["lower_pp"])}|')
    lines += ['', 'Paired lower bounds use disagreement on unknown outcomes, not subtraction of individual lower bounds.',
        'Count matching is utility only. Required empty views stay not estimable, not safely zero. Missing outcomes are uncertainty, not measured harm.', '',
        '## Registered Two-Sided Diagnostic', '', '|Arm|Pointwise blocked views|Singleton joint obstruction views|Rejected safe easy actions|Rejected beneficial safe easy actions|Admitted unsafe actions|', '|---|---:|---:|---:|---:|---:|']
    for a, r in s['guard_diagnostic'].items(): lines.append(f'|{a}|{r["pointwise_blocked_views"]}|{r["native_joint_singleton_proof_views"]}|{r["safe_easy_rejected"]}|{r["beneficial_safe_easy_rejected"]}|{r["known_unsafe_admitted"]}|')
    lines += ['', 'Action occurrences repeat candidates, contexts and seeds. Realized-cost errors are not a unique proof of conditional calibration failure.',
        'Future labels identify these errors only after causal choices are fixed. No oracle-filtered policy is evaluated as a deployable model.', '',
        '## Compute And Checks', '']
    for label, r in (('Actual 400-update pilot', v['pilot']), ('Formal training including full refits', v['training']), ('Complete policy evaluation', v['complete']), ('Separate numerical replay', v['replay'])):
        lines.append(f'{label}: PID {r["pid"]}, {r["seconds"]:.6f} s, peak RSS {r["peak_RSS_bytes"]:,} B.')
    lines += ['', 'Native arm64 CPU4/inter-op1/workers0; atomic 50-update checkpoints, heartbeat, explicit resume. No speed-based budget reduction.', '', '## Fitting Gates', '']
    lines += [f'- {k}: {"pass" if value else "fail"}' for k, value in s['fitting_gates'].items()]
    lines += ['', f'Fitting advancement: {s["fitting_advancement"]}. Independent confirmation: false. Deployment changed: false.', '',
        '![Matched cost-boundary comparison](figures/boundary_guard.svg)', '',
        'The original 149,510 targets, 74 recordings and 12 localities remain. Intervals average repeated views within locality before 3,000 bootstrap draws.',
        'They are conditional development evidence, not independent confirmation. Full-roster FDE stays not estimable where final endpoints are unavailable.',
        'EuropeanSquares detector-silver/oblique local-image, obs8/pred12/stride12 raw frames. No SDDt+50, metric/seconds, human-gold, true3D, foundation, new dynamics or physical-safety claim.',
        'Stage5C execution and SMC remain off.', '']
    (p/'readout.md').write_text('\n'.join(lines)); run.write(p/'supplementary_statistics.json', aux)
    (p/'gates.md').write_text('# Boundary-Guard Fitting Gates\n\n'+'\n'.join(f'- {k}: {v}' for k,v in s['fitting_gates'].items())+
        f'\n\nFitting advancement: {s["fitting_advancement"]}. No independent confirmation or deployment. Stage5C/SMC off.\n')
    run.write(p/'evidence.json', dict(summary_sha256=run.sha(p/'summary.json'), complete_sha256=run.sha(p/'complete.json'),
        verification_sha256=run.sha(p/'verification.json'), new_real_fits=36, primary_updates=208320,
        exact_full_refit_updates=14336, fitting_advancement=s['fitting_advancement'], independent_confirmation=False, deployment_changed=False))


def figure(v):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = v['summary']; names = ('frozen_tdef_guarded', 'moment_guarded', 'boundary_guarded')
    fig, grid = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    axes = grid[0]
    for i, a in enumerate(names):
        x = s['accuracy'][a]; axes[0].bar(i, x['point'], color=('#555555', '#326a95', '#9a6a16')[i], width=.6)
        if x['CI95'] is not None: axes[0].vlines(i, *x['CI95'], color='black')
    axes[0].set_xticks(range(3), ['Frozen guard', 'Moment refit', 'Boundary refit']); axes[0].set_ylabel('Known ADE gain over floor (%)')
    for i, key in enumerate(('known_pp', 'lower_pp')):
        x = s['effects']['boundary_minus_moment']['full'][key]
        if x['CI95'] is not None: axes[1].hlines(i, *x['CI95'], color='#326a95'); axes[1].plot(x['point'], i, 'o', color='#326a95')
    axes[1].set_yticks([0, 1], ['Known', 'Completion lower']); axes[1].set_ylim(-.3, 1.3)
    axes[1].set_xlabel('Boundary minus moment utility (pp)'); axes[1].axvline(0, color='black', linewidth=.7)
    progress = run.api.np.linspace(0, 1, 101)
    for arm, color in (('moment', '#326a95'), ('boundary', '#9a6a16')):
        rows = [r for r in v['traces']['cells'] if r['arm'] == arm]
        for axis, key in zip(grid[1], ('anchor_monitor', 'boundary_monitor')):
            curves = [run.api.np.interp(progress, [t['step']/r['budget'] for t in r['trace']], [t[key] for t in r['trace']]) for r in rows]
            axis.plot(progress, run.api.np.mean(curves, axis=0), color=color, label=arm)
            axis.set_xlabel('Fraction of each complete update budget')
    grid[1, 0].set_ylabel('Same anchor objective'); grid[1, 1].set_ylabel('Boundary BCE (measured in both arms)')
    for axis in grid[1]: axis.legend(); axis.set_title('Fixed fitting-query monitor; not validation')
    for ax in grid.flat: ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exposed SOURCE: frozen ranking, matched cost-boundary loss')
    (run.PUBLIC/'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(run.PUBLIC/'figures/boundary_guard.svg'); fig.savefig(run.PRIVATE/'boundary_guard.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    value = build(); render(value); figure(value)
    print(json.dumps(dict(groups=72, fitting_advancement=value['summary']['fitting_advancement'], independent_confirmation=False)))
