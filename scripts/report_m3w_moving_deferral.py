"""Audit and report the frozen public deferral control without changing policies."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_public_deferral as run
from scripts.report_m3w_moving_harm_refinement import text_once
from scripts.report_m3w_moving_amplitude_cost import interval

source, PUBLIC = run.source, run.PUBLIC


def audit_fits():
    freeze = source.read(PUBLIC / 'training_freeze.json')
    if freeze['complete_heads'] != 54 or freeze['primary_updates'] != 108000 or not freeze['all_before_validation']:
        raise ValueError('Every registered full-budget fit must be frozen')
    parents = {}
    for ref in source.read(run.old.PUBLIC / 'training_freeze.json')['fits']:
        row = source.read(source.checked(ref))
        if row['replica'] == 0:
            parents[row['job'], row['part']] = row
    reg = source.read(run.REG)
    rows = []
    for ref in freeze['fits']:
        row = source.read(source.checked(ref))
        state = run.api.base.read_checkpoint(source.checked(row['checkpoint']))
        old = run.api.base.read_checkpoint(source.checked(parents[row['job'], row['part']]['checkpoint']))
        if (row['steps'] != 2000 or state['step'] != 2000 or state['budget'] != 2000 or state['parameters'] != 27969
                or state['identity']['registration_sha256'] != source.sha(run.REG)):
            raise ValueError('Full budget, registered model and schema required')
        for name in ('preprocess', 'sampler', 'fit_ids', 'source_sites', 'draw_hash'):
            run.api.base.exact(state[name], old[name])
        parts = reg['partitions'][row['job']]
        expected = sorted(parts[row['part']]['names'])
        if sorted(state['source_sites']) != expected or sorted(state['preprocess']['source_names']) != expected:
            raise ValueError('Only own original inner-part sources may fit anything')
        if any(int(v['step']) != 2000 for v in state['optimizer']['state'].values()):
            raise ValueError('Actual optimizer states must confirm all updates')
        if not all(run.api.torch.isfinite(v).all() for v in state['model'].values()):
            raise ValueError('Finite learned model required')
        if not any(not run.api.torch.equal(v, state['initial_model'][k]) for k, v in state['model'].items()):
            raise ValueError('Do not report an unchanged initialization as trained')
        rows.append(dict(job=row['job'], part=row['part'], steps=state['step'],
            parameters=state['parameters'], checkpoint=row['checkpoint'],
            own_part_preprocess_sampling_exact=True, optimizer_complete=True))
    first = source.read(source.checked(freeze['fits'][0]))
    repeated = source.read(PUBLIC / 'first_full_refit.json')
    run.api.exact_state(run.api.base.read_checkpoint(source.checked(first['checkpoint'])),
        run.api.base.read_checkpoint(source.checked(repeated['checkpoint'])))
    out = dict(full54_complete=True, primary_updates=108000, first_full_refit_exact=True,
        rows=rows, training_freeze=source.artifact(PUBLIC / 'training_freeze.json'))
    source.once(PUBLIC / 'fit_audit.json', out)
    return out


def outcome(analysis, decision):
    primary = analysis['nonempty_matched']['public']['completion_lower_pp']
    positive = primary['CI95'] is not None and primary['CI95'][0] > 0
    risk = analysis['safety']['public_veto']['full_original_risk_pass']
    if bool(positive and risk) != decision['development_criteria_pass']:
        raise ValueError('The report cannot override the frozen scientific decision')
    return ('Required nonempty matched utility is not estimable, not passed.' if primary['CI95'] is None
        else 'Required nonempty matched utility has '+('a positive' if positive else 'no positive')+' lower confidence limit.') + (
        ' All original full-policy risk/support requirements pass.' if risk
        else ' Original full-policy risk/support requirements fail.')


def risk_diagnostics(groups):
    rows = [r for g in groups for r in g['rows']]
    result = {}
    for arm in run.POLICIES:
        views = [v for r in rows for v in r['policies'][arm]['recordings'].values()]
        defined = [100*v['known_easy_net_degradation'] for v in views if v['known_easy_net_degradation'] is not None]
        if len(views) != 696:
            raise ValueError('Do not drop required recording views')
        result[arm] = dict(required_record_views=696,
            worst_defined_signed_easy_net_percent=max(defined) if defined else None,
            undefined_full_known_easy_net_views=sum(v['known_easy_net_degradation'] is None for v in views),
            no_intervention_views=sum(v['selected'] == 0 for v in views))
    return result


def figures(fits, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'm3w-moving-public-deferral-v1', 'font.size': 9})
    if len(fits) != 54:
        raise ValueError('Plot all54 fits')
    steps = [r['step'] for r in fits[0]['trace']]
    if any([r['step'] for r in f['trace']] != steps for f in fits):
        raise ValueError('Same actual training-update grid required')
    losses = source.np.array([[r['total'] for r in f['trace']] for f in fits])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    lo, mid, hi = source.np.quantile(losses, [.1, .5, .9], axis=0)
    axes[0].plot(steps, mid, color='#176b8c')
    axes[0].fill_between(steps, lo, hi, color='#176b8c', alpha=.15)
    axes[0].set(xlabel='Updates per model', ylabel='Fitting surrogate loss', title='54 fits: median and 10th-90th percentiles')
    for y, name in enumerate(run.CONTRASTS):
        result = analysis['direct_contrasts'][name]['known_pp']
        if result['CI95'] is not None:
            axes[1].plot(result['CI95'], [y, y], color='#b25140')
            axes[1].plot(result['point'], y, 'o', color='#b25140')
        else:
            axes[1].text(0, y, 'NA')
    axes[1].set_yticks([0, 1], ['Direct routing', 'Shared risk veto'])
    axes[1].set_ylim(1.5, -.5); axes[1].axvline(0, color='.5', lw=1)
    axes[1].set(xlabel='Public minus point, known effect (pp)', title='Development contrast, not confirmation')
    names = ('point_proposer', 'point_veto', 'public_router', 'public_veto')
    axes[2].barh(range(4), [analysis['safety'][a]['known_easy_violations'] for a in names], color='#967048')
    axes[2].set_yticks(range(4), ['Point', 'Point + veto', 'Public', 'Public + veto'])
    axes[2].invert_yaxis(); axes[2].set(xlabel='Violations among 696 repeated record views', title='Original 2% easy-risk limit')
    fig.tight_layout()
    public, private = PUBLIC / 'figures', run.PRIVATE / 'figure_previews'
    public.mkdir(exist_ok=True); private.mkdir(exist_ok=True)
    svg = public / 'training_and_risk.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(private / 'training_and_risk.png', dpi=160)
    plt.close(fig)


def main():
    audit_fits()
    analysis = source.read(PUBLIC / 'analysis.json')
    verification = source.read(PUBLIC / 'verification.json')
    decision = source.read(PUBLIC / 'decision.json')
    if (not verification['all108_policy_blocks_and_prediction_hashes_exact']
            or verification['analysis_sha256'] != source.sha(PUBLIC / 'analysis.json')):
        raise ValueError('Separate exact inference replay required before final report')
    fits = [source.read(source.checked(r)) for r in source.read(PUBLIC / 'training_freeze.json')['fits']]
    groups = [source.read(source.checked(r)) for r in analysis['group_receipts']]
    if len(groups) != 108 or sum(len(g['rows']) for g in groups) != 276:
        raise ValueError('All orientations and sources required')
    result = outcome(analysis, decision)
    phases = {p: source.read(PUBLIC / (p+'_execution.json')) for p in ('pilot', 'train', 'evaluate', 'verify')}
    diagnostics = risk_diagnostics(groups)
    source.once(PUBLIC / 'risk_diagnostics.json', dict(values=diagnostics, counts_overlap_not_additive=True))
    direct = ['|Public minus point control|Known effect pp [95% CI]|Direct lower pp [95% CI]|', '|---|---:|---:|']
    for name in run.CONTRASTS:
        m = analysis['direct_contrasts'][name]
        direct.append(f"|{name}|{interval(m['known_pp'])}|{interval(m['completion_lower_pp'])}|")
    primary = ['|Veto vs own count/amplitude-matched control|Nonempty known pp [95% CI]|Missing /276|Full-strategy known pp, secondary|Full-strategy lower pp, secondary|', '|---|---:|---:|---:|---:|']
    for name in ('point', 'public'):
        m, f = analysis['nonempty_matched'][name], analysis['secondary_full_strategy'][name]
        primary.append(f"|{name}|{interval(m['known_pp'])}|{analysis['unavailable_primary_views'][name]}|{interval(f['known_pp'])}|{interval(f['completion_lower_pp'])}|")
    risk = ['|Policy|Supported /696|Known all-risk violations|Known easy-risk violations|Undefined selected easy risk|Switch % [95% CI]|', '|---|---:|---:|---:|---:|---:|']
    for name in run.POLICIES:
        r = analysis['safety'][name]
        risk.append(f"|{name}|{r['supported']}|{r['known_all_violations']}|{r['known_easy_violations']}|{r['undefined_selected_easy']}|{interval(analysis['switch_percent'][name])}|")
    text_once(PUBLIC / 'readout.md', '# Moving Public Regression-Deferral Control\n\n'
        '**fresh_run:**54 real Torch routing heads,108000primary updates, complete nested-C evaluation and separate replay. '
        '**cached_verified:** fixed forecasters, parent point heads, own-part preprocessing, original splits and data admission. '
        '**not_run:** new independent confirmation, new trajectory fitting and deployment.\n\n'+result+'\n\n'
        '## Public Method And What Changed\n\n'
        '[Mao, Mohri and Zhong (ICML2024)](https://proceedings.mlr.press/v235/mao24d.html), '
        'Section4 Equation3, supplies the two-stage logistic cost-sensitive objective. This is a '
        'task adaptation on our fixed seven-action bank, not their original experiment or a new M3W invention. '
        'The model directly ranks expected decision costs through a surrogate; it does not fit a one-hot '
        'oracle label or report softmax as calibrated confidence. Costs use available-label ADE divided '
        'by frozen causal scale. Unknown outcomes are excluded, not labeled zero.\n\n'
        'The original point learner used conservative envelope targets for wholly unknown '
        'futures. The new public learner excludes those outcome labels. The causal row roster '
        'and sampler match, but the effective supervised loss support differs.\n\n'
        'All predictors, source roles, 380 active local features, query context, training budget, '
        'preprocessing and sampling are unchanged. The public logit output has27969parameters '
        'versus28229in the point head. Different head/objective/decision semantics mean this is not '
        'an isolated loss-only causal comparison. Unbounded costs, partial labels and causal legal '
        'action masks do not automatically inherit the paper theorem or a safety guarantee.\n\n'
        '## Direct Development Contrasts\n\n'+'\n'.join(direct)+'\n\n'
        'Positive means the public policy has lower available-label error than the point policy. '
        'The guarded comparison uses the exact same source-excluded frozen point critic. Full '
        'policies can have different coverage, so this is not a matched-ranking test.\n\n'
        '## Same-Count Controls\n\n'+'\n'.join(primary)+'\n\n'
        'Required empty matched views remain NA. The original24group primary is still not '
        'estimable and not passed. Full-strategy secondary results do not replace it. Intervals '
        'use3000 paired source resamples after averaging contexts, seeds and orientations within '
        '23exposed source units. The276source views and696recording views are repeated, not independent.\n\n'
        '## Original Full-Policy Risk\n\n'+'\n'.join(risk)+'\n\n'
        'The2%positive-harm and easy-net requirements and reference-support conditions are '
        'unchanged. No identity certificate is introduced, even for the physical floor. Counts '
        'overlap and are not summed. Undefined risks are not zero. Every policy retains all required views.\n\n'
        '## Runtime And Evidence\n\n'
        +'\n'.join(f"- {p}: PID{r['pid']}, {r['seconds']:.2f}s, peak RSS{r['peak_RSS_bytes']/1e9:.3f}GB." for p, r in phases.items())
        +'\n\nThree real pilot resumed/direct fits and the first full2000-update refit agree exactly. '
        'All54fit normalizers, row IDs, sampler states and draw hashes equal the corresponding '
        'frozen point controls. All new heads froze before readout. All108policy blocks and '
        'prediction hashes match a separate replay. Native arm64 CPU4/interop1/workers0, '
        'no resource probing. Optimizer and query sampler states are checkpointed every100updates.\n\n'
        'Run `.venv-pytorch/bin/python scripts/run_m3w_moving_public_deferral.py PHASE` '
        'for register/pilot/train/evaluate/verify; interrupted unchanged phases accept `--resume`. '
        'Do not overwrite completed phases. Local weights and arrays are not committed.\n\n'
        'Observation8/prediction12 annotation order in image pixels. This is not verified seconds, '
        'metric, raw-frame t50, physical-safety, true3D or foundation evidence. Stage5C and SMC remain off.\n')
    text_once(PUBLIC / 'gates.md', '# Original Rules, No Exceptions\n\n'+result+'\n\n'
        f"Development criteria pass: **{decision['development_criteria_pass']}**.\n\n"
        'The old primary and negative experiments remain unchanged. Engineering completion does '
        'not establish scientific success. No deployment, independent confirmation or submission readiness.\n')
    text_once(PUBLIC / 'model_data_card.md', '# Model And Data Card\n\n'
        '## Intended Use\n\nA task-adapted public routing comparison in exposed-source research. '
        'Not a deployed trajectory generator, physical controller or certified safe policy. '
        'No existing deployable model is replaced.\n\n'
        '## Model\n\n54independently fitted width64 routing heads,27969parameters each, '
        'three producer seeds and original nested source partitions. The seven candidate '
        'trajectories are frozen. Local causal features, action code and same-query other-agent '
        'context produce one logit per action. The48shared-latent fields remain zero. '
        'The public loss comes from Mao et al. (ICML2024), not a new architecture claim. '
        'A separately fitted point critic supplies the unchanged guarded-policy risk test.\n\n'
        '## Data And Roles\n\nMovingDroneCrowd++ SOURCE only, observation8/prediction12 in native '
        'annotation order and image pixels. Original admission and source grouping are inherited '
        'with hash checks. Causal rows and sampling are matched to the parent point control. '
        'Each model and all learned preprocessing exclude its validation sources. This is '
        'exposed development evidence, not independent final test evidence. No reserved Moving, '
        'DUT or DroneCrowd confirmation data is opened. Future labels only supervise fitting '
        'and evaluation; eventual mask availability is not an inference input. Wholly unknown '
        'outcomes are excluded from this loss, unlike the parent bound-target loss.\n\n'
        '## Safety And Limits\n\n'+result+'\n\n'
        'Original2%positive harm/easy-net and support requirements are preserved. The original '
        '24group primary remains not estimable, not passed. Missing reference support cannot be '
        'converted into measured safety by an improved average loss. No identity fallback '
        'certification is used. Softmax/logit scores are not calibrated safety probabilities. '
        'No true3D, foundation, metric, calibrated-time or human-gold claim is supported. '
        'Stage5C and SMC stay off.\n\n'
        '## Reproduction\n\nConfiguration, registration, full training receipts and inference '
        'hashes are public; raw data, private arrays and checkpoints remain local. Resume '
        'requires the unchanged definition and checkpoint. Separate inference replay '
        'verifies108policy blocks and all prediction hashes. Use '
        '`.venv-pytorch/bin/python scripts/replay_m3w_moving_deferral.py` for a read-only '
        'repeat after original verification. Native arm64 CPU4/interop1/workers0; no NumPy '
        'training fallback or MPS-performance claim.\n')
    source.once(PUBLIC / 'runtime_summary.json', dict(phases=phases, fresh_heads=54, primary_updates=108000,
        pilot_updates=600, first_full_refit_updates=2000, saved_primary_fitter_seconds=sum(f['seconds'] for f in fits),
        source=source.artifact(PUBLIC / 'analysis.json')))
    per_source = ['|Source|Direct public-point known pp|Guarded public-point known pp|Public matched known pp|', '|---|---:|---:|---:|']
    rows = [r for g in groups for r in g['rows']]
    for site in sorted({r['site'] for r in rows}):
        cells = []
        for getter in (lambda r:r['direct_contrasts']['public_minus_point']['known_pp'],
                lambda r:r['direct_contrasts']['public_veto_minus_point_veto']['known_pp'],
                lambda r:r['nonempty_matched']['public']['contrast']['known_pp']):
            values = [getter(r) for r in rows if r['site'] == site]
            cells.append('NA' if any(v is None for v in values) else f'{source.np.mean(values):.6f}')
        per_source.append('|'+site+'|'+'|'.join(cells)+'|')
    text_once(PUBLIC / 'per_source.md', '# All Exposed Source Units\n\n'+'\n'.join(per_source)
        +'\n\nNo independent test claim. NA retains missing required support, not a zero or excluded scene.\n')
    figures(fits, analysis)
    print(source.json.dumps(dict(result=result, decision=decision, report=str(PUBLIC / 'readout.md')), indent=2))


if __name__ == '__main__':
    main()
