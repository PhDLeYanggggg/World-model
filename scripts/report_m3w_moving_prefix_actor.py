"""Render scalar evidence from the fixed matched producer-loss experiment."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_prefix_actor as run


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}'


def main():
    a = run.read(run.PUBLIC/'analysis.json'); cfg = run.read(run.CONFIG)
    primary = a['primary']; fits = [run.read(p) for p in sorted((run.PUBLIC/'fits').glob('*.json'))]
    if len(fits) != 6 or not all(f['result']['complete'] for f in fits):
        raise ValueError('Only complete fixed-budget experiment can be reported')
    control = run.read(run.PUBLIC/'control_refit.json'); pilot = run.read(run.PUBLIC/'pilot.json')
    lines = ['# Prefix-Balanced Trajectory Producer: Actual Matched Fitting', '',
        '## Result', '',
        f'Component progression screen: **{"PASS" if primary["actor_component_progression_screen_pass"] else "FAIL"}**.',
        f'Equal-source paired available12ADE gain difference: **{fmt(primary["equal_source_paired_known_pp"])}pp**.',
        f'Conservative paired completion-lower: **{fmt(primary["equal_source_paired_completion_lower_pp"])}pp**,',
        f'exploratory source bootstrap95% interval **[{fmt(primary["lower_exploratory_bootstrap_CI"][0])}, '
        f'{fmt(primary["lower_exploratory_bootstrap_CI"][1])}]pp**.',
        f'{primary["positive_source_known_contrasts"]}/14sources improve on the whole available12-step outcome.',
        f'Producer-fold known mean differences: {primary["fold_known_means_pp"]}.', '',
        'This is a fitting-donor component contrast, not an independent test or',
        'a deployment result. The source bootstrap is exploratory because training',
        'roles overlap across folds and sources have prior exposure. One producer',
        'seed does not meet the main three-seed paper requirement. Prefix-only',
        'improvement cannot replace a failed whole12 contrast.', '',
        '## What Was Actually Trained', '',
        'Fresh_run:six3,027,268-parameter EqMotion K1 trajectory producers,4000updates',
        'each,plus one full4000update old-loss control refit. The100+100vs200pilot',
        'exactly reproduces model,optimizer,sampler,Torch RNG and trace. One pilot',
        'prefix is resumed into its corresponding full treatment,not counted as',
        'another completed model. All six controls/125540rows are cached_verified.',
        'No new own-B,C/H/reserved data or new controller training. No rollout',
        'ramp,ordinary residual head,threshold sweep or changed safety certificate.',
        f'Control scientific state and common trace exact: {control["all_model_optimizer_sampler_RNG_draw_state_exact"]};',
        f'old pilot-only extra log steps: {control["original_pilot_only_extra_log_steps"]}.',
        f'Recovery exact: {pilot["resume100plus100_equals200"]}.', '',
        'The sole intervention is a fixed loss:half available12ADE,half mean',
        'supported1/3/6prefix ADEs. Rows with only late labels keep full loss;',
        'wholly unknown rows stay in the sampler with zero gradient,not zero-error',
        'labels. Inputs remain original past geometry476. Deep supervision is a',
        'conventional backbone repair,not a novel paper-method claim.', '',
        '## Every Source And Prefix', '',
        '| Source | Prefix | Rows | Known | Partial | Unknown | Control gain % | New gain % | Paired pp | Paired lower pp | Control easy +harm % | New easy +harm % | New signed easy loss % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source in a['sources']:
        for r in source['summary']:
            old, new = r['control'], r['treatment']
            lines.append(f'| {source["source"]} | {r["prefix"]} | {r["rows"]} | {r["known_rows"]} | '
                f'{r["partial_rows"]} | {r["wholly_unknown_rows"]} | {fmt(old["known_gain_percent"])} | '
                f'{fmt(new["known_gain_percent"])} | {fmt(r["paired_known_pp"])} | {fmt(r["paired_completion_lower_pp"])} | '
                f'{fmt(old["known_easy_positive_harm_percent"])} | {fmt(new["known_easy_positive_harm_percent"])} | '
                f'{fmt(new["known_easy_signed_degradation_percent"])} |')
    lines += ['', '## Every Recording: Whole Available12', '',
        '| Recording | Rows | Unknown | Old gain % | New gain % | Paired pp | New positive harm % | New easy +harm % | New signed easy loss % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rec in a['recordings']:
        r = rec['summary'][-1]; old, new = r['control'], r['treatment']
        lines.append(f'| {rec["recording"]} | {rec["rows"]} | {r["wholly_unknown_rows"]} | '
            f'{fmt(old["known_gain_percent"])} | {fmt(new["known_gain_percent"])} | {fmt(r["paired_known_pp"])} | '
            f'{fmt(new["known_positive_harm_percent"])} | {fmt(new["known_easy_positive_harm_percent"])} | '
            f'{fmt(new["known_easy_signed_degradation_percent"])} |')
    lines += ['', 'Known-label raw-actor harm columns are diagnostics,not the full policy',
        'certificate. Original recording2% all/easy positive harm,signed easy-net',
        'and support requirements remain unchanged. NA is not zero or pass. No',
        'actor is directly deployed; new risk heads would need new causal labels',
        'and the complete original policy checks before any later deployment.', '',
        '## Compute And Reproduction', '',
        f'Six treatment cumulative fitting seconds:{sum(f["result"]["seconds"] for f in fits):.6f}.',
        f'Full control fitting seconds:{control["result"]["seconds"]:.6f}.',
        'Native arm64 CPU4/inter-op1/workers0. Atomic200update checkpoints and',
        '50update heartbeats; checkpoints,arrays and private caches stay local.', '',
        '```sh',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py pilot',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py control --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py train --resume',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py forecast',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py evaluate',
        '.venv-pytorch/bin/python scripts/run_m3w_moving_prefix_actor.py replay',
        '```', '',
        'Completed pilot is immutable;verify its receipt rather than re-running.',
        'Replay regenerates predictions and scalars without optimizer updates.',
        'Current task is native observation8/prediction12 annotation stride1 image',
        'coordinates. No meters,seconds,true3D,foundation,human-gold,physical-safety',
        'or submission-ready claim. Stage5C execution and SMC remain off.']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    traces = []
    for f in fits:
        key = f['job']['key']; old = run.read(run.source.PUBLIC/'fits'/f'{key}.json')
        for name, receipt in (('available12', old), ('prefix_balanced', f)):
            for row in receipt['result']['losses']:
                traces.append(dict(job=key, arm=name, **row))
    with (run.PUBLIC/'loss_trace.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(traces[0])); writer.writeheader(); writer.writerows(traces)
    print('Complete paired actor evidence reported; no policy deployment.')


if __name__ == '__main__':
    main()
