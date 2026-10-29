"""Evidence-bound report for the one-factor bounded-range continuation."""
import argparse
import json

from scripts import run_m3w_full_range_harm as run


def interval(v):
    if v['CI95'] is None:
        return 'undefined'
    return f"{v['mean']:.6f} [{v['CI95'][0]:.6f}, {v['CI95'][1]:.6f}]"


def render(summary, training, readout):
    lines = ['# Full-Range Bounded Harm: Complete Matched Experiment', '',
        '## Material Passport', '',
        f"Fresh18 fits,36,000 updates in{training['seconds']:.2f}s;144 readout views in{readout['seconds']:.2f}s.",
        'Six old controls are cached_verified models with fresh exactly matching',
        'inference. No old trajectory predictor or new data role was trained/opened.',
        'The18 new finals were all frozen before this readout. No checkpoint or',
        'threshold selection. Native arm64 CPU4/workers0, read-only CREATE TRAIN',
        'streaming; no new scheduler job.', '',
        '## Controlled Change', '',
        'The harm branch uses additive probability shifts across its full causal',
        'range instead of bounded log-odds shifts. Benefit/reference outputs remain',
        'exactly fixed. Initial weights, source/query sampling, draw hashes, optimizer',
        'settings, budget and12,258 trainable/25,407 total parameters match the',
        'isolated-odds control. Range and gradient geometry both change; they are',
        'not separately identified. This is a cost head, not a dynamics forecaster.', '',
        '## Complete Results', '']
    for scope in ('fit', 'held'):
        data = summary[scope]
        lines += [f'### {scope.title()}', '',
            '| Policy | Signed MSE | Selected occurrences | Known easy violations /72 | Upper easy violations /72 | Empty | Undefined easy | Full support /72 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in run.evaluation.ARMS:
            p = data['safety'][a]
            lines.append('| '+a+' | '+f"{data['mean_view_MSE'][a]:.6f}"+' | '+' | '.join(str(p[k]) for k in
                ('selected', 'known_easy_violations', 'upper_easy_violations', 'empty', 'undefined_easy', 'full_support'))+' |')
        lines += ['', '| Full-range minus control | Full lower utility pp [95% CI] | Matched-count lower utility pp [CI] | Signed MSE difference [CI] |',
            '|---|---:|---:|---:|']
        for a in run.evaluation.CONTROLS:
            lines.append('| '+a+' | '+' | '.join(interval(data['contrasts'][a+'_'+kind]) for kind in ('full', 'matched', 'MSE'))+' |')
        lines.append('')
    lines += ['## Registered Gate', '',
        f"Advance to later development design: **{summary['advance']}**.",
        'Failures: '+(', '.join(summary['failures']) or 'none')+'.', '',
        'Positive lower95% intervals required against every control at both full',
        'and matched counts, plus full-policy risk support in all72 held views.',
        'Empty/undefined or incomplete-label policies do not earn a safety pass.',
        'A lower loss/MSE or one positive source cannot substitute for this rule.', '',
        'Utility is pair-specific completion-lower mass divided by full-known',
        'reference mass, NOT ADE/FDE. Bounds are not transitive. Original TRAIN',
        'fit/held recordings are already exposed. Bootstrap3,000 averages repeated',
        'contexts/seeds within12 localities; nominal, dependent and not multiplicity',
        'adjusted or independent confirmation. Three cost seeds are not three',
        'new forecasting producers. No deployable world-model success is inferred.', '',
        '## Reproduction', '', '```bash',
        '.venv-pytorch/bin/python -m scripts.run_m3w_full_range_harm check',
        '.venv-pytorch/bin/python -m scripts.report_m3w_full_range_harm --check', '```', '',
        'Complete outputs are immutable; only unfinished phases permit --resume.',
        'Private checkpoints and raw histories are excluded from Git. Public outputs',
        'contain aggregate metrics, provenance and reproducible code.', '',
        'Obs8/pred12 raw stride12, detector-track pixel/local coordinates. No metric,',
        'seconds, physical safety, human gold, true3D or foundation claim.',
        'Stage5C/SMC disabled. Historical Stage37 is exploratory, not recertified.', '']
    return '\n'.join(lines)


def main(check=False, plot=False):
    run.evaluate(check=True)
    reg, _, _ = run.admission()
    _, documents = run.models(reg)
    _, controls = run.prior.parent.models(reg)
    frozen = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if len(frozen['fits']) != 18 or len({v['path'] for v in frozen['fits']}) != 18:
        raise ValueError('Exactly18 unique frozen receipts required')
    initial_replays = 0
    for (context, seed), doc in documents.items():
        state = run.api.core.read_checkpoint(run.ROOT/doc['checkpoint']['path'])
        old = run.api.core.read_checkpoint(run.ROOT/controls[context, seed, 'isolated_risk']['checkpoint']['path'])
        if state['experiment_sha256'] != run.run.base.sha(run.REG) or state['step'] != 2000:
            raise ValueError('Final model binding changed')
        for key in ('loss', 'primary_loss', 'regret', 'mean_soft_action'):
            run.api.core.exact(state['trace'][0][key], old['trace'][0][key])
        initial_replays += 1
    if initial_replays != 18:
        raise ValueError('All real initial objective values must replay')
    out = run.PUBLIC/'readout'
    summary = json.loads((out/'summary.json').read_text())
    training = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    readout = json.loads((out/'complete.json').read_text())
    if readout['registration_sha256'] != run.run.base.sha(run.REG):
        raise ValueError('Readout completion binding changed')
    result = render(summary, training, readout)
    if check:
        if (run.PUBLIC/'conclusions.md').read_text() != result:
            raise ValueError('Report differs from verified evidence')
    else:
        (run.PUBLIC/'conclusions.md').write_text(result)
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        arms = run.evaluation.ARMS
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(arms))
        axes[0].bar(x-.18, [summary['fit']['safety'][a]['known_easy_violations'] for a in arms], .36, label='fit')
        axes[0].bar(x+.18, [summary['held']['safety'][a]['known_easy_violations'] for a in arms], .36, label='held')
        axes[0].set_xticks(x, [a.replace('_', '\n') for a in arms], fontsize=8)
        axes[0].set_ylabel('Known easy-risk violating views /72')
        axes[0].set_ylim(0, 76)
        axes[0].legend()
        arms = run.evaluation.CONTROLS
        yy = np.arange(len(arms))
        values = [summary['held']['contrasts'][a+'_matched'] for a in arms]
        means = np.array([v['mean'] for v in values])
        ci = np.array([v['CI95'] for v in values])
        axes[1].hlines(yy, ci[:, 0], ci[:, 1], linewidth=1.5)
        axes[1].scatter(means, yy, s=24)
        axes[1].axvline(0, color='black', linewidth=.8)
        axes[1].set_yticks(yy, arms, fontsize=8)
        axes[1].set_xlabel('Full-range minus control: matched lower utility pp')
        axes[1].set_title('Nominal locality bootstrap95% CI')
        fig.suptitle('Bounded-range repair | exposed TRAIN-internal, not deployment evidence')
        fig.tight_layout()
        path = run.PRIVATE/'figures/comparison.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        run.once(run.PUBLIC/'figure_receipt.json', dict(path=str(path.relative_to(run.ROOT)),
            sha256=run.run.base.sha(path), source_summary_sha256=run.run.base.sha(out/'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args.check, args.plot)
