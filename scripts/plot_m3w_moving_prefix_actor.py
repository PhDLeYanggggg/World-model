"""Local scalar-only figures; never expose raw frames, tracks or checkpoints."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_prefix_actor as run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    a = run.read(run.PUBLIC/'analysis.json'); home = run.PRIVATE/'figures'; home.mkdir(parents=True, exist_ok=True)
    x = run.np.arange(len(a['sources'])); fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for k, color in ((1, '#157A6E'), (3, '#5375AD'), (6, '#B77816'), (12, '#A43B58')):
        y = [next(r for r in s['summary'] if r['prefix'] == k)['paired_known_pp'] for s in a['sources']]
        ax[0].plot(x, y, '.-', label=f'Prefix{k}', color=color, linewidth=1, markersize=5)
    ax[0].axhline(0, color='black', linewidth=.7); ax[0].set_ylabel('Gain difference vs old actor (pp)')
    ax[0].set_title('Fitting donors: all sources, no held-test claim'); ax[0].legend(ncol=2, fontsize=8)
    for arm, color, shift in (('control', '#5375AD', -.12), ('treatment', '#A43B58', .12)):
        y = [s['summary'][-1][arm]['known_easy_positive_harm_percent'] for s in a['sources']]
        yy = [run.np.nan if v is None else v for v in y]
        ax[1].plot(x+shift, yy, 'o', label=arm, color=color, markersize=5)
    ax[1].axhline(2, color='#A43B58', linestyle='--', linewidth=1, label='Original2% reference')
    ax[1].set_yscale('symlog', linthresh=2); ax[1].set_ylabel('Known easy positive harm / easy CV mass (%)')
    ax[1].set_title('Raw actor diagnostic, not a policy certificate'); ax[1].legend(fontsize=8)
    labels = [s['source'].rsplit('_', 1)[-1] for s in a['sources']]
    for axis in ax:
        axis.set_xticks(x, labels, rotation=45); axis.set_xlabel('Source component'); axis.grid(axis='y', alpha=.2)
    fig.savefig(home/'source_prefix_and_easy_harm.png', dpi=160); plt.close(fig)

    with (run.PUBLIC/'loss_trace.csv').open() as f:
        trace = list(csv.DictReader(f))
    jobs = sorted({r['job'] for r in trace}); fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for axis, job in zip(axes.ravel(), jobs):
        for arm, color in (('available12', '#5375AD'), ('prefix_balanced', '#A43B58')):
            rows = [r for r in trace if r['job'] == job and r['arm'] == arm]
            axis.plot([int(r['step']) for r in rows], [float(r['mean_past_normalized_ADE']) for r in rows],
                linewidth=.8, alpha=.85, color=color, label=arm)
        axis.set_title(job, fontsize=9); axis.set_yscale('log'); axis.grid(alpha=.2)
        axis.set_xlabel('Optimizer updates'); axis.set_ylabel('Unweighted minibatch available ADE (image coords)')
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle('Matched minibatches: actual full-ADE diagnostic, not unlike objective-loss comparison')
    fig.savefig(home/'matched_training_available_ADE.png', dpi=150); plt.close(fig)
    print('Two local scalar-only figures written; no raw data or paper-success claim.')


if __name__ == '__main__':
    main()
