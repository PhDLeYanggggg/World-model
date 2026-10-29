"""Render already frozen contrasts; no new analysis or selection."""
import json

from scripts import run_m3w_query_context_risk as run


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = json.loads((run.PUBLIC/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.45, 1]})
    controls = run.evaluation.CONTROLS
    for offset, mode, color in ((-.13, 'full', '#a3322d'), (.13, 'matched', '#236d92')):
        vals = [s['contrasts'][a+'_'+mode] for a in controls]
        means = run.api.np.array([v['mean'] for v in vals]); ci = run.api.np.array([v['CI95'] for v in vals])
        axes[0].errorbar(means, run.api.np.arange(len(vals))+offset,
            xerr=run.api.np.vstack((means-ci[:, 0], ci[:, 1]-means)), fmt='o', color=color, capsize=3, label=mode)
    axes[0].axvline(0, color='#555555', linewidth=.8)
    axes[0].set_yticks(range(len(controls)), controls)
    axes[0].invert_yaxis(); axes[0].legend(frameon=False)
    axes[0].set_xlabel('Peer minus comparator: completion-lower utility (pp)')
    axes[0].set_title('Nominal 95% locality-bootstrap intervals')
    arms = ('rational', 'self', 'peer'); positions = run.api.np.arange(len(arms))
    for off, key, label, color in ((-.23, 'known_easy_violations', 'Known easy-risk violations', '#a3322d'),
                                  (0., 'undefined_easy', 'Undefined easy risk', '#c58a20'),
                                  (.23, 'full_support', 'Complete support', '#327e5b')):
        vals = [s['safety'][a][key] for a in arms]
        axes[1].bar(positions+off, vals, .22, label=label, color=color)
        for x, v in zip(positions+off, vals): axes[1].text(x, v+1, str(v), ha='center', fontsize=9)
    axes[1].set_xticks(positions, arms); axes[1].set_ylim(0, 80)
    axes[1].set_ylabel('Dependent views, out of 72'); axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    axes[1].set_title('Observed violations and support are distinct')
    fig.suptitle('Contemporaneous peer context: original TRAIN only', fontsize=13)
    fig.text(.5, .015, '12 exposed localities; no independent confirmation, deployment, metric or physical-safety claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .94])
    path = run.PRIVATE/'query_context_results.png'; fig.savefig(path, dpi=160); plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json', dict(path=str(path.relative_to(run.ROOT)), sha256=run.sha(path),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'), numerical_data_committed=False))
    print(path)


if __name__ == '__main__': main()
