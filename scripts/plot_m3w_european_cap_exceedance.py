"""All six assignment intervals, both arms and both input families."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_m3w_european_cap_exceedance import PUBLIC, PRIVATE


def main():
    doc = json.loads((PUBLIC/'aggregate_metrics.json').read_text())
    rows = doc['contrasts']; pairs = ['full', 'motion_only']
    keys = [('BCE_gain_prior', 'Log-loss improvement vs fitting prior', 1),
            ('Brier_gain_prior', 'Brier improvement vs fitting prior', 1),
            ('AP_gain_envelope', 'AP gain vs causal envelope (percentage points)', 100),
            ('capture_gain_envelope', 'Top10% overshoot capture gain (percentage points)', 100)]
    assignments = sorted({(r['producer'], r['controller']) for r in rows})
    with plt.rc_context({'font.size': 9, 'svg.hashsalt': 'm3w-cap-event-v1'}):
        fig, axes = plt.subplots(4, 2, figsize=(12, 14))
        for ri, (key, label, scale) in enumerate(keys):
            for ci, pair in enumerate(pairs):
                ax = axes[ri, ci]; ax.axvline(0, color='#808080', lw=.8)
                for arm, offset, color, marker in [('linear', -.12, '#c25a31', 'o'), ('mlp', .12, '#19736a', 's')]:
                    for j, (producer, controller) in enumerate(assignments):
                        r = next(r for r in rows if (r['producer'], r['controller'], r['pair'], r['arm'], r['contrast']) ==
                                 (producer, controller, pair, arm, key))
                        if r['status'] == 'measured':
                            x = r['point']*scale
                            ax.errorbar(x, j+offset, xerr=[[max(0, (r['point']-r['low'])*scale)],
                                                          [max(0, (r['high']-r['point'])*scale)]],
                                        fmt=marker, color=color, capsize=2, label=arm if j == 0 else None)
                        else:
                            ax.text(.97, j+offset, arm+': missing support', ha='right', va='center', fontsize=7,
                                    transform=ax.get_yaxis_transform(), color=color)
                ax.set_yticks(range(6), [f'P{p} / C{c}' for p, c in assignments])
                ax.set_title(pair.replace('_', ' ')); ax.set_xlabel(label); ax.grid(axis='x', alpha=.15)
                ax.spines[['top', 'right']].set_visible(False)
                if ri == 0: ax.legend(loc='best')
        fig.suptitle('Causal cap-event learnability: all registered contrasts', fontsize=15, y=.995)
        fig.text(.5, .01, 'Positive favors the probe. Three seeds averaged within each of four source-development localities.\n'
                 '3000 paired locality resamples; assignments overlap; exploratory intervals, not deployment guarantees.',
                 ha='center', va='bottom', fontsize=9)
        fig.tight_layout(rect=(0, .047, 1, .98))
        fig.savefig(PUBLIC/'cap_event_contrasts.svg', metadata={'Date': None})
        fig.savefig(PRIVATE/'cap_event_contrasts.png', dpi=140)
        plt.close(fig)


if __name__ == '__main__': main()
