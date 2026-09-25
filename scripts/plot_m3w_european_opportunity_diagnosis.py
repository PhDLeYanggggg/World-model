"""Paired frozen-candidate opportunity and ordered gate attrition."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1'


def main():
    raw = (PUBLIC/'analysis.json').read_bytes()
    r = json.loads(raw)
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if not audit['all_passed'] or audit['analysis_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Verified accounting required')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout='constrained')
    colors = ['#32927b', '#747db2', '#c87935', '#9b6173']
    fields = ['captured_gain', 'missed_nonpositive_utility', 'missed_risk_veto']
    for ax, candidate in zip(axes, ('neural', 'damping097')):
        bottom = np.zeros(3)
        rows = [r['seeds'][s]['policies'][f'{s}_{candidate}_easy_neural_underharm4_no_guard']['ledgers']['all']
                for s in ('17', '29', '43')]
        for color, field, label in zip(colors, fields, ('Captured gross benefit', 'Rejected: utility', 'Rejected: risk')):
            values = np.array([x['summary'][field]['equal_locality'] for x in rows])
            ax.bar(range(3), values, bottom=bottom, color=color, label=label, width=.58)
            bottom += values
        harm = np.array([x['summary']['switched_harm']['equal_locality'] for x in rows])
        net = np.array([x['summary']['net_gain']['equal_locality'] for x in rows])
        ax.bar(range(3), -harm, color=colors[3], width=.58, label='Switched harm (subtract)')
        ax.plot(range(3), net, 'ko', markersize=5, label='Actual net gain')
        for i, x in enumerate(rows):
            oracle = x['summary']['oracle_gain']['equal_locality']
            assert np.isclose(bottom[i], oracle)
            ax.text(i, oracle+.28, f'{100*x["gain_capture_fraction"]:.1f}% captured', ha='center', fontsize=10)
        ax.set_ylim(-1, 20)
        ax.set_xticks(range(3), ['Seed 17', 'Seed 29', 'Seed 43'])
        ax.set_ylabel('Percentage points of locality CV error')
        ax.set_title('Neural trajectory candidate' if candidate=='neural' else 'Fixed damping 0.97 candidate')
        ax.axhline(0, color='#444444', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
    fig.suptitle('Hindsight opportunity is not captured predictive gain\nEasy-event neural risk, no support guard; frozen source-development diagnostic', fontsize=13)
    fig.savefig(PUBLIC/'opportunity_attribution.png', dpi=160)
    fig.savefig(PUBLIC/'opportunity_attribution.svg')
    svg = PUBLIC/'opportunity_attribution.svg'
    svg.write_text(re.sub(r'[ \t]+\n', '\n', svg.read_text()))
    print('Figure rendered; visual inspection required.')


if __name__ == '__main__': main()
