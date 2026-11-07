"""Render the frozen capacity and causal policy results in separate panels."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'


def load(study, name, receipt):
    base = HOME/study
    raw = (base/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads((base/receipt).read_text())['artifact']['sha256']:
        raise ValueError('Plot requires the scored aggregate')
    return json.loads(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/stage_cvpr2027_experiments/m3w_decision_bottlenecks.pdf')
    a = p.parse_args()
    retrieval = load('native_recording_witness_retrieval_v1', 'retention.json', 'retention_execution.json')
    portfolio = load('native_joint_path_portfolio_v1', 'analysis.json', 'replay_execution.json')
    order = ('retrieved_bank', 'decoded_queries', 'admitted_queries')
    caps = [retrieval['stages'][k] for k in order]
    risks = [portfolio['grouped']['risk'][k] for k in ('single_gated', 'matched_joint', 'screen_first_joint')]
    if not all(v['required_views'] == 102 for v in risks) or any(v['unresolved'] for v in caps):
        raise ValueError('Require the complete fixed development views')
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.2), layout='constrained')
    keys = [
        (left, caps, 'A  Hindsight capacity', ('Retrieved bank', 'One decoded option', 'Admitted options'),
         ('feasible', 'positive'), ('Risk-feasible', 'Positive lower gain vs. OLS4'), ('#AFC9D7', '#137471')),
        (right, risks, 'B  Actual policy outcomes',
         tuple(f'{name} ({v["switches"]:,})' for name, v in zip(('Original', 'Matched joint', 'Screen first'), risks)),
         ('all_risk_over_2_views', 'supported_views'), ('All-risk violation', 'All checks pass'), ('#C66140', '#137471')),
    ]
    for ax, values, title, names, fields, legend, color in keys:
        for j, v in enumerate(values):
            for i, (f, c, label) in enumerate(zip(fields, color, legend)):
                y = j+(.17 if i else -.17)
                n = v[f]
                ax.barh(y, n, height=.29, color=c, label=label if j == 0 else None)
                ax.text(n+1.5, y, str(n), va='center', fontsize=8, color='#132E34')
        ax.set_title(title, loc='left', weight='bold', fontsize=11)
        ax.set_yticks(range(3), names, fontsize=8)
        ax.set_xlim(0, 102)
        ax.set_xticks((0, 25, 50, 75, 102))
        ax.set_xlabel('Recording-seed views', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='both', length=0)
        ax.grid(axis='x', color='#E8ECEF', linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_ylim(2.55, -.55)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#BAC4C9')
        ax.legend(loc='upper left', bbox_to_anchor=(0, -.22), fontsize=7.3, frameon=False, handlelength=1.3)
    fig.suptitle('More available choices do not guarantee more passing recordings', weight='bold', fontsize=12)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
