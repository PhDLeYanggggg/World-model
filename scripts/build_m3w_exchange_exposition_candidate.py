"""Build the v211 closed-form risk and matched exchange exposition."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_full_grid_exchange_candidate as prior
from scripts.build_m3w_full_grid_candidate import builder

OUT = prior.OUT.with_name('evidence_manuscript_v211_exchange_exposition')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
FIG = OUT.parent/'opendd_full_grid_exchange_v1/exchange_gap.pdf'
LIST = PUBLIC.parent/'opendd_listwise_exchange_v1/readout.json'

RISK_OLD = '''We also check risk on the full grid under free missing-position
completions. The worst all-edited ratio places missing targets
at CV. For the selected-easy ratio, sum positive harm minus
the risk limit times CV error over mandatory easy tracks and
take only positive excesses from optionally easy tracks. A
complete easy edit guarantees reference support. The supplement
derives these checks; they do not replace the release rule.'''

RISK_NEW = r'''\paragraph{Risk on the complete grid.}
For each edited track $i$, let $B_i$ be the sum of its observed
CV errors and $D_i$ the sum of its observed error changes plus the
forecast--CV distances at its missing steps. Define the excess
$X_i=[D_i]_+-rB_i$, with $r=0.02$. Let $E$ contain complete
edits with $0<B_i\leq12\tau_E$, and let $U$ contain incomplete
edits with $B_i\leq12\tau_E$. For every free completion, all
three risk ratios are defined and meet the limit exactly when
\begin{equation}
 |E|>0,\quad
 \sum_{i\,\mathrm{edited}} X_i\leq0,\quad
 \sum_{i\in E}X_i+\sum_{i\in U}[X_i]_+\leq0.
 \label{eq:full-grid-risk}
\end{equation}
Placing a missing target at CV jointly minimizes baseline error
and maximizes deterioration. An incomplete track can instead
leave the easy subset if a missing position is far from CV;
if $B_i=0$, the easy supremum uses vanishing positive CV error.
With a positive selected-easy denominator, the positive-harm
ratio also bounds signed harm on the whole easy roster.
This is a sensitivity test for the scored recording, not a
new release rule.'''

FIG_OLD = r'''\includegraphics[width=.89\textwidth]{capacity_discrimination.pdf}
\caption{OpenDD development. Left: label-aware plans that pass the
risk rule and beat both references, versus the past-only count
controller. Right: peer-head utility against motion and prefix
controls. Intervals resample locations after averaging seeds.}'''

FIG_NEW = r'''\includegraphics[width=\textwidth]{exchange_gap.pdf}
\caption{OpenDD development under the frozen one-exchange budget.
Left: original dual passes and those that also satisfy
Eq.~\ref{eq:full-grid-risk}. The uniform arm is an expectation;
the hindsight arms use the outcomes. Right: location-level paired
differences (gray points) and equal-location means with 95\% intervals
over 3,000 location resamples (diamonds and bars). Each of the 202
recordings is repeated across three seeds.}'''

RANK_OLD = r'''A donor-trained ranker gives 55/606, with no reliable gain over
same-view random exchange.
Learning does not close this gap: at the same caps,
gain-and-risk regression passes 42/606 views; independent plan
classification and pairwise ranking pass 54/606 and 59/606.
Risk-conditional utility training passes 58/606 versus 55/606 for
equal-update continuation without it (paired conservative gain
-0.00197 pp, 95\% location interval
[-0.01659, +0.01418]). None exceeds the
frozen scores on dual passes in these development fits.'''

RANK_NEW = r'''Gain-and-risk regression, plan classification, pairwise ranking
and risk-conditional training obtain 42, 54, 59 and 58 dual passes
at the same caps. None beats the frozen scores.'''


def inputs():
    _, parent_hash = prior.inputs()
    raw = LIST.read_bytes()
    d = json.loads(raw)
    if (d['new_external_outcome_access'] or d['deployment_changed'] or d['independent_confirmation']
            or not d['original_2pct'] or not d['no_held_tuning']
            or d['summary']['counts'] != {'views': 606, 'edits': 780, 'switched': 101}):
        raise ValueError('Unexpected OpenDD development roles or budget')
    sha = builder.main.digest
    return d, dict(**parent_hash, listwise_readout=sha(raw), exchange_figure=sha(FIG.read_bytes()))


def revise(text, data):
    anchor = 'Missing labels do not explain most of the gap.'
    intro = 'with a continuous displacement bound that does not depend on\nthe forecast bank.'
    fig_ref = 'Figure~\\ref{fig:capacity-discrimination} shows the corresponding\ntest on OpenDD. With the same total quota and original risk,'
    if any(text.count(v) != 1 for v in (RISK_OLD, FIG_OLD, RANK_OLD, anchor, fig_ref, intro)):
        raise ValueError('Expected the v210 text and figure anchors')
    text = text.replace(RISK_OLD, RISK_NEW, 1).replace(FIG_OLD, FIG_NEW, 1).replace(RANK_OLD, RANK_NEW, 1)
    text = text.replace(intro, intro+' A closed-form test checks whether\nmissing positions can overturn a recording-level risk pass.', 1)
    text = text.replace(fig_ref, 'On OpenDD, with the same total quota and original risk,', 1)
    row = data['summary']['arms']
    add = (f"\nFigure~\\ref{{fig:capacity-discrimination}} shows these contrasts.\n"
           f"On the same mask, equal-training pointwise and listwise controls\n"
           f"give {row['point_original']['full_grid']:.0f} and {row['list_original']['full_grid']:.0f} full-grid passes; "
           f"listwise with full-grid labels gives {row['list_full_grid']['full_grid']:.0f}.\n"
           f"None exceeds the incumbent's {row['incumbent']['full_grid']:.0f}.")
    return text.replace(anchor, anchor+add, 1), ('Aligned vs. same-mask uniform', '175/606', 'full-grid labels')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    args = p.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise, extra_assets={'exchange_gap.pdf': FIG.read_bytes()})
