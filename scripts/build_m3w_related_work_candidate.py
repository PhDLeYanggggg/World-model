"""Build a separate draft with direct attribution of two close prior works."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_metric_scope_candidate as scope


ORIGINAL = scope.DESTINATION
DESTINATION = ORIGINAL.parent/'evidence_manuscript_v169_related_work'
RECEIPT = scope.main.PUBLIC/'evidence_manuscript_v169_related_work/receipt.json'
BIB_END = r'\end{thebibliography}'
OLD = r'''Constant velocity is a strong comparator in pedestrian forecasting~\cite{schoeller}.
Choosing among forecasters is also an established problem~\cite{mao}.
What is less clear is whether a useful joint replacement can be found
without seeing its outcome. Work on raw-video forecasting addresses
tracking errors through re-tracking, denoising, prediction from
detections, or propagation of tracking uncertainty~\cite{yu,natra,realtraj,martins}.
We study a different decision: when an available forecast should
replace a motion reference if future labels are incomplete.'''
NEW = r'''Constant velocity is a strong comparator~\cite{schoeller}.
Regression deferral routes inputs among experts~\cite{mao}; loss-conditioned
execution tests proposals against persistence~\cite{xuloss}. Wang~\cite{wangpair}
tests forecast-mode pairing with a fixed planner. Raw-video forecasting
addresses tracking errors through re-tracking, denoising, detection-based
prediction and uncertainty propagation~\cite{yu,natra,realtraj,martins}.
We ask which agents' forecasts to replace at a fixed budget when
future labels are incomplete.'''
ADDED = r'''\bibitem{xuloss}
J. Xu, Z. Chen, B. Zhang, Y. Qi, and J. Zhang.
When should a world model move? Loss-conditioned state execution.
\emph{arXiv:2609.15801}, 2026.

\bibitem{wangpair}
J. Wang. From learned-mode AV--traffic pairing to planner decisions:
A marginal-preserving study on Argoverse 2. \emph{arXiv:2609.14997}, 2026.

'''


def revise(text):
    if text.count(OLD) != 1 or text.count(BIB_END) != 1 or any(
            r'\bibitem{'+k+'}' in text for k in ('xuloss', 'wangpair')):
        raise ValueError('Expected the unchanged metric-scope candidate')
    changed = text.replace(OLD, NEW).replace(BIB_END, ADDED+BIB_END)
    if changed.replace(NEW, OLD).replace(ADDED+BIB_END, BIB_END) != text:
        raise ValueError('Unrelated source change')
    return changed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', required=True, type=Path)
    p.add_argument('--output', type=Path, default=DESTINATION)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    a = p.parse_args()
    v168 = scope.supplement.read_receipt('evidence_manuscript_v168_metric_scope')
    scope.build(a.output, a.compiler.resolve(), a.receipt, original=ORIGINAL, expected=v168['artifacts'],
                revise_source=revise, changed_paragraphs=1, added_citations=('xuloss', 'wangpair'),
                markers=('2609.15801', '2609.14997'))
