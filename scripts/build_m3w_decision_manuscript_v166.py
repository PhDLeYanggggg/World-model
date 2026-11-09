"""Revise the working paper's abstract and opening without changing results."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v165 as previous

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v166'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name
ABSTRACT = r"""Changing more forecasts can make a joint policy look better without
improving its choice of whom to help. We introduce a controlled comparison
for top-down multi-agent forecasting with incomplete future labels. It
holds the number and size of forecast replacements fixed and tests them
under a recording-level risk limit relative to constant velocity. A
second analysis uses future labels to ask whether any eligible replacements
could beat both constant velocity and a four-point velocity fit. On OpenDD
development data, this search finds qualifying plans in \DirectCountStrict{}
of \DirectCountViews{} recording--seed views; a past-only policy qualifies
in \DirectCountCount{}. On MovingDroneCrowd++ development data, the
displacement budget rules out beating the four-point fit in
\RecordPeakExcluded{} of \RecordPeakViews{} views, even with unrestricted
forecast shapes. In a frozen evaluation on \WepExtSources{} WEPDTOF
sources, no matched-count joint proposal meets both the risk and accuracy
criteria. These tests distinguish when a better forecast bank is needed
from when the bottleneck is choosing among the forecasts already available."""

OPENING = """Consider two pedestrians in the same video. A neural predictor may
help for someone starting to turn and do worse for someone walking
steadily. Lower average error does not say whether to use both
predictions. We treat each proposed replacement as a choice: keep
the motion forecast or use another one. The choice changes a
prediction, not a person's behavior. We ask whether making these
choices jointly helps under a shared intervention budget and risk limit."""

RELATED = r"""Constant velocity is a strong comparator in pedestrian forecasting~\cite{schoeller}.
Choosing among forecasters is also an established problem~\cite{mao}.
What is less clear is whether a useful joint replacement can be found
without seeing its outcome. Work on raw-video forecasting addresses
tracking errors through re-tracking, denoising, prediction from
detections, or propagation of tracking uncertainty~\cite{yu,natra,realtraj,martins}.
We study a different decision: when an available forecast should
replace a motion reference if future labels are incomplete.

We fix both edit count and displacement to compare who gets a new
forecast, rather than how many forecasts are changed. We then compare
the past-only policies with what a search given the outcomes can achieve
under the same limits. This separates an inadequate set of forecasts
from a policy that cannot find the good options already in it."""


def build(out, style, archive):
    doc = previous.build(out, style, archive, 'main')
    sha = BASE.cal.digest
    tex = out / 'manuscript.tex'
    text, n = re.subn(r'(?s)(\\begin\{abstract\}\n).*?(\n\\end\{abstract\})',
                      lambda m: m[1] + ABSTRACT + m[2], tex.read_text())
    if n != 1 or r'\newcommand{\WepExtFixedDual}{0}' not in (out / 'evidence_values.tex').read_text():
        raise ValueError('Abstract needs the single working draft and the frozen null external result')
    text = BASE.replace_once(text, r'\noindent\textbf{Working evidence manuscript. Confirmation remains sealed.}'
                             + '\n\n\n', '')
    start, end = 'A model can reduce error for some trajectories', 'The observation is eight annotated image positions'
    i, j = text.index(start), text.index(end)
    text = text[:i] + OPENING + '\n\n' + RELATED + '\n\n' + text[j:]
    text = BASE.replace_once(text,
        'OpenAI Codex assisted code development, experimental checks,\n'
        'analysis, figure preparation, and manuscript drafting. The author\n'
        'is responsible for verifying the methods and claims.',
        'OpenAI Codex helped draft and revise manuscript text and locate\n'
        'and check references. It was also used in multi-step workflows to\n'
        'implement code, prepare derived data, test hypotheses, run and\n'
        'analyze experiments, and draw figures from measured results. The\n'
        'author is responsible for the methods, results, and claims.')
    text = BASE.replace_once(text, r'\end{thebibliography}',
        r'''\bibitem{martins}
S. Da Silva Martins, V. Petrovic, E. Aldea, and S. Le H\'egarat-Mascle.
Uncertainty-aware trajectory forecasting from imperfect tracking.
arXiv:2608.30899, 2026.

\end{thebibliography}''')
    tex.write_text(text)
    doc['build_bindings'][tex.name] = sha(tex)
    doc['editorial_code_bindings'] = {f'scripts/{Path(__file__).name}': sha(Path(__file__))}
    (out / 'source_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path)
    parser.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    parser.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    args = parser.parse_args()
    out = DEFAULT / 'build'
    doc = build(out, args.style, args.archive)
    if args.source_zip:
        PACK.main_package.package(out, args.source_zip, doc['build_bindings'])
