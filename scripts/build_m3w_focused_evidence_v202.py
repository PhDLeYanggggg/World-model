"""Build a clearer anonymous draft around the verified paired comparisons."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_step_budget_candidate as prior

OUT = prior.OUT.parent/'evidence_manuscript_v202_focused_evidence'
RECEIPT = prior.main.PUBLIC/OUT.name/'receipt.json'
BEFORE = r'\paragraph{An available improvement need not be selectable.}'
AFTER = r'\begin{table}[t]'
SOCIAL_BIB = r'''\bibitem{socialstgcnn}
A. Mohamed, K. Qian, M. Elhoseiny, and C. Claudel.
Social-STGCNN: A social spatio-temporal graph convolutional neural
network for human trajectory prediction.
In \emph{CVPR}, pages 14424--14432, 2020.

'''


def inputs():
    files = dict(social='opendd_vehicle_stgcnn_v1/readout.json',
                 eq='opendd_vehicle_eqmotion_v1/readout.json',
                 pairs='opendd_vehicle_stgcnn_v1/backbone_pairs.json',
                 freeze='opendd_vehicle_stgcnn_v1/freeze.json',
                 cases='songdo_failure_cases_v1/readout.json',
                 supplement='songdo_failure_cases_v1/review_receipt.json')
    raw = {k: (prior.main.PUBLIC/v).read_bytes() for k, v in files.items()}
    rows = {k: json.loads(v) for k, v in raw.items()}
    hashes = {k: prior.main.digest(v) for k, v in raw.items()}
    social, eq, pairs, freeze, cases, supplement = (rows[k] for k in files)
    if (social['binding'] != dict(eqmotion=hashes['eq'], freeze=hashes['freeze'])
            or any(pairs['binding'][k] != hashes[k] for k in ('eq', 'social'))
            or freeze['optimizer_updates'] != 16800 or len(freeze['checkpoints']) != 42
            or not social['original_2pct'] or not eq['original_2pct_unchanged']
            or not pairs['original_reference_metrics_reproduced']
            or any(x[k] for x in (social, eq, pairs) for k in
                   ('external_outcomes_opened', 'independent_confirmation', 'deployment_changed'))
            or cases['new_cohort_opened'] or cases['policy_changed']
            or supplement['source_readout_sha256'] != hashes['cases']
            or supplement['figure_sha256'] != cases['identity']['figure_sha256']
            or supplement['companion_pdf_sha256'] != prior.bundle.read_receipt(prior.OUT.name)['artifacts']['manuscript.pdf']):
        raise ValueError('Expected the matched OpenDD results and unchanged Songdo case replay')
    prior.bundle.checked(prior.bundle.LOCAL/'review_supplement_v34.zip', supplement['archive_sha256'])
    return rows, hashes


def bracket(row):
    return '['+', '.join(f'{v:+.2f}' for v in row['ci95'])+']'


def backbone_table(rows):
    social, eq, pairs = (rows[k] for k in ('social', 'eq', 'pairs'))
    m = social['metrics']
    e = eq['metrics']['comoving_peers']
    tab = []
    for name, metrics, gain in (
        ('CV', m['cv'], None), ('OLS4', m['ols4'], m['ols4']['cv_gain_lower_pp']),
        ('Completion-trained', m['completion'], m['completion']['cv_gain_lower_pp']),
        (r'EqMotion~\cite{eqmotion}', e, e['cv_gain']['lower']),
        (r'Social-STGCNN~\cite{socialstgcnn}', m['social_peers'], m['social_peers']['cv_gain_lower_pp']),
    ):
        err = ' & '.join(f"{metrics[k]['mean']:.3f}" for k in ('known_ade_native', 'fde_available_native', 'p95_ade_native'))
        g, risk = (f"{gain['mean']:+.2f}", str(metrics['risk_passes'])) if gain else ('ref.', '--')
        tab.append(f'{name} & {err} & {g} & {risk} '+r'\\')
    b = pairs['paired']['social_peers_minus_eqmotion_peers']['lower']
    p = social['paired']['social_peers_minus_ego']
    o, l = p['observed'], p['lower']
    return rf'''\paragraph{{Matched forecast backbones.}}
We adapt two published architectures to the same deterministic
vehicle task. Both graph models see the same eight-step history and
nearest three observed vehicles and use the same loss, displacement
cap and training budget. The comparison covers 500 OpenDD recordings
at seven held locations, with three seeds per location.
Social-STGCNN's neighbor input improves observed gain by
{o['mean']:+.2f} pp over its ego-only version, but the direct
conservative contrast is {l['mean']:+.2f} pp (95\% location interval
{bracket(l)}). The direct contrast with the matched EqMotion model is
{b['point']:+.2f} pp {bracket(b)}. Table~\ref{{tab:backbones}} reports
the forecasts before release. Neither graph model exceeds the
completion-trained forecaster on these measures. These are
adaptations, not comparisons of the original benchmark scores.

\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Forecast & ADE & FDE & P95 & CV gain & Risk \\
\midrule
{chr(10).join(tab)}
\bottomrule
\end{{tabular}}
\caption{{Matched OpenDD vehicle forecasts. ADE, FDE and p95 ADE
use available labels in release coordinates. CV gain is conservative,
in pp. Risk counts passes out of 1,500
recording--seed views; CV does not edit and has no selected-risk ratio.}}
\label{{tab:backbones}}
\end{{table}}

''', (f"{b['point']:+.2f}", f"{l['mean']:+.2f}", '1,500')


def revise(source, rows):
    start = source.index(BEFORE)
    end = source.index(AFTER, start)
    if source.count(BEFORE) != 1 or r'\label{tab:matched}' not in source[end:]:
        raise ValueError('Expected the detailed vehicle ablations in the v201 draft')
    table, markers = backbone_table(rows)
    abridged = r'''\paragraph{Why nomination still matters.}
On a separate OpenDD development roster of 493 vehicle recordings,
we fix the recipients and their displacement at each step. In
1,393 of 8,874 repeated views, a hindsight rotation of one path
can preserve trajectory utility, improve relative motion and pass
both references and the risk rule. The learned nominator finds
420, even before abstention. Holding the original 526-rotation
budget fixed, action margin gives 54 strict successes versus 41
for a separate opportunity screen. A shortlist of three raises
the hindsight ceiling on the same recording mask from 54 to 85,
but utility-aware selection finds only 56 and worsens relative
motion. The supplementary ablations examine nominee-specific
training, interval targets and a second calibrated replacement.
These change selection on exposed development data; they do not
establish a release improvement.

'''
    text = source[:start]+table+abridged+source[end:]
    anchor = 'The intervals resample sources. Neither check changes release.'
    if text.count(anchor) != 1 or text.count(r'\end{thebibliography}') != 1:
        raise ValueError('Unexpected Songdo discussion or bibliography')
    text = text.replace(anchor, anchor+'''
Three retrospective examples in the supplement show a recipient-driven
loss, a loss with recipients fixed, and an observed gain whose sign
is unresolved once missing future labels are accounted for.''')
    text = text.replace(r'\end{thebibliography}', SOCIAL_BIB+r'\end{thebibliography}')
    return text, (*markers, '1,393 of 8,874', 'Three retrospective examples')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    prior.builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=prior.OUT,
                        read_inputs=inputs, revise_source=revise, added_citations=('socialstgcnn',))
