"""Build a one-page table from replay-verified development ablations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_paper_v10 import sha, write_json
import json

BASE = ROOT/'outputs/publication_readiness_2026_09'
OUT = BASE/'evidence_manuscript_v10'
BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/activation_appendix_build'


def table_row(label, doc, left, right, pair_key, hits):
    g = doc['grouped']
    a, b = (g['risk'][k] for k in (left, right))
    if a['switches'] != b['switches'] or g['views'] != 102:
        raise ValueError('Matched table requires exact switch counts and the original view roster')
    v = g['effects'][pair_key]['completion_lower_pp']
    lo, hi = v['interval95']
    return (f"{label} & {a['switches']:,} & {hits[0]} to {hits[1]} & "
            f"{a['supported_views']} to {b['supported_views']} & "
            f"{v['point']:+.5f} [{lo:+.5f}, {hi:+.5f}] \\\\")


def main():
    sources = {}

    def read(f):
        sources[str(f.relative_to(ROOT))] = sha(f)
        return json.loads(f.read_text())

    def get(name):
        f = BASE/name/'analysis.json'
        v = read(f)
        for phase in ('score', 'replay'):
            if read(f.with_name(f'{phase}_execution.json'))['artifact']['sha256'] != sha(f):
                raise ValueError('Study replay not verified')
        if v['independent_confirmation'] or v['deployment_changed'] or not v['original_rules_unchanged']:
            raise ValueError('Wrong evidence scope')
        return v

    retrieval, rank, activation, value = map(get, ('native_support_retrieval_v1',
        'native_feasible_slate_v1', 'native_activation_v1', 'native_value_activation_v1'))
    rows = [
        table_row('Gain to support retrieval', retrieval, 'gain_retrieved', 'support_retrieved', 'support_vs_gain',
            [retrieval['counts'][a]['query_feasible'] for a in ('gain_retrieved', 'support_retrieved')]),
        table_row('Support to feasibility rank', rank, 'gated_support', 'matched_feasible', 'matched_vs_old',
            [rank['feasible_queries'][a] for a in ('gated_support', 'matched_feasible')]),
        table_row('Gain to event activation', activation, 'matched_gain', 'matched_probability', 'matched_prob_vs_gain',
            [activation['counts'][a]['locally_feasible_queries'] for a in ('matched_gain', 'matched_probability')]),
        table_row('Gain to value-risk activation', value, 'matched_gain', 'matched_value_risk', 'value_risk_vs_gain',
            [value['counts'][a]['feasible'] for a in ('matched_gain', 'matched_value_risk')]),
    ]
    causal = value['grouped']['risk']['causal_value_risk']
    text = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{booktabs}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\pagestyle{empty}
\begin{document}
{\Large\bfseries Where Do Selective Forecast Substitutions Fail?}

\textbf{Development evidence.} These comparisons reuse one native
8-observation / 12-prediction image-coordinate bank on 34 previously exposed
recordings, grouped into 14 sources and evaluated with three forecast seeds.
The 102 recording-seed views are repeated evaluations, not independent
recordings. Each row holds intervention count fixed within its own comparison;
counts are not matched between rows.

\begin{center}
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Control to treatment & Switches & Local hits & Passes & Paired lower utility [95\% interval] \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}

\textbf{What is matched?}
Retrieval uses the same number of proposals and agent slots for every query
before any gate. Ranking retains the same active queries, predicted gate and
agent counts. The two activation rows fix the proposed actions and rank across
each full recording to match the prior switch budget by recording and action
count. That last comparison is retrospective, not an online operating rule.

\textbf{What is measured?}
A local hit is a proposal with positive conservative gain, nonpositive
all/easy excess, and positive easy reference. Recording passes require
the original 2\% caps and positive reference support; an undefined risk
is not a pass. The utility column gives percentage-point reductions in
error relative to the shared CV reference, with a missing-label
completion penalty. Intervals resample 14 paired source groups 3,000
times after averaging seeds.

\textbf{Interpretation.}
Better local retrieval need not translate into positive utility or a
recording-level pass. The current-query value-risk rule uses disjoint
donor-source thresholds and makes """ + f"{causal['switches']} switches; it passes {causal['supported_views']} of 102 views." + r"""
The retrospective numbers cannot be used to claim risk calibration.
No rule here justifies deployment or demonstrates independent transfer.
\end{document}
"""
    BUILD.mkdir(parents=True, exist_ok=True)
    for p in (OUT/'activation_appendix.tex', BUILD/'activation_appendix.tex'):
        p.write_text(text)
    write_json(OUT/'activation_appendix_bindings.json', dict(source_sha256=sources,
        builder_sha256=sha(Path(__file__)), tex_sha256=sha(OUT/'activation_appendix.tex'),
        independent_confirmation=False, submission_ready=False, new_outcome_readout=False))


if __name__ == '__main__':
    main()
