"""Render the recorded utility and harm tradeoffs without opening trajectory labels."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_paper_v10 import latex_document, sha, write_json

HOME = ROOT/'outputs/publication_readiness_2026_09'
LEDGER = HOME/'moving_record_joint_actions_v1/causal_ledger'
OUT = HOME/'evidence_manuscript_v10'
BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/utility_tradeoff_build'


def evidence(folder):
    src = folder/'analysis.json'
    checks = [folder/f'{phase}_execution.json' for phase in ('score', 'replay')]
    for check in checks:
        if json.loads(check.read_text())['artifact']['sha256'] != sha(src):
            raise ValueError('Require byte-identical scored and replayed analysis')
    a = json.loads(src.read_text())
    if len(a['records']) != 102 or a['independent_confirmation'] or not a['original_rules_unchanged']:
        raise ValueError('Require complete exposed development and unchanged rules')
    return a, {str(p.relative_to(ROOT)): sha(p) for p in (src, *checks)}


def main():
    inputs = (
        ('Six versus three calibration sources', LEDGER/'veto_refit/guard/size',
         'six_vs_three_mean', None, 'six'),
        ('Source-stable versus pooled risk', LEDGER/'veto_refit/guard/stability',
         'stable_vs_pooled', 'pooled', 'stable'),
        ('Directional versus direct gain', LEDGER/'directional_utility',
         'directional_vs_direct', 'direct', 'directional'),
        ('Direction only versus direct gain', LEDGER/'directional_utility',
         'tangent_vs_direct', 'direct', 'tangent'),
    )
    hashes, table = {}, []
    for name, folder, contrast, baseline, arm in inputs:
        data, binding = evidence(folder)
        hashes.update(binding)
        effect = data['comparisons'][contrast]
        effect = effect.get('completion_lower_pp', effect)
        ci = effect['interval95']
        end = data['risk'][arm]
        start = data['risk'][baseline] if baseline else data['risk']['three_a']
        if baseline is None and data['risk']['three_b']['passes'] != start['passes']:
            raise ValueError('Calibration controls need separate pass columns')
        if start['switches'] != end['switches'] or end['switches'] != 6952:
            raise ValueError('Appendix scope is the fixed-count seven-source fit')
        v = f"{effect['point']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        risk = f"{start['passes']} to {end['passes']}"
        easy = f"{start['easy_net_over_2']} to {end['easy_net_over_2']}" if baseline else '8/12 to 10'
        table.append(f"| {name} | {v} | {risk} | {easy} | {end['undefined_easy']} |")
    text = '''# Matched Intervention Utility and Observed Harm

## Development Protocol

We hold the candidate trajectories and 6,952 intervention counts fixed query
by query. The data are exposed Moving A, with 8 observed and 12 predicted
annotation steps in native image coordinates. Seven fitting sources and one
held source per cell give 42 source-seed comparisons. Three frozen forecaster
seeds share the 34 recordings, giving 102 views, not 102 independent units.

### Table 1. Completion-lower ADE contrasts and original recording-risk checks

| Component contrast | Difference (pp), 95% source interval | Full passes / 102 | Signed easy net >2% | Undefined easy in new arm |
| --- | --- | --- | --- | --- |
''' + '\n'.join(table) + '''

Each row compares only its own registered control. In the first row the two
three-source contrasts are averaged; their easy violation counts are shown
separately. Intervals use 3,000 whole-source draws after averaging the
forecaster seeds. The bounds account for wholly missing labels using paired
path disagreement; they are not confidence intervals for a complete-grid
ADE effect. The unchanged risk rule includes positive harm, signed easy
degradation and required support.

## What Changed

More calibration data does not improve threshold transfer in this design.
Source deletion at the residual fitting stage reduces some observed harm,
but loses matched utility. The directional gain head changes which forecasts
are used with the same predicted harm, risk budgets and counts. For baseline
error e and candidate displacement d, the ADE gain per step is the projection
of d onto the unit error direction minus a nonnegative remainder. The shared
direction head and per-action remainder are fit only on observed training
labels. The model scores fixed forecasts; it does not alter their paths.

## Interpretation

The directional head's positive point estimate is inconclusive, and four
recording passes do not validate a policy. Direction alone does not improve
the control reliably either. Model-side gains and recording-level safety
must both be demonstrated on eligible independent sources before any
deployment claim. These exploratory model changes do not select a winner
for confirmation. No protected outcomes, Stage5C or SMC were used.
'''
    tex = latex_document(text).replace('[10pt,twocolumn,letterpaper]', '[10pt,onecolumn,letterpaper]')
    tex = tex.replace(r'\begin{table*}[t]\centering\small',
                      r'\noindent\begin{minipage}{\textwidth}\small')
    tex = tex.replace(r'\caption{', r'\captionof{table}{').replace(
        r'\end{table*}', r'\end{minipage}\par\medskip')
    (OUT/'utility_tradeoff_appendix.md').write_text(text)
    (OUT/'utility_tradeoff_appendix.tex').write_text(tex)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD/'main.tex').write_text(tex)
    shutil.copyfile(BUILD.parent/'paper_build/cvpr.sty', BUILD/'cvpr.sty')
    write_json(OUT/'utility_tradeoff_bindings.json', dict(
        source_sha256=hashes, builder_sha256=sha(Path(__file__)),
        tex_sha256=sha(OUT/'utility_tradeoff_appendix.tex'),
        independent_confirmation=False, new_outcome_readout=False, submission_ready=False))


if __name__ == '__main__':
    main()
