"""Assemble the exact-count feasibility and decision distillation evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v13 as v13

base = v13.prior
PUBLIC = base.HOME/'evidence_manuscript_v14'
DEFAULT = base.DATA/'evidence_manuscript_v14/build'
NEW_BIB = r"""
\bibitem{spo}
A. N. Elmachtoub and P. Grigas. Smart predict, then optimize.
\emph{Management Science}, 68(1):9--26, 2022.

\bibitem{wilder}
B. Wilder, B. Dilkina, and M. Tambe. Melding the data-decisions pipeline:
Decision-focused learning for combinatorial optimization.
In \emph{AAAI}, 2019.

"""


def values(bound, exact, student):
    b = bound['bounds']
    e = exact['risk_capacity']
    g = student['grouped']
    r = g['risk']
    if any(r[a]['switches'] != r['gain']['switches'] for a in ('student_full', 'student_nomination')):
        raise ValueError('Student does not match the budget')
    fmt = base.citation_metric
    full = g['effects']['student_full_vs_gain']
    result = dict(FixedPeople=b['fixed_neural_count']['counts']['provably_infeasible'],
        FreeNeural=b['free_neural_count']['counts']['provably_infeasible'],
        NominationFeasible=e['fixed_nomination']['verified_feasible'],
        FullFeasible=e['full_forecast_bank']['verified_feasible'],
        StructuredSwitches=f'{r["gain"]["switches"]:,}',
        StudentVsGain=fmt(full['completion_lower_pp'], 3)+' pp',
        StudentKnown=fmt(full['known_pp'], 3)+' pp',
        StudentMissing=f'{full["known_pp"]["point"]-full["completion_lower_pp"]["point"]:.3f}',
        NomineeVsGain=fmt(g['effects']['student_nomination_vs_gain']['completion_lower_pp'], 3)+' pp',
        StudentPass=r['student_full']['supported_views'], NomineePass=r['student_nomination']['supported_views'],
        GainPass=r['gain']['supported_views'], ScreenSwitches=r['student_screen']['switches'],
        ScreenPass=r['student_screen']['supported_views'],
        StudentPair=fmt(student['relative_pair']['observed_ade_reduction'], 3))
    return '\n'.join("\\newcommand{\\"+k+"}{"+str(v)+"}" for k, v in result.items())+'\n'


def build(out, style):
    doc = v13.build(out, style)
    src = doc['source_bindings']
    measures = [v13.verified_analysis(a, src) for a in ('native_fixed_recipient_risk_bound_v1',
                 'native_exact_count_joint_capacity_v1', 'native_structured_imitation_v1')]
    v13.verified_analysis('native_recipient_shortfall_v1', src)
    tex = out/'manuscript.tex'
    source = base.replace_once(tex.read_text(), r'\begin{document}', r'\input{structured_values.tex}'+'\n')
    source = base.replace_once(source, r'\section{Limitations and Independent Evidence}',
                                r'\input{structured_results.tex}'+'\n\n')
    source = base.replace_once(source, r'\end{thebibliography}', NEW_BIB)
    old = ('At identical recipients,\nlearning the neural increment directly worsens conservative utility\n'
           'relative to a CV-difference critic. These experiments identify\n'
           'decision bottlenecks, not deployable improvement or independent\n'
           'generalization.')
    if source.count(old) != 1:
        raise ValueError('Expected the frozen v13 abstract ending')
    source = source.replace(old, (
        'At the same exact query counts, a label-aware solver finds\n'
        r'feasible choices in \NominationFeasible{} views with one frozen path per person,'+'\n'
        'but past-only gain ranking passes only four. Distilling the joint\n'
        'solver into past-only ranks worsens conservative utility. These\n'
        'results do not establish deployable improvement or generalization.'))
    tex.write_text(source)
    block = PUBLIC/'structured_results.tex'
    shutil.copyfile(block, out/block.name)
    src[f'evidence_manuscript_v14/{block.name}'] = base.sha(block)
    (out/'structured_values.tex').write_text(values(*measures))
    fig = out/'recipient_feasibility.pdf'
    subprocess.run([sys.executable, ROOT/'scripts/plot_m3w_recipient_feasibility.py', '--output', fig],
                   check=True)
    if base.sha(fig) != 'e78b7b840587832348b7cf907052d376cab7fd7b1e8772c77a584b05343285c7':
        raise ValueError('Reproduced feasibility figure changed')
    for name in (tex.name, block.name, 'structured_values.tex', fig.name):
        doc['build_bindings'][name] = base.sha(out/name)
    (out/'source_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--style', type=Path, default=base.DATA/'evidence_manuscript_v10/paper_build/cvpr.sty')
    p.add_argument('--source-zip', type=Path)
    a = p.parse_args()
    result = build(a.output_dir, a.style)
    if a.source_zip:
        v13.package(a.output_dir, a.source_zip, result['build_bindings'])
