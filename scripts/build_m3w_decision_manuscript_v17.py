"""Integrate the matched neighbor-context and relative-geometry comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v16 as v16

base, v13 = v16.base, v16.v13
PUBLIC = base.HOME/'evidence_manuscript_v17'
DEFAULT = base.DATA/'evidence_manuscript_v17/build'
BIB = r"""
\bibitem{jointmetrics}
E. Weng, H. Hoshino, D. Ramanan, and K. Kitani.
Joint metrics matter: A better standard for trajectory forecasting.
In \emph{ICCV}, 2023.

"""


def values(doc):
    grouped = doc['grouped']
    arms = ('gain', 'ego', 'pooled_motion', 'local_relative_motion')
    if doc['shared_substitution_rows'] != 9675 or any(grouped['risk'][a]['switches'] != 9675 for a in arms):
        raise ValueError('Require the fixed recipients and exact common substitution set')
    rel = doc['relative_geometry']['local_relative_motion_vs_pooled_motion']
    effect = grouped['effects']['local_relative_motion_vs_pooled_motion']
    f = base.citation_metric
    fields = dict(LocalVsPooledLower=f(effect['completion_lower_pp'], 3),
                  LocalPairObserved=f(rel['observed_ade_reduction'], 3),
                  LocalPairLower=f(rel['grid_ade_lower'], 3),
                  LocalPass=grouped['risk']['local_relative_motion']['supported_views'],
                  PooledPass=grouped['risk']['pooled_motion']['supported_views'])
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n'


def build(out, style):
    doc = v16.build(out, style)
    analysis = v13.verified_analysis('native_neighbor_motion_adapter_v1', doc['source_bindings'])
    text = out/'structured_results.tex'
    section = PUBLIC/'context_results.tex'
    text.write_text(text.read_text()+'\n\n'+section.read_text())
    main = out/'manuscript.tex'
    main.write_text(base.replace_once(main.read_text(), r'\end{thebibliography}', BIB))
    definitions = out/'structured_values.tex'
    definitions.write_text(definitions.read_text()+values(analysis))
    doc['source_bindings'][f'evidence_manuscript_v17/{section.name}'] = base.sha(section)
    for name in (main.name, text.name, definitions.name):
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
