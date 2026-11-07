"""Add direct relative-path supervision to the matched development evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v17 as v17

base, v13 = v17.base, v17.v13
PUBLIC = base.HOME/'evidence_manuscript_v18'
DEFAULT = base.DATA/'evidence_manuscript_v18/build'


def values(doc):
    label = 'with_relative_vs_individual'
    risk = doc['grouped']['risk']
    if doc['shared_substitutions'] != 9675 or any(
            risk[a]['switches'] != 9675 for a in ('gain', 'individual', 'with_relative')):
        raise ValueError('Expected identical per-query intervention recipients')
    m = doc['relative_geometry'][label]
    u = doc['grouped']['effects'][label]
    fmt = base.citation_metric
    fields = dict(PairTrainedObserved=fmt(m['observed_ade_reduction'], 3),
                  PairTrainedLower=fmt(m['grid_ade_lower'], 3),
                  PairTrainedEnd=fmt(m['observed_fde_reduction'], 3),
                  PairTrainedUtility=fmt(u['completion_lower_pp'], 3),
                  PairTrainedPass=risk['with_relative']['supported_views'],
                  IndividualPass=risk['individual']['supported_views'])
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n'


def build(out, style):
    doc = v17.build(out, style)
    analysis = v13.verified_analysis('native_relative_path_training_v1', doc['source_bindings'])
    block = PUBLIC/'relative_training.tex'
    sec = out/'structured_results.tex'
    sec.write_text(sec.read_text()+'\n\n'+block.read_text())
    macros = out/'structured_values.tex'
    macros.write_text(macros.read_text()+values(analysis))
    doc['source_bindings'][f'evidence_manuscript_v18/{block.name}'] = base.sha(block)
    for f in (sec, macros):
        doc['build_bindings'][f.name] = base.sha(f)
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
