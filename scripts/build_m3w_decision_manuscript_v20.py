"""Add the fixed-recipient recording feasibility bound to the evidence draft."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v19 as v19

base, v13 = v19.base, v19.v13
PUBLIC = base.HOME/'evidence_manuscript_v20'
DEFAULT = base.DATA/'evidence_manuscript_v20/build'


def build(out, style):
    doc = v19.build(out, style)
    a = v13.verified_analysis('native_recording_blend_feasibility_v1', doc['source_bindings'])
    if a['switches'] != 9675 or sum(a['counts'].values()) != a['views'] or not a['hindsight_only']:
        raise ValueError('Expected the complete registered fixed-recipient hindsight diagnostic')
    values = dict(RiskViews=a['views'], RiskFeasible=a['counts']['checked_feasible'],
        RiskExcluded=a['counts']['risk_infeasible_with_fixed_recipients'],
        RiskUndefined=a['counts']['structural_support_absent'],
        RiskUnknown=a['exclusion_channels']['unknown_radius_alone_excludes_easy'],
        RiskOlsPositive=a['feasible_with_positive_pair_vs_ols'])
    block = PUBLIC/'recording_feasibility.tex'
    sec, macros = out/'structured_results.tex', out/'structured_values.tex'
    prior_block = (v19.PUBLIC/'pair_direction.tex').read_text()
    content = sec.read_text()
    if not content.endswith(prior_block):
        raise ValueError('Expected the previous directional discussion at the end of results')
    sec.write_text(content[:-len(prior_block)]+block.read_text())
    macros.write_text(macros.read_text()+'\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in values.items())+'\n')
    doc['source_bindings'][f'evidence_manuscript_v20/{block.name}'] = base.sha(block)
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
