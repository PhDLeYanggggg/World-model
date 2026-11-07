"""Add the continuous pair-slate bound to the matched development evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v18 as v18

base, v13 = v18.base, v18.v13
PUBLIC = base.HOME/'evidence_manuscript_v19'
DEFAULT = base.DATA/'evidence_manuscript_v19/build'


def values(doc):
    if doc['matched_switches'] != 9675 or not doc['hindsight_only']:
        raise ValueError('Expected matched hindsight experiment')
    a, b = (doc['counts'][k] for k in ('shared_hindsight', 'separate_hindsight'))
    if any(v['unresolved'] or sum(v.values()) != doc['active_queries'] for v in (a, b)):
        raise ValueError('Expected all registered active queries resolved')
    risk = doc['recording_readout']['risk']
    fields = dict(DirectionQueries=f'{doc["active_queries"]:,}', DirectionShared=f'{a["verified_gain"]:,}',
        DirectionSeparate=f'{b["verified_gain"]:,}', DirectionOnly=doc['separate_verified_with_shared_no_gain'],
        DirectionSharedPass=risk['shared_hindsight']['supported_views'],
        DirectionSeparatePass=risk['separate_hindsight']['supported_views'])
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n'


def build(out, style):
    doc = v18.build(out, style)
    analysis = v13.verified_analysis('native_pair_blend_direction_v1', doc['source_bindings'])
    block = PUBLIC/'pair_direction.tex'
    sec = out/'structured_results.tex'
    sec.write_text(sec.read_text()+'\n\n'+block.read_text())
    macros = out/'structured_values.tex'
    macros.write_text(macros.read_text()+values(analysis))
    doc['source_bindings'][f'evidence_manuscript_v19/{block.name}'] = base.sha(block)
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
