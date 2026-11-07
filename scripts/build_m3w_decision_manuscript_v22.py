"""Add constructive missing-label witnesses to the anonymous decision paper."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v20 as main
from scripts import build_m3w_decision_supplement_v21 as supp

base = main.base
PUBLIC = base.HOME/'evidence_manuscript_v22'
DEFAULT = base.DATA/'evidence_manuscript_v22'
STYLE = supp.base.STYLE


def add_witness(out, doc, fragment, name, macro_file, payload_key, insertion=None):
    result = main.v13.verified_analysis('native_pair_completion_witness_v1', doc['source_bindings'])
    if (not result['released_labels_unchanged'] or not result['all_seeds_use_the_same_world']
            or not result['official_metric_unchanged'] or not result['empirical_opposite_signs']['1']
            or (result['recordings'], result['source_groups'], result['original_queries']) != (34, 14, 1613)):
        raise ValueError('Expected verified coherent development witnesses')
    fields = dict(WitnessLow=f'{result["cases"]["low_1x"]["point"]:+.4f}',
                  WitnessHigh=f'{result["cases"]["high_1x"]["point"]:+.4f}',
                  WitnessInterpolation=f'{result["cases"]["interpolation"]["point"]:+.4f}')
    block = PUBLIC/fragment
    tex, macros = out/name, out/macro_file
    content = tex.read_text()
    if insertion:
        if content.count(insertion) != 1:
            raise ValueError('Expected one supplement insertion point')
        content = content.replace(insertion, block.read_text()+'\n\n'+insertion)
    else:
        content += '\n\n'+block.read_text()
    tex.write_text(content)
    macros.write_text(macros.read_text()+'\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in fields.items())+'\n')
    doc['source_bindings'][f'{PUBLIC.name}/{fragment}'] = base.sha(block)
    for p in (tex, macros):
        doc[payload_key][p.name] = base.sha(p)
    return doc


def build(out, style, archive, part):
    if part == 'main':
        doc = add_witness(out, main.build(out, style), 'main_witness.tex', 'structured_results.tex',
                          'structured_values.tex', 'build_bindings')
        manifest = out/'source_bindings.json'
    else:
        doc = add_witness(out, supp.build(out, archive, style), 'supplement_witness.tex', 'supplement.tex',
                          'supplement_values.tex', 'files',
                          insertion=r'\section{Why The Matched Recording Policies Fail}')
        manifest = out/'supplement_bindings.json'
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--archive', type=Path, default=supp.base.REPLAY)
    p.add_argument('--style', type=Path, default=STYLE)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--source-zip', type=Path)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip:
        if a.part == 'main':
            main.v13.package(out, a.source_zip, doc['build_bindings'])
        else:
            supp.prior.package(out, a.source_zip, doc)
