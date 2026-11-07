"""Add the forecast-blend and missing-label-debit development contrasts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v15 as v15

base, v13 = v15.base, v15.v13
PUBLIC = base.HOME/'evidence_manuscript_v16'
DEFAULT = base.DATA/'evidence_manuscript_v16/build'


def values(blend, missing):
    if ((blend['views'], blend['matched_switches']) != (102, 9675)
            or (missing['views'], missing['matched_switches']) != (102, 9675)
            or not missing['only_diagnostic']):
        raise ValueError('Require matching exposed development diagnostics')
    if missing['capacity']['original'] != blend['risk_capacity']['native']:
        raise ValueError('The original native comparator changed')
    c = missing['capacity']
    arms = ('omit_utility_debit', 'omit_risk_debit', 'omit_both_debits')
    fields = dict(BlendCausal=blend['risk_capacity']['add_causal_blends']['verified_both'],
                  BlendNeural=blend['risk_capacity']['add_causal_and_learned_blends']['verified_both'])
    for name, key in zip(('Utility', 'Risk', 'Both'), arms, strict=True):
        if c[key]['unresolved']:
            raise ValueError('Manuscript requires fully resolved ablation outcomes')
        fields['No'+name+'Debit'] = c[key]['verified_both']
    fields['StillFailNoDebits'] = missing['views']-c['omit_both_debits']['verified_both']
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{n}}}' for k, n in fields.items())+'\n'


def build(out, style):
    doc = v15.build(out, style)
    blend, missing = (v13.verified_analysis(k, doc['source_bindings']) for k in (
        'native_ols_blend_capacity_v1', 'native_missing_debit_capacity_v1'))
    section = out/'structured_results.tex'
    block = PUBLIC/'missing_debit.tex'
    marker = r'\paragraph{Distilling the recording teacher.}'
    section.write_text(base.replace_once(section.read_text(), marker, block.read_text()+'\n\n'))
    definitions = out/'structured_values.tex'
    definitions.write_text(definitions.read_text()+values(blend, missing))
    doc['source_bindings'][f'evidence_manuscript_v16/{block.name}'] = base.sha(block)
    for name in (section.name, definitions.name):
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
