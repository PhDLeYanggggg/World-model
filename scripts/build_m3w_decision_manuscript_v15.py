"""Add the exact dual-reference capacity result to the working paper."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v14 as v14

base, v13 = v14.base, v14.v13
PUBLIC = base.HOME/'evidence_manuscript_v15'
DEFAULT = base.DATA/'evidence_manuscript_v15/build'


def values(result):
    if (result['views'], result['matched_switches']) != (102, 9675) or result['independent_confirmation']:
        raise ValueError('Expected exposed development exact-count experiment')
    fields = {}
    for k, prefix in (('fixed_nomination', 'Fixed'), ('full_forecast_bank', 'Full')):
        a = result['risk_capacity'][k]
        labels = dict(Dual='verified_both', RiskInfeasible='risk_count_infeasible',
                      NegativeBound='negative_maximin_upper_bound', Unresolved='unresolved',
                      CVWitness='previous_cv_maximizer_both')
        fields.update({prefix+name: a[field] for name, field in labels.items()})
        if sum(a[labels[name]] for name in ('Dual', 'RiskInfeasible', 'NegativeBound', 'Unresolved')) != 102:
            raise ValueError('Dual-reference outcomes do not partition the recording views')
    return '\n'.join(f'\\newcommand{{\\{k}}}{{{n}}}' for k, n in fields.items())+'\n'


def build(out, style):
    doc = v14.build(out, style)
    result = v13.verified_analysis('native_dual_reference_capacity_v1', doc['source_bindings'])
    section = out/'structured_results.tex'
    source = section.read_text()
    anchor = r'\paragraph{Distilling the recording teacher.}'
    if source.count(anchor) != 1:
        raise ValueError('Expected a single structured evidence insertion point')
    block = PUBLIC/'dual_reference.tex'
    section.write_text(source.replace(anchor, block.read_text()+'\n\n'+anchor))
    main = out/'manuscript.tex'
    main.write_text(base.replace_once(main.read_text(), 'Distilling the joint\n',
        'Full-bank hindsight search verifies improvement over both motion\n'
        r'references in \FullDual{} of the 102 views. '+'\n'))
    definitions = out/'structured_values.tex'
    definitions.write_text(definitions.read_text()+values(result))
    doc['source_bindings'][f'evidence_manuscript_v15/{block.name}'] = base.sha(block)
    for name in (main.name, section.name, definitions.name):
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
