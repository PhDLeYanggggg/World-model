"""Add the variable-count scene intervention experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v140 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v141'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_joint_count_v1'
    fragment = PUBLIC / 'joint_count.tex'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'capacity.json')]
    reg, freeze, readout, capacity = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (freeze['registration'] != h(files[0]) or readout['freeze'] != h(files[1])
            or capacity['parent']['sha256'] != h(files[2]) or not freeze['before_held_scoring']
            or not reg['analytic_only'] or not capacity['labels_hindsight_only']
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                   for z in (readout, capacity))):
        raise ValueError('Require frozen development-only joint-count results')
    risk = readout['summary']['risk']
    if len({risk[k]['switches'] for k in ('joint', 'no_pair_matched', 'independent_matched')}) != 1:
        raise ValueError('Joint-count comparator lost exact count match')
    a, b = (capacity['summary'][k] for k in ('fixed_count', 'variable_count'))
    values = dict(Updates=f'{freeze["optimizer_steps"]:,}', Edits=f'{risk["joint"]["switches"]:,}',
        Qual=readout['qualified'], Passes=risk['joint']['supported_views'],
        Worst=f'{risk["joint"]["worst_signed_easy_percent"]:.2f}',
        Verified=f'{a["verified"]}/{b["verified"]}',
        Failed='/'.join(str(b[k]) for k in ('excluded', 'lp_infeasible', 'unsupported')))
    for name, key in (('Ind', 'joint_at_count'), ('Count', 'count_choice'), ('Pair', 'pair_features_at_count')):
        v = readout['summary']['effects'][key]
        q = v['completion_lower_pp']
        p, lo, hi = (f'{x:+.3f}' for x in (q['point'], *q['interval95']))
        values[name] = f'{p} [{lo}, {hi}]'
        values[name+'Known'] = f'{v["known_pp"]["point"]:+.3f}'
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\JointCount{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): h(p) for p in (*files, fragment)})
    doc['files'].update({p.name: h(p) for p in (tex, defs)})
    (out / 'supplement_bindings.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT / ('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out, a.source_zip, doc)
