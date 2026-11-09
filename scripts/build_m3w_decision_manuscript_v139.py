"""Bind the amplitude-response experiment to the working supplement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v138 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v139'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    src = PUBLIC.parent / 'moving_cv_amplitude_response_v1'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'response_audit.json')]
    reg, freeze, readout, audit = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (freeze['registration'] != h(files[0]) or readout['freeze'] != h(files[1])
            or audit['parent']['sha256'] != h(files[2]) or not freeze['before_held_scoring']
            or not reg['original_2pct'] or not audit['diagnostic_only']
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                   for z in (readout, audit))):
        raise ValueError('Require frozen development-only amplitude results')
    if part == 'main':
        return doc

    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    fragment = PUBLIC / 'amplitude_response.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    risk, effects = readout['summary']['risk'], readout['summary']['effects']
    values = dict(Updates=f'{freeze["optimizer_steps"]:,}', Qual=readout['qualified'],
        Edits=f'{risk["shape_joint"]["switches"]:,}',
        Risk=risk['shape_joint']['supported_views'], Known=audit['risk']['shape_joint']['known_only_passes'],
        Worst=f'{risk["shape_joint"]["worst_signed_easy_percent"]:.2f}',
        BoostWorst=f'{risk["frozen_boost"]["worst_signed_easy_percent"]:.2f}')

    def ci(point, bounds, precision=3):
        p, lo, hi = [f'{x:+.{precision}f}' for x in (point, *bounds)]
        return f'{p} [{lo}, {hi}]'

    for name, k in (('Count', 'shape_at_count'), ('Portfolio', 'portfolio_at_count'),
                    ('Mass', 'amplitude_at_mass'), ('CV', 'shape_joint_vs_cv'), ('OLS', 'shape_joint_vs_ols4')):
        a = effects[k]['completion_lower_pp']
        values[name] = ci(a['point'], a['interval95'])
    for name, k in (('GainError', 'dual_gain'), ('SignedError', 'signed_easy')):
        a = audit['accuracy'][k]
        values[name] = ci(a['direct_minus_shape'], a['interval95'], 4)
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\AmplitudeStudy{k}}}{{{v}}}' for k, v in values.items()) + '\n')
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
