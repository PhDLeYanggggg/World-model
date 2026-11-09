"""Add checked CV-anchored neural menu capacity to the working manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v123 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v124'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent / 'moving_cv_joint_feasibility_v1'
    files = [root / f'{k}.json' for k in ('registration', 'readout', 'confirmed_new_witnesses', 'matched_shape_diagnostic')]
    digest = BASE.cal.digest
    result, diagnostics, shape = (json.loads(f.read_text()) for f in files[1:])
    if (result['registration'] != digest(files[0]) or diagnostics['parent']['sha256'] != digest(files[1])
            or shape['parent']['sha256'] != digest(files[1]) or not shape['post_hoc']
            or result['independent_confirmation'] or not result['original_2pct']):
        raise ValueError('Require the bound development-only witness replay')
    values = {}
    for k, name in (('8', 'Eight'), ('32', 'Wide')):
        v = result['summary']['caps'][k]
        for label, value in (('Analytic', v['analytic']['verified']), ('Expanded', v['expanded']['verified']),
                             ('New', v['confirmed_new_views'])):
            values[label + name] = value
        values['Rematch' + name] = shape['statuses'][k].get('verified_analytic_rematch', 0)
        values['Pair' + name] = f'{shape["pair_geometry"][k]["ols4"]["observed_gain_pixel"]:+.3f}'
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = r'the analytic menu and relaxed original risk constraints (supplement).'
        new = (r'At a cap of eight, frozen neural alternatives supply \CVJointNewEight{} checked '
               r'views whose analytic controls are ruled out, not a learned selection gain.')
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, anchor + ' ' + new))
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'cv_joint_feasibility.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        body = BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor)
        end = r'\end{thebibliography}'
        citation = (r'\bibitem{jointmetrics} E. Weng, H. Hoshino, D. Ramanan, and K. Kitani. '
                    r'Joint metrics matter: A better standard for trajectory forecasting. '
                    r'In \emph{ICCV}, 2023.')
        tex.write_text(BASE.replace_once(body, end, citation + '\n\n' + end))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\CVJoint{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(f.relative_to(PUBLIC.parent)): digest(f) for f in files})
    doc[key].update({f.name: digest(f) for f in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
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
        PACK.supplement_package.package(out, a.source_zip, doc)
