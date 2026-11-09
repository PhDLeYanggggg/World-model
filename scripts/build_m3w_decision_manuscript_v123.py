"""Bind the CV-anchored capacity result to the working main and supplement."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v122 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v123'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    folder = PUBLIC.parent / 'moving_cv_anchor_budget_v1'
    files = [folder / f'{n}.json' for n in ('registration', 'readout')]
    result = json.loads(files[1].read_text())
    digest = BASE.cal.digest
    if result['registration'] != digest(files[0]) or result['independent_confirmation'] or not result['original_2pct']:
        raise ValueError('Require the bound CV-anchor development audit')
    r = result['summary']
    bound, counts = r['point_pp'], r['positive_records']
    table = '\n'.join(f'{r["caps"][i]} & {counts[2][1][i]} & {counts[0][1][i]} & {counts[1][1][i]} \\\\'
                      for i in (1, 2, 3, 4, 5, 6, 7))
    values = dict(Table=table, VsCV=f'{bound[0][0][1]:+.3f}', Tight=f'{bound[1][1][1]:+.3f}',
                  Arbitrary=f'{bound[2][1][1]:+.3f}', ArbitraryFail=r['no_ols4_at_cap_one_even_arbitrary'],
                  UnlimitedFail=r['records']-counts[1][1][-1])
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = r'positive harm on untouched tracks is already too large (supplement).'
        insert = (r'With CV instead, even arbitrary one-track edits cannot certify '
                  r'gain over OLS4 on \CVAnchorArbitraryFail{}/34 recordings; none can with '
                  r'the analytic menu and relaxed original risk constraints (supplement).')
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, anchor + ' ' + insert))
    else:
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'cv_anchor_budget.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\CVAnchor{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in files})
    doc[key].update({p.name: digest(p) for p in (tex, defs)})
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
