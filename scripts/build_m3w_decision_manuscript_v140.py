"""Add the matched-count selected-harm study to the working supplement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v139 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v140'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_selected_harm_v1'
    fragment = PUBLIC / 'selected_harm.tex'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'capacity.json')]
    _, freeze, r, capacity = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (freeze['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or capacity['parent']['sha256'] != h(files[2]) or not freeze['before_held_scoring']
            or not capacity['label_aware_upper_only'] or not capacity['ignores_risk']
            or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                   for z in (r, capacity))):
        raise ValueError('Require frozen development-only selected-harm results')
    risk = r['summary']['risk']
    keys = ('parent', 'median', 'tail')
    if len({risk[k]['switches'] for k in keys}) != 1:
        raise ValueError('Selected-harm count control changed')
    values = dict(Updates=f'{freeze["optimizer_steps"]:,}', Edits=f'{risk["tail"]["switches"]:,}',
        Passes='/'.join(str(risk[k]['supported_views']) for k in keys),
        Worst='/'.join(f'{risk[k]["worst_signed_easy_percent"]:.2f}' for k in keys), Qual=r['qualified'],
        Exceed='/'.join(f'{100*r["parent_nomination_exceedance"][k][0]["point"]:.1f}' for k in keys[1:]),
        Capacity='/'.join(str(capacity['summary'][k]['ols4']['nonpositive'])
            for k in ('one_path_exact', 'analytic_exact', 'full_exact', 'full_at_most', 'full_free')))
    for name, key in (('Parent', 'tail_vs_parent'), ('Median', 'tail_vs_median')):
        a = r['summary']['effects'][key]['completion_lower_pp']
        p, lo, hi = (f'{x:+.3f}' for x in (a['point'], *a['interval95']))
        values[name] = f'{p} [{lo}, {hi}]'
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\SelectedHarm{k}}}{{{v}}}' for k, v in values.items()) + '\n')
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
