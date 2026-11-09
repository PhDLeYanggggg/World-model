"""Bind matched risk-boundary supervision to the working supplement."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v142 as parent
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent / 'evidence_manuscript_v143'
DEFAULT = parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = parent.build(out, style, archive, part)
    if part == 'main':
        return doc
    src = PUBLIC.parent / 'moving_cv_risk_crossing_v1'
    fragment = PUBLIC / 'risk_crossing.tex'
    files = [src / k for k in ('registration.json', 'freeze.json', 'readout.json', 'replay_receipt.json')]
    reg, f, r, replay = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1])
            or replay['parent']['sha256'] != h(files[2]) or not f['before_held_scoring']
            or not reg['train_only_witnesses'] or not r['original_2pct']
            or r['a_labels_opened'] or r['independent_confirmation']):
        raise ValueError('Require frozen fitting-only witness supervision')
    counts = f['teacher_stats']
    n = sum(x['pairs'] for x in counts)
    risk = r['summary']['risk']
    if len({risk[k]['switches'] for k in ('portfolio', 'independent', 'utility_matched', 'risk_matched')}) != 1:
        raise ValueError('Matched risk-boundary edits changed')
    values = dict(Pairs=f'{n:,}', Disagree=f'{100*sum(x["utility_prefers_unsafe"] for x in counts)/n:.1f}',
        Updates=f'{f["optimizer_steps"]:,}', Edits=f'{risk["risk_matched"]["switches"]:,}', Qual=r['qualified'],
        Passes='/'.join(str(risk[k]['supported_views']) for k in ('utility_matched', 'risk_matched')),
        Worst='/'.join(f'{risk[k]["worst_signed_easy_percent"]:.2f}' for k in ('utility_matched', 'risk_matched')))
    for name, k in (('Utility', 'risk_vs_utility_at_count'), ('Independent', 'risk_vs_independent_at_count'),
                    ('Free', 'risk_vs_utility_free')):
        q = r['summary']['effects'][k]['completion_lower_pp']
        p, lo, hi = (f'{x:+.3f}' for x in (q['point'], *q['interval95']))
        values[name] = f'{p} [{lo}, {hi}]'
    tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
    anchor = r'\section*{LLM Usage Statement}'
    tex.write_text(BASE.replace_once(tex.read_text(), anchor, fragment.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\RiskCrossing{k}}}{{{v}}}' for k, v in values.items()) + '\n')
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
