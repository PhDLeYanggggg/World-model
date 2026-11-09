"""Rebuild v160 with restrained, print-friendly figures and shorter captions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v160 as previous
from scripts import plot_m3w_manuscript_figures as plot

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v161'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    doc = previous.build(out, style, archive, part)
    sha = BASE.cal.digest
    if part == 'main':
        key, receipt = 'build_bindings', 'source_bindings.json'
        tex = out/'manuscript.tex'
        tex.write_text(BASE.replace_once(tex.read_text(),
            'Hindsight capacity and actual policy outcomes must be\n'
            'interpreted separately. Parentheses indicate switch counts.\n'
            'The 102 views are 34 exposed recordings repeated at three\n'
            'forecaster seeds, not 102 independent recordings.',
            'Hindsight and past-only results on 34 development recordings\n'
            'at three forecaster seeds (102 views). Parentheses give the\n'
            'number of switches.'))
        doc[key][tex.name] = sha(tex)
        for name, draw in (('decision_bottlenecks', plot.decision), ('recipient_feasibility', plot.recipient),
                           ('capacity_discrimination', plot.capacity)):
            figure = out/f'{name}.pdf'
            draw(figure)
            doc[key][figure.name] = sha(figure)
    else:
        key, receipt = 'files', 'supplement_bindings.json'
        src = PUBLIC.parent/'moving_cv_signed_coverage_v1'
        r, m = [json.loads((src/f'{k}.json').read_text()) for k in ('readout', 'matched_gain')]
        figure = out/'signed_coverage_ablation.pdf'
        plot.signed_coverage(figure, r, m)
        doc[key][figure.name] = sha(figure)
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        previous.previous.package_supplement(out, a.source_zip, doc)
