"""Redraw the two figures placed in v164 without changing the evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v164 as previous
from scripts import plot_m3w_manuscript_figures_v165 as plot

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v165'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    doc = previous.build(out, style, archive, part)
    sha = BASE.cal.digest
    if part == 'main':
        key, receipt = 'build_bindings', 'source_bindings.json'
        tex, figure = out / 'manuscript.tex', out / 'decision_bottlenecks.pdf'
        plot.decision(figure)
        old = 'Hindsight and past-only results on 34 development recordings\n'
        old += 'at three forecaster seeds (102 views). Parentheses give the\nnumber of switches.'
        new = ('Counts from 34 development recordings at three forecaster seeds.\n'
               'The hindsight gain count requires a positive lower bound against OLS4.\n'
               'All-risk violations exceed the 2\\% limit relative to CV. '
               'Parentheses give switch counts.')
    else:
        key, receipt = 'files', 'supplement_bindings.json'
        tex, figure = out / 'supplement.tex', out / 'signed_coverage_ablation.pdf'
        src = PUBLIC.parent / 'moving_cv_signed_coverage_v1'
        paths = [src / f'{name}.json' for name in ('readout', 'matched_gain')]
        if any(doc['source_bindings'][str(p.relative_to(PUBLIC.parent))] != sha(p) for p in paths):
            raise ValueError('Coverage figure sources must match the bound readout')
        plot.coverage(figure, *(json.loads(p.read_text()) for p in paths))
        old = ('Support recognition improves, but positive effects on known\n'
               'labels do not survive the missing-step utility bound. Intervals are\n'
               'paired source bootstrap intervals. The dashed line marks all 2,453\n'
               'query views with a true seed action.')
        new = ('Recovered seed actions and matched-count gain relative to the frozen rule. '
               'In (a), the reference line marks all 2,453 query views with a true seed action. '
               'In (b), intervals are paired 95\\% source bootstrap intervals. '
               'The full-grid lower bound accounts for missing future steps.')
    tex.write_text(BASE.replace_once(tex.read_text(), old, new))
    doc[key].update({p.name: sha(p) for p in (tex, figure)})
    doc['figure_code_bindings'] = {str(p.relative_to(PUBLIC.parent.parents[1])): sha(p)
                                   for p in (Path(__file__).resolve(), Path(plot.__file__))}
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=('main', 'supplement'))
    parser.add_argument('--source-zip', type=Path)
    parser.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    parser.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    args = parser.parse_args()
    out = DEFAULT / ('build' if args.part == 'main' else 'supplement')
    doc = build(out, args.style, args.archive, args.part)
    if args.source_zip and args.part == 'main':
        PACK.main_package.package(out, args.source_zip, doc['build_bindings'])
    elif args.source_zip:
        previous.package_supplement(out, args.source_zip, doc)
