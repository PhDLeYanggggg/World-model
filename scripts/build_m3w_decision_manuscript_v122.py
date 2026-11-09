"""Add the sparse OLS4 anchor impossibility result to the working paper."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v119 as main_parent
from scripts import build_m3w_decision_manuscript_v121 as supp_parent

BASE, PACK = supp_parent.BASE, supp_parent.PACK
PUBLIC = supp_parent.PUBLIC.parent / 'evidence_manuscript_v122'
DEFAULT = supp_parent.DEFAULT.parent / PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent / 'moving_motion_cost_stability_v1'
    files = [root / f'{k}.json' for k in ('readout', 'sparse_anchor_risk', 'freeze')]
    score, bound, _ = [json.loads(f.read_text()) for f in files]
    digest = BASE.cal.digest
    if (score['freeze'] != digest(files[2]) or bound['parent']['sha256'] != digest(files[0])
            or score['independent_confirmation'] or bound['independent_confirmation']
            or not bound['original_2pct'] or bound['max_edits_per_query'] != 1):
        raise ValueError('Require the bound sparse-anchor result')
    r = score['summary']['effects']['min_vs_mean']
    l, u = r['known_pp']['interval95']
    macros = dict(N=bound['recording_count'], Union=bound['impossible_recordings'],
                  All=bound['by_metric']['all']['impossible'], Easy=bound['by_metric']['easy']['impossible'],
                  Known=f'{r["known_pp"]["point"]:+.3f}', KnownCI=f'[{l:+.3f}, {u:+.3f}]',
                  Lower=f'{r["completion_lower_pp"]["point"]:+.3f}')
    if part == 'main':
        doc = main_parent.build(out, style, archive, part)
        tex, defs = out / 'manuscript.tex', out / 'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        anchor = r'\section{Limitations and Independent Evidence}'
        para = (r'With OLS4 as the starting forecast, even an oracle allowed one arbitrary edit '
                r'per query cannot meet the CV-relative 2\% rule on \SparseUnion{}/\SparseN{} '
                r'exposed recordings: the positive harm on untouched tracks is already too large '
                r'(supplement).')
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, para + '\n\n' + anchor))
    else:
        doc = supp_parent.build(out, style, archive)
        tex, defs = out / 'supplement.tex', out / 'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC / 'sparse_anchor.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(), anchor, frag.read_text() + '\n' + anchor))
    defs.write_text(defs.read_text() + '\n' + '\n'.join(
        rf'\newcommand{{\Sparse{k}}}{{{v}}}' for k, v in macros.items()) + '\n')
    doc['source_bindings'].update({str(f.relative_to(PUBLIC.parent)): digest(f) for f in files})
    doc[key].update({f.name: digest(f) for f in (tex, defs)})
    (out / receipt).write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main', 'supplement'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT / ('build' if a.part == 'main' else 'supplement')
    doc = build(out, a.style, a.archive, a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out, a.source_zip, doc['build_bindings'])
    elif a.source_zip:
        PACK.supplement_package.package(out, a.source_zip, doc)
