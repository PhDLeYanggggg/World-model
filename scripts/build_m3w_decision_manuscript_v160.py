"""Bind the camera action-capacity diagnostic to the evidence manuscript."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v159 as previous

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v160'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root/'wepdtof_camera_capacity_v1'
    files = [src/k for k in ('registration.json', 'readout.json', 'protocol.md')]
    registration, readout, _ = files
    reg, r = [json.loads(p.read_text()) for p in files[:2]]
    sha = BASE.cal.digest
    if (r['registration'] != sha(registration) or not r['original_2pct']
            or r['sources'] != 1 or r['held_labels_access'] or r['independent_confirmation']
            or reg['held_scores_or_labels_read'] or reg['source'] != 'convenience_store'):
        raise ValueError('Require the exposed camera diagnostic')
    cells = list(r['cells'].values())
    if sorted(c['seed'] for c in cells) != [17, 29, 43]:
        raise ValueError('Require all three donor seeds')
    first = cells[0]
    metric = first['metrics']['median4_anchored']
    values = dict(Edits=first['edits'], Histories=r['online'],
                  Observed=f"{max(c['bounds']['known_pp'][1] for c in cells):+.2f}",
                  Conservative=f"{max(c['bounds']['conservative_pp'][1] for c in cells):+.2f}",
                  PrefixPass=sum(c['metrics']['prefix_rank']['risk']['supported'] for c in cells),
                  PrefixDual=sum(c['metrics']['prefix_rank']['dual_pass'] for c in cells),
                  ExpandedExclusion=sum(c['expanded']['solver']['status']=='nonpositive_bound' for c in cells),
                  MedianVsOLS=f"{metric['lower_vs_ols4']:+.2f}",
                  MedianEasy=f"{100.*metric['risk']['completion_easy_risk_upper']:.2f}")
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'We also qualify a separate fisheye source, where the CV--OLS4 ranking reverses between camera and annotation histories in one matched pilot (supplement).',
            r'On the exposed fisheye pilot, even arbitrary paths at the frozen per-query count and per-edit radius cannot beat OLS4 on known steps (\WepCapObserved{} pp best upper bound; supplement).')
    else:
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC/'wepdtof_capacity.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(), anchor, frag.read_text()+'\n'+anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\WepCap{k}}}{{{v}}}' for k, v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)): sha(p) for p in files})
    doc[key].update({p.name: sha(p) for p in (tex, defs)})
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
        previous.package_supplement(out, a.source_zip, doc)
