"""Bind matched screening-order and conservative within-agent loss evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v126 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v127'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    a,b = (PUBLIC.parent/k for k in ('moving_cv_screen_order_v1','moving_cv_paired_nomination_v1'))
    files = [d/f'{name}.json' for d in (a,b) for name in ('registration','freeze','readout')]
    digest = BASE.cal.digest
    s,n = [json.loads((d/'readout.json').read_text()) for d in (a,b)]
    for d,r in ((a,s),(b,n)):
        freeze = json.loads((d/'freeze.json').read_text())
        if (r['freeze'] != digest(d/'freeze.json') or freeze['registration'] != digest(d/'registration.json')
                or r['a_labels_opened'] or r['independent_confirmation'] or not r['original_2pct']):
            raise ValueError('Require verified fitting-only results')

    def effect(summary, key):
        d = summary['effects'][key]['completion_lower_pp']
        lo,hi = d['interval95']
        return f'{d["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    sr,nr = (r['summary']['risk'] for r in (s,n))
    v = {**{k:sr[arm]['supported_views'] for k,arm in
        (('NominateFirst','before'),('AddBack','addback_match'),('FullScreen','screen_match'),('Joint','joint_match'))},
        **{k:nr[arm]['supported_views'] for k,arm in (('Point','point_match'),('Paired','paired_match'))},
        'PointQual':n['qualifying_calibration_cells']['point'],
        'PairedQual':n['qualifying_calibration_cells']['paired'],
        'PointEasy':f'{nr["point_match"]["worst_signed_easy_percent"]:.2f}',
        'PairedEasy':f'{nr["paired_match"]["worst_signed_easy_percent"]:.2f}',
        'UndefinedEasy':nr['paired_match']['undefined_easy_views'],
        'AllOver':nr['paired_match']['all_risk_over_2_views'],
        'SharedWins':n['shared_path_comparison']['paired_definite_wins'],
        'SharedLosses':n['shared_path_comparison']['point_definite_wins'],
        'SharedChanged':n['shared_path_comparison']['changed_shared_paths'],
        'Contrast':effect(n['summary'],'paired_vs_point')}
    doc = parent.build(out,style,archive,part)
    if part == 'main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'of fixed nominees recovers only \AdmissionDual{}/72 dual-reference views.'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' Allowing more candidates through screening does not qualify calibration either (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'cv_paired_nomination.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Nomination{k}}}{{{value}}}' for k,value in v.items())+'\n')
    doc['source_bindings'].update({str(f.relative_to(PUBLIC.parent)):digest(f) for f in files})
    doc[key].update({f.name:digest(f) for f in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=BASE.v22.STYLE)
    p.add_argument('--archive',type=Path,default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    doc = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        PACK.supplement_package.package(out,a.source_zip,doc)
