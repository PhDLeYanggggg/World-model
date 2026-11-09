"""Bind the matched reference bridge and label-support experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v156 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent / 'evidence_manuscript_v157'
DEFAULT = previous.DEFAULT.parent / PUBLIC.name


def estimate(x, digits=3):
    a,b = x['interval95']
    return f'{x["point"]:+.{digits}f} [{a:+.{digits}f}, {b:+.{digits}f}]'


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root / 'moving_reference_bridge_v1'
    files = [src / k for k in ('registration.json','freeze.json','readout.json')]
    g,f,r = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration'] != h(files[0]) or r['freeze'] != h(files[1]) or not f['before_held_scoring']
            or any(z['a_labels_opened'] or z['confirmation_access'] or not z['original_2pct'] for z in (g,r))
            or r['deployment_changed'] or r['independent_confirmation']):
        raise ValueError('Require frozen development-only bridge results')
    s = r['summary']
    v = dict(Fits=sum(len(z['fits']) for z in f['cells']), Updates=f'{f["configured_updates"]:,}',
        Views=s['views'], Records=s['recordings'], Sources=s['source_groups'],
        Selected=r['strict_selected'], Cells=len(f['cells']),
        SupportBrier=f'{r["brier"]["causal_support"]:.4f}', MeanBrier=f'{r["brier"]["mean"]:.4f}',
        BrierContrast=estimate(r['brier_support_minus_mean'],4),
        RawContrast=estimate(s['effects']['support_vs_mean']['completion_lower_pp']),
        MatchedContrast=estimate(s['effects']['support_vs_mean_matched']['completion_lower_pp']),
        MatchedKnown=estimate(s['effects']['support_vs_mean_matched']['known_pp']))
    for k,name in (('bridge','Line'), ('lead_mean','Mean'), ('causal_support','Support'), ('v155_ols','Earlier')):
        v[name+'Lower'] = f'{s["effects"][k+"_vs_ols4"]["completion_lower_pp"]["point"]:+.2f}'
        v[name+'ADE'] = f'{r["forecast"][k]["ade_pixel"]:.2f}'
        v[name+'Risk'] = s['risk'][k]['supported_views']
        v[name+'Dual'] = r['dual'][k]
        v[name+'Peak'] = f'{r["forecast"][k]["mean_peak_pixel"]:.2f}'
    doc = previous.build(out,style,archive,part)
    if part == 'main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'A matched risk-aware forecast also adds no pass at fixed recipients and radii (supplement).',
            'We also test a CV--OLS4 bridge with a past-only model of label availability (supplement).')
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'reference_bridge.tex'
        files.append(frag)
        marker = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),marker,frag.read_text()+'\n'+marker)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(
        rf'\newcommand{{\RefBridge{k}}}{{{x}}}' for k,x in v.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)):h(p) for p in files})
    doc[key].update({p.name:h(p) for p in (tex,defs)})
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
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out,a.source_zip,doc)
