"""Add the source-excluded slate imitation comparison."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v163 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v164'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    root = PUBLIC.parent
    src = root/'source_excluded_slate_v1'
    files = [src/f'{n}.json' for n in ('registration','freeze','readout','diagnostics')]
    reg,f,r,d = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (r['registration'] != sha(files[0]) or f['registration'] != sha(files[0])
            or r['freeze']['sha256'] != sha(files[1]) or d['readout']['sha256'] != sha(files[2])
            or not f['camera_outcomes_unopened'] or not f['held_site_targets_used_only_for_other_folds']
            or r['external_test_reopened'] or reg['threshold_tuning'] or r['independent_confirmation']
            or r['deployment_changed'] or not r['original_2pct'] or not d['no_new_target_access']):
        raise ValueError('Require the unchanged, source-excluded decision readout')
    summary = r['summary']
    o,c = (summary[k] for k in ('opendd','camera'))
    contrast = summary['opendd_matched']['matched_independent']
    lo,hi = contrast['ci95']
    cam = [s['arms']['joint'] for s in r['camera'].values()]
    v = dict(Fits=len(f['fits']),Steps=f"{f['steps']:,}",Views=o['joint']['views'],
             Pass=o['joint']['dual'],Matched=o['matched_independent']['dual'],
             Sources=contrast['sources'],Utility=f"{contrast['point_pp']:+.3f} [{lo:+.3f}, {hi:+.3f}]",
             CamEdits=c['joint']['edits'],CamPass=c['joint']['dual'],
             CamOls=f"{min(x['lower_vs_ols4'] for x in cam):+.2f} to {max(x['lower_vs_ols4'] for x in cam):+.2f}")
    for name,label in (('joint','Joint'),('independent','Ind'),('matched_independent','Match'),
                       ('matched_uniform','Uniform')):
        for k in ('views','edits','risk','dual'):
            v[label+k.title()] = o[name][k]
    doc = previous.build(out,style,archive,part)
    if part=='main':
        key,receipt = 'build_bindings','source_bindings.json'
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        text = BASE.replace_once(tex.read_text(),
            'The action menu and the ability to recognize useful recipients are distinct limitations.',
            r'The action menu and recipient recognition are distinct limitations. '
            r'Learning full query plans gives \SlatePass{}/\SlateViews{} passes on held-site '
            r'OpenDD development, versus \SlateMatched{} for the same-budget independent control. '
            'Paired utility does not establish a joint advantage (supplement).')
    else:
        key,receipt = 'files','supplement_bindings.json'
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        frag = PUBLIC/'slate_imitation.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Slate{k}}}{{{x}}}' for k,x in v.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)):sha(p) for p in files})
    doc[key].update({p.name:sha(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__=='__main__':
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
