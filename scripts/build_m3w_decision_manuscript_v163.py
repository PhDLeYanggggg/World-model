"""Bind the causal full-path selection test without changing prior scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v162 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v163'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root/'camera_full_path_selection_v1'
    files = [src/f'{n}.json' for n in ('registration_v2','freeze','readout','failure_analysis',
                                      'signed_registration','signed_freeze','signed_readout')]
    reg,f,r,diag,sreg,sfreeze,signed = [json.loads(p.read_text()) for p in files]
    sha = BASE.cal.digest
    if (any(z['registration'] != sha(files[0]) for z in (f,r,diag))
            or r['freeze']['sha256'] != sha(files[1]) or diag['readout']['sha256'] != sha(files[2])
            or not f['camera_targets_not_read'] or not f['no_camera_selection']
            or reg['threshold_tuning'] or diag['camera_tuning']
            or r['independent_confirmation'] or r['external_test_reopened']
            or not r['original_2pct'] or r['deployment_changed']
            or sreg['parent']['sha256'] != sha(files[1]) or sreg['threshold_tuning']
            or not sreg['frozen_gain_head'] or sfreeze['registration'] != sha(files[4])
            or signed['registration'] != sha(files[4]) or signed['freeze']['sha256'] != sha(files[5])
            or sfreeze['camera_outcomes_read'] or signed['external_test_reopened']
            or not signed['original_2pct'] or signed['deployment_changed']):
        raise ValueError('Require matching causal development readout')
    seed = ('17','29','43')
    rows = [r['seeds'][k] for k in seed]
    if len(f['predictors']) != 42 or len(f['value_heads']) != 3:
        raise ValueError('Require all folded producer and selector fits')
    full = [x['arms']['expanded_raw_joint'] for x in rows]
    eq = [x['arms']['expanded_equal_joint'] for x in rows]
    v = dict(Fits=len(f['predictors']), Updates=f"{f['configured_updates']:,}",
             RawRisk=sum(x['risk']['supported'] for x in full),
             RawDual=sum(x['dual_pass'] for x in full), EqualDual=sum(x['dual_pass'] for x in eq),
             FullCapacity=sum(diag['seeds'][s]['expanded_raw']['capacity']['solver']['status']=='verified' for s in seed),
             EqualCapacity=sum(diag['seeds'][s]['expanded_equal']['capacity']['solver']['status']=='verified' for s in seed))
    repair = [signed['seeds'][s]['signed_raw'] for s in seed]
    easy = [100*x['risk']['completion_easy_risk_upper'] for x in repair]
    v.update(SignedEdits=', '.join(str(x['risk']['selected']) for x in repair),
             SignedDual=sum(x['dual_pass'] for x in repair),
             SignedRisk=f'{min(easy):.1f}--{max(easy):.1f}',
             EasyRejected=sum(diag['seeds'][s]['expanded_raw']['predicted']['nonpositive_easy_excess']==0
                              for s in seed))
    for key,x,d,e in zip(('A','B','C'),rows,full,eq,strict=True):
        paired = x['paired_joint_known_lower_pp']['expanded_equal_independent']
        v[key+'Edits'] = d['risk']['selected']
        v[key+'Equal'] = e['risk']['selected']
        v[key+'Gain'] = f"{e['lower_vs_ols4']:+.2f}"
        v[key+'Direct'] = 'n/a' if paired is None else f'{paired[1]:+.2f}'
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        key,receipt = 'build_bindings','source_bindings.json'
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        text = BASE.replace_once(tex.read_text(), 'without a neural candidate (supplement).',
            r'without a neural candidate. A value head trained on held-site donor predictions '
            r'passes both risk and utility in \FullPathRawDual{}/3 seed views on the same '
            'exposed camera (supplement).')
    else:
        key,receipt = 'files','supplement_bindings.json'
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        frag = PUBLIC/'full_path_selection.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),
            'capacity under the changed action set. No causal\nselector for these larger actions has been evaluated.',
            'capacity under the changed action set. The next experiment\ntests whether a past-only selector can recover that capacity.')
        text = BASE.replace_once(text,anchor,frag.read_text()+'\n'+anchor)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\FullPath{k}}}{{{x}}}' for k,x in v.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(root)):sha(p) for p in files})
    doc[key].update({p.name:sha(p) for p in (tex,defs)})
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
