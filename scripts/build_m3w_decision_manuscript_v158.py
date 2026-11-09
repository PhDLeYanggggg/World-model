"""Bind the WEPDTOF source and frozen-camera qualification."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v157 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v158'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root/'wepdtof_source_probe_v1'
    files = [src/k for k in ('registration_v3.json','support.json','camera_freeze.json','cold_replay.json',
        'pilot_readout.json','rapid_registration.json','rapid_freeze.json','rapid_cold.json','rapid_comparison.json',
        'input_contrast.json','external_roles.json')]
    g,s,f,c,m,q,k,l,r,a,roles = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    reg = h(files[0])
    rapid = h(files[5])
    if (any(z['registration_sha256']!=reg for z in (s,f,c,m)) or q['camera_registration']!=reg
            or q['parent_camera_freeze']!=h(files[2]) or m['camera_freeze_sha256']!=h(files[2])
            or any(z['registration']!=rapid for z in (k,l,r)) or r['freeze_sha256']!=h(files[6])
            or any(z['annotation_geometry_read'] for z in (f,c,k,l))
            or a['parent_sha256']!=h(files[8]) or roles['parents']['input_contrast.json']!=h(files[9])
            or roles['external_labels_opened_for_scoring'] or not roles['roles_frozen']
            or any(z['independent_confirmation'] for z in (g,s,f,c,m,q,k,l,r,a,roles))):
        raise ValueError('Require the same frozen image-only pilot and separate scores')
    t = s['total']['4']
    sources = {z['source'] for z in s['clips'] if z['support']['4']['queries']>0}
    v = dict(Clips=g['clips'], Sources=g['sources'], EligibleSources=len(sources), Queries=t['queries'],
        Targets=f'{t["targets"]:,}', Complete=f'{t["complete"]:,}', CompleteRate=f'{100*t["complete"]/t["targets"]:.1f}',
        Ambiguous=sum(z['quality']['ambiguous_identity_ticks'] for z in s['clips']),
        Unlisted=sum(z['quality']['unlisted_images'] for z in s['clips']), Windows=len(f['windows']),
        Ref=sum(z['reference_histories'] for z in m['windows']))
    for key,name in (('mobile','Mobile'),('rapid','Rapid'),('matched','Both')):
        v[name+'Hist'] = r['matched_histories'][key]
        v[name+'Windows'] = r['matched_windows'][key]
    v['MobilePaths'] = sum(z['forecasts'] for z in f['windows'])
    v['RapidPaths'] = sum(z['forecasts'] for z in k['windows'])
    common = r['common_scores']
    if common is not None:
        values = (common['mobile'],common['rapid'])
        def error(z):
            return ', '.join(f'{x:.2f}' for x in z['available_ade_cv_ols4_px'])
        v['Common'] = (f'On the same {values[0]["counts"]["matched"]} matched histories, available-label '
            f'ADE (CV, OLS4) is ({error(values[0])}) px with MobileNet and ({error(values[1])}) px with RAPiD. '
            f'With publisher annotation histories on those same targets, the pair is '
            f'({error(a["results"]["annotation_history"])}) px, reversing the ranking in this pilot.')
    else:
        v['Common'] = 'The shared matched subset did not clear the pilot support floor, so no paired ADE is reported.'
    v['Reserved'] = roles['external_sources']
    doc = previous.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'We also test a CV--OLS4 bridge with a past-only model of label availability (supplement).',
            'We also qualify a separate fisheye source, where the CV--OLS4 ranking reverses between camera and annotation histories in one matched pilot (supplement).')
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'wepdtof_source.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor)
        marker = r'\end{thebibliography}'
        text = BASE.replace_once(text,marker,r'\bibitem{wepdtof} M. O. Tezcan, Z. Duan, M. Cokbas, P. Ishwar, and J. Konrad. '
            r'WEPDTOF: A dataset and benchmark algorithms for in-the-wild people detection and tracking from overhead fisheye cameras. '
            r'In \emph{WACV}, pages 503--512, 2022.'+'\n\n'+marker)
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Wep{k}}}{{{z}}}' for k,z in v.items())+'\n')
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
