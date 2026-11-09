"""Bind the reserved-source WEPDTOF result without promoting a failed guard."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v158 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE,PACK = previous.BASE,previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v159'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def interval(z):
    if z['estimate'] is None:
        return 'not estimable'
    lo,hi = z['ci95']
    return f'{z["estimate"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def build(out, style, archive, part):
    root = PUBLIC.parent
    src = root/'wepdtof_external_transfer_v1'
    camera = root/'wepdtof_external_camera_v1'
    files = [src/k for k in ('registration.json','head_freeze.json','pilot_freeze.json',
        'evaluation_registration.json','forecast_freeze.json','augmented_freeze.json','external_readout.json')]
    files += [camera/k for k in ('registration.json','camera_freeze.json','cold_replay.json')]
    g,h,p,e,f,a,r,c,cam,cold = [json.loads(z.read_text()) for z in files]
    sha = BASE.cal.digest
    if (any(z['registration']!=sha(files[0]) for z in (h,p,f)) or e['training_registration']!=sha(files[0])
            or p['head_sha256']!=sha(files[1]) or a['evaluation_registration']!=sha(files[3])
            or a['original_prediction_freeze']['sha256']!=sha(files[4]) or r['forecast_freeze']!=sha(files[5])
            or not r['external_scoring_completed'] or r['external_tuning'] or r['dronecrowd_confirmation_access']
            or r['deployment_changed'] or r['submission_ready'] or not a['outcomes_unopened']
            or cam['registration']!=sha(files[7]) or cold['registration']!=sha(files[7])
            or cam['annotations_read'] or cold['annotations_read']
            or f['camera']['sha256']!=sha(files[8]) or a['camera']['sha256']!=sha(files[8])):
        raise ValueError('Require exact frozen external paths and one completed outcome readout')
    s = r['summary']
    v = dict(Online=s['totals']['online'],Matched=s['totals']['matched'],Complete=s['totals']['full'],
        Unavailable=s['totals']['missing'],Ref=s['totals']['reference_histories'],
        Sources=s['sources'],Views=s['recording_seed_views'],Windows=len(cam['records']),
        SupportedWindows=cam['windows_with_forecasts'],Frames=cam['totals']['frames'],
        Updates=f'{h["configured_updates"]:,}',Guard=('joint' if p['guarded_uses_joint'] else 'CV'),
        Easy=f'{p["cut_px"][0]:.3f}',Hard=f'{p["cut_px"][1]:.3f}',
        FixedIndependent=interval(s['matched_comparisons']['fixed_independent']),
        FixedUniform=interval(s['matched_comparisons']['fixed_uniform']),
        FixedNoPair=interval(s['matched_comparisons']['fixed_no_pair']),
        MatchedSources=s['matched_comparisons']['fixed_independent']['estimable_sources'])
    for k,n in (('cv','CV'),('ols4','OLS'),('spline','Spline'),('joint','Strict'),('fixed_joint','Fixed')):
        x = s['arms'][k]
        v[n+'Edits'] = x['edits']
        v[n+'Risk'] = x['risk_passes']
        v[n+'Dual'] = x['dual_passes']
        v[n+'Undefined'] = x['undefined_easy']
        for ref,nr in (('cv','CV'),('ols4','OLS')):
            v[n+nr] = (f'{x["lower_vs_"+ref]["estimate"]:+.2f}'
                        if x['lower_vs_'+ref]['estimate'] is not None else 'n/a')
    doc = previous.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        text = BASE.replace_once(tex.read_text(),
            'where causal intervention falls short; they do not establish reliable\n'
            'generalization. Independent confirmation remains sealed.',
            r'where causal intervention falls short. A one-shot WEPDTOF test covers \WepExtSources{} new sources; '
            r'the pilot guard retains \WepExtGuard{} and the fixed-count joint proposal passes '
            r'\WepExtFixedDual{}/\WepExtViews{} joint risk and utility checks. DroneCrowd confirmation remains sealed.')
        text = BASE.replace_once(text,
            'The development evidence is exposed and the separate release was used for calibration, not independent confirmation.',
            r'The original development sets are exposed. WEPDTOF provides a frozen source-disjoint evaluation, '
            r'not independent confirmation of a safe deployment. Its fixed-count joint proposal gives '
            r'\WepExtFixedIndependent{} conservative pp against matched independent selection '
            r'(\WepExtMatchedSources{}/\WepExtSources{} estimable sources; supplement).')
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'wepdtof_external.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        text = BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor)
        text = BASE.replace_once(text,
            'for a frozen one-shot transfer test and have no forecast readout.',
            'for the frozen one-shot transfer test reported in the next section.')
    tex.write_text(text)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\WepExt{k}}}{{{z}}}' for k,z in v.items())+'\n')
    doc['source_bindings'].update({str(x.relative_to(root)):sha(x) for x in files})
    doc[key].update({x.name:sha(x) for x in (tex,defs)})
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
