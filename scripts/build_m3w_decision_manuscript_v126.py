"""Bind the fitting-only admission experiment and withholding diagnosis."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v125 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v126'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent/'moving_cv_admission_v1'
    files = [src/f'{k}.json' for k in ('registration','freeze','readout','withholding_capacity','unpruned_capacity')]
    reg, freeze, result, capacity, extra = [json.loads(f.read_text()) for f in files]
    digest = BASE.cal.digest
    if (result['registration'] != digest(files[0]) or result['freeze'] != digest(files[1])
            or any(x['parent']['sha256'] != digest(files[2]) for x in (capacity,extra))
            or result['a_labels_opened'] or result['independent_confirmation'] or not result['original_2pct']):
        raise ValueError('Require the exact fitting-only admission evidence')
    effects, risk = (result['summary'][k] for k in ('effects','risk'))
    def fmt(key):
        d = effects[key]['completion_lower_pp']
        lo,hi=d['interval95']
        return f'{d["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    values = dict(Qualified=result['qualified'], Calibration=len(result['calibration']),
        Updates=f'{freeze["updates"]:,}', Edits=f'{risk["relaxed"]["switches"]:,}',
        Independent=risk['relaxed']['supported_views'], Joint=risk['relaxed_joint']['supported_views'],
        Gain=risk['relaxed_gain_match']['supported_views'], RiskContrast=fmt('relaxed_risk_vs_gain_match'),
        JointContrast=fmt('relaxed_joint_vs_independent'),
        CV=capacity['status']['cv_only']['verified'], Dual=capacity['status']['dual']['verified'],
        Unpruned=extra['status']['unpruned']['verified'], AddBack=extra['status']['ols4_addback']['verified'])
    doc = parent.build(out,style,archive,part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt='build_bindings','source_bindings.json'
        anchor=r'for agent error and \CVSelectorPairPoint{} pixels for pair displacement (supplement).'
        new=(r' A separate fitting-only admission check qualifies '
             r'\AdmissionQualified{}/\AdmissionCalibration{} calibrators; even hindsight withholding '
             r'of fixed nominees recovers only \AdmissionDual{}/72 dual-reference views.')
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+new))
    else:
        tex,defs=out/'supplement.tex',out/'supplement_values.tex'
        key,receipt='files','supplement_bindings.json'
        frag=PUBLIC/'cv_admission.tex'
        files.append(frag)
        anchor=r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\Admission{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(f.relative_to(PUBLIC.parent)):digest(f) for f in files})
    doc[key].update({f.name:digest(f) for f in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=BASE.v22.STYLE)
    p.add_argument('--archive',type=Path,default=BASE.v22.supp.base.REPLAY)
    a=p.parse_args()
    out=DEFAULT/('build' if a.part=='main' else 'supplement')
    doc=build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        PACK.supplement_package.package(out,a.source_zip,doc)
