"""Bind fixed-budget recipient imitation to the evidence manuscript."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v151 as previous
from scripts.build_m3w_decision_manuscript_v131 import package_supplement

BASE, PACK = previous.BASE, previous.PACK
PUBLIC = previous.PUBLIC.parent/'evidence_manuscript_v152'
DEFAULT = previous.DEFAULT.parent/PUBLIC.name


def build(out, style, archive, part):
    src = PUBLIC.parent/'opendd_feasible_recipient_imitation_v1'
    files = [src/n for n in ('registration.json','freeze.json','readout.json','attribution.json')]
    f, r, a = [json.loads(p.read_text()) for p in files[1:]]
    digest = BASE.cal.digest
    if (f['registration'] != digest(files[0]) or r['freeze'] != digest(files[1])
            or a['parent']['sha256'] != digest(files[2]) or not f['before_held_scoring']
            or r['independent_confirmation'] or not a['label_aware_only']
            or any(z['confirmation_access'] or not z['original_2pct'] for z in (r,a))):
        raise ValueError('Require bound development-only recipient comparison')
    s, attr = r['summary'], a['summary']
    values = dict(Fits=len(f['fits']), Updates=f'{f["configured_updates"]:,}',
        Analytic=s['capacity']['analytic']['qualified'], Both=s['capacity']['both_shapes']['qualified'],
        Chosen=attr['risk_teacher']['capacity']['qualified'])
    for k,n in (('anchor','Anchor'),('prefix_match','Prefix'),('gain_teacher','Gain'),('risk_teacher','Risk')):
        for field,label in (('risk_pass','Risk'),('dual_positive','Dual'),('undefined','Undefined')):
            values[n+label] = s['policies'][k][field]
    for k,n in (('risk_teacher_vs_gain_teacher','Primary'),('risk_teacher_vs_prefix_match','AgainstPrefix')):
        v = s['effects'][k]
        lo, hi = v['interval95']
        values[n] = f'{v["point"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    doc = previous.build(out, style, archive, part)
    if part == 'main':
        tex, defs = out/'manuscript.tex', out/'evidence_values.tex'
        key, receipt = 'build_bindings', 'source_bindings.json'
        body = tex.read_text()
        start = 'With OLS4 as the starting forecast, even an oracle'
        before, remainder = body.split(start, 1)
        _, after = remainder.split(r'\section{Limitations and Independent Evidence}', 1)
        content = before + (
            r'\paragraph{A feasible plan is not a learned policy.}'+'\n'
            r'On OpenDD, adding the spline to an analytic recipient menu raises verified hindsight capacity '
            r'from \FeasibleRecipientAnalytic{} to \FeasibleRecipientBoth{} of 606 views at identical query counts '
            r'and radii. A ranker trained from the exact feasible assignments passes only '
            r'\FeasibleRecipientRiskDual{} views, versus \FeasibleRecipientPrefixDual{} for the past-prefix control. '
            r'Even optimizing shape after fixing the learned recipients recovers only '
            r'\FeasibleRecipientChosen{} views. The intervention menu and the ability to recognize useful '
            r'recipients are distinct limitations (supplement). On MovingDroneCrowd++ development data, '
            r'conditional risk training also fails to recover a conservative gain at matched budgets. '
            r'Neither result warrants replacing the baseline.'+'\n\n'+
            r'\section{Limitations and Independent Evidence}'+after)
    else:
        tex, defs = out/'supplement.tex', out/'supplement_values.tex'
        key, receipt = 'files', 'supplement_bindings.json'
        frag = PUBLIC/'feasible_recipient.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        content = BASE.replace_once(tex.read_text(), anchor, frag.read_text()+'\n'+anchor)
    tex.write_text(content)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\FeasibleRecipient{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)): digest(p) for p in files})
    doc[key].update({p.name: digest(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part', choices=('main','supplement'))
    p.add_argument('--source-zip', type=Path)
    p.add_argument('--style', type=Path, default=BASE.v22.STYLE)
    p.add_argument('--archive', type=Path, default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part == 'main' else 'supplement')
    doc = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part == 'main':
        PACK.main_package.package(out,a.source_zip,doc['build_bindings'])
    elif a.source_zip:
        package_supplement(out,a.source_zip,doc)
