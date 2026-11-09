"""Bind signed excess and within-query swap learning evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_decision_manuscript_v130 as parent

BASE, PACK = parent.BASE, parent.PACK
PUBLIC = parent.PUBLIC.parent/'evidence_manuscript_v131'
DEFAULT = parent.DEFAULT.parent/PUBLIC.name


def build(out,style,archive,part):
    src = PUBLIC.parent/'moving_cv_signed_coverage_v1'
    files = [src/f'{k}.json' for k in ('registration','freeze','readout','matched_gain_freeze',
                                      'matched_gain','risk_failures')]
    reg,f,r,mf,m,fail = [json.loads(p.read_text()) for p in files]
    h = BASE.cal.digest
    if (f['registration']!=h(files[0]) or r['freeze']!=h(files[1]) or not f['before_held_scoring']
            or mf['parent']['sha256']!=h(files[2]) or not mf['before_held_scoring']
            or m['freeze']['sha256']!=h(files[3]) or fail['parent']['sha256']!=h(files[2])
            or r['a_labels_opened'] or any(v['independent_confirmation'] or not v['original_2pct'] for v in (reg,r,m,fail))):
        raise ValueError('Require matching signed-risk evidence')
    risk, effect, d = r['summary']['risk'],r['summary']['effects'],r['donors']

    def fmt(v):
        lo,hi = v['interval95']
        return f'{v["point"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'

    values = dict(Updates=f'{f["optimizer_steps"]:,}',Qual=r['qualified'],Views=r['summary']['views'],
        OldDonor=f'{d["old"]["same_action_overlap"]:,}',
        DirectDonor=f'{d["direct"]["same_action_overlap"]:,}',
        ConditionalDonor=f'{d["conditional"]["same_action_overlap"]:,}',
        TrueDonor=f'{d["old"]["actual"]:,}',
        DirectEdits=f'{risk["direct_old"]["switches"]:,}',
        ConditionalEdits=f'{risk["conditional_old"]["switches"]:,}',
        DirectPass=risk['direct_old']['supported_views'],ConditionalPass=risk['conditional_old']['supported_views'],
        DirectSwapPass=risk['direct_swap']['supported_views'],ConditionalSwapPass=risk['conditional_swap']['supported_views'],
        SwapOnDirect=fmt(effect['swap_on_direct']['completion_lower_pp']),
        SwapOnConditional=fmt(effect['swap_on_conditional']['completion_lower_pp']),
        DirectMatched=fmt(effect['direct_vs_gain_match']['completion_lower_pp']),
        ConditionalMatched=fmt(effect['conditional_vs_gain_match']['completion_lower_pp']),
        SwapMatchEdits=f'{m["summary"]["risk"]["conditional_swap"]["switches"]:,}',
        SwapMatchKnown=fmt(m['summary']['effects']['conditional']['known_pp']),
        SwapMatchLower=fmt(m['summary']['effects']['conditional']['completion_lower_pp']),
        KnownEasyOver=fail['arms']['conditional_old']['known_easy_over'],
        MissingFlip=fail['arms']['conditional_old']['missing_debit_flips_known_pass'])
    doc = parent.build(out,style,archive,part)
    if part=='main':
        tex,defs = out/'manuscript.tex',out/'evidence_values.tex'
        key,receipt = 'build_bindings','source_bindings.json'
        anchor = r'where a seed action exists with outcomes, and qualifies no calibration setting (supplement).'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,anchor+
            r' Conditional signed retraining recovers \SignedCoverConditionalDonor{} true seed queries, '
            r'but no calibrator qualifies and matched swap gains on known labels do not survive the '
            r'missing-step bound (supplement).'))
    else:
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        key,receipt = 'files','supplement_bindings.json'
        frag = PUBLIC/'signed_coverage.tex'
        files.append(frag)
        anchor = r'\section*{LLM Usage Statement}'
        tex.write_text(BASE.replace_once(tex.read_text(),anchor,frag.read_text()+'\n'+anchor))
        from scripts import plot_m3w_cv_signed_coverage as plot
        figure = out/'signed_coverage_ablation.pdf'
        plot.draw(r,m,figure)
        doc[key][figure.name] = h(figure)
    defs.write_text(defs.read_text()+'\n'+'\n'.join(rf'\newcommand{{\SignedCover{k}}}{{{v}}}' for k,v in values.items())+'\n')
    doc['source_bindings'].update({str(p.relative_to(PUBLIC.parent)):h(p) for p in files})
    doc[key].update({p.name:h(p) for p in (tex,defs)})
    (out/receipt).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


def package_supplement(out,destination,doc):
    from scripts.build_m3w_review_replay import write_zip
    expected = set(PACK.supplement_package.PAYLOAD)|{'signed_coverage_ablation.pdf'}
    if set(doc['files'])!=expected or any(BASE.cal.digest(out/k)!=v for k,v in doc['files'].items()):
        raise ValueError('Unexpected supplement payload or modified file')
    files = {k:(out/k).read_bytes() for k in expected}
    files['README.txt'] = b'Anonymous working supplement. Supply cvpr.sty and compile supplement.tex.\n'
    write_zip(destination,files)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=BASE.v22.STYLE)
    p.add_argument('--archive',type=Path,default=BASE.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = DEFAULT/('build' if a.part=='main' else 'supplement')
    d = build(out,a.style,a.archive,a.part)
    if a.source_zip and a.part=='main':
        PACK.main_package.package(out,a.source_zip,d['build_bindings'])
    elif a.source_zip:
        package_supplement(out,a.source_zip,d)
