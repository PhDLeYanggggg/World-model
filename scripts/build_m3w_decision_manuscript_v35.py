"""Bring the matched-motion evidence into the main draft and supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import build_m3w_decision_manuscript_v24 as main
from scripts import build_m3w_decision_supplement_v34 as supp

PUBLIC = supp.PUBLIC.parent/'evidence_manuscript_v35'
DEFAULT = supp.DEFAULT.parent/PUBLIC.name
base = supp.base

ABSTRACT = r"""\begin{abstract}
Lower average trajectory error does not show when a learned forecast
should replace a strong motion baseline. We audit this decision under
partial future labels and recording-level positive-harm constraints.
Our comparisons separate candidate availability from causal selection,
and match both the intervention roster and realized motion budget where
required. On moving-drone development data, hindsight can find useful
replacements that past-only controllers miss. On OpenDD, distributing
the same query movement over more targets restores easy-reference
support and raises risk passes from \AuditJointPass{} to
\AuditSpreadPass{} of 606 recording--seed views, at a small loss of
paired conservative utility. Joint choices at matched displacement and
learning the exact exchange do not establish an accuracy advantage.
On 40 separate DroneCrowd calibration recordings, no frozen threshold
qualifies and a simple velocity fit is much stronger on trajectory
error. These findings distinguish a larger action menu, a safer
observed ratio, and a useful causal decision. Independent confirmation
remains unopened.
\end{abstract}"""


def evidence():
    home = PUBLIC.parent
    name = 'opendd_context_exchange_v1'
    files = [home/name/k for k in ('registration.json','analysis.json','execution.json','replay_execution.json')]
    side = [home/k for k in ('opendd_causal_amplitude_v1/amplitude_spread.json',
        'opendd_support_exchange_v1/oracle_headroom.json','opendd_direct_exchange_v1/analysis.json')]
    reg,d,fresh,replay = [json.loads(p.read_text()) for p in files]
    digest = base.cal.digest(files[1])
    if (d['registration']!=base.cal.digest(files[0]) or d['formal_confirmation'] or d['deployment']
        or not d['matched_query_budget'] or not d['matched_all_recipients'] or not d['original_risk_unchanged']
        or reg['dronecrowd_confirmation_access'] or fresh['new_heads_this_execution']!=84
        or replay['new_heads_this_execution']!=0 or fresh['fresh_exact_replay'] or not replay['fresh_exact_replay']
        or fresh['pid']==replay['pid'] or any(r['analysis_sha256']!=digest for r in (fresh,replay))):
        raise ValueError('Require exact replay of the scene-context study')
    # Reuse the existing checks on the main-text evidence, not just JSON paths.
    _, bindings = supp.evidence()
    spread,oracle,exact = [json.loads(p.read_text()) for p in side]
    values = {}
    for tag,a in (('Joint','joint'),('Spread','query_spread'),('Half','fixed_half')):
        v = spread['summary'][a]
        values.update({tag+'Radius':f'{v["radius_mass"]:.3f}',tag+'Pass':v['passed'],
                       tag+'Undefined':v['undefined_selected_easy']})
    for tag,a in (('Own','own'),('Ped','pedestrian'),('Motor','traffic'),('Shuffle','shuffled_traffic')):
        values.update({tag+'Gain':f'{d["contrasts"][a+"_vs_half"]["point"]:+.4f}',
            tag+'Actions':f'{d["policies"][a]["choice_quality"]["selected"]:,}',tag+'Pass':d['policies'][a]['passes']})
    for tag,a in (('Traffic','traffic_vs_pedestrian'),('ShuffleContrast','traffic_vs_shuffled')):
        v = d['contrasts'][a]
        values[tag] = f'{v["point"]:+.4f}'
        values['TrafficCI' if tag=='Traffic' else 'ShuffleCI'] = '['+', '.join(f'{x:+.4f}' for x in v['ci95'])+']'
    values.update(SpreadLoss=f'{abs(spread["spread_vs_joint"]["point"]):.3f}',
        Oracle=f'{oracle["summary"]["guarded_oracle"]["lower_pp"]:+.3f}',
        PairGuard=f'{exact["summary"]["direct_guard"]["vs_half"]["point"]:+.4f}',
        Motor=f'{sum(d["motor_supported_rows"].values()):,}')
    bindings.update({str(p.relative_to(home)):base.cal.digest(p) for p in (*files,*side)})
    return values,bindings


def build(out,style,archive,part):
    values,bindings = evidence()
    if part=='main':
        doc = main.build(out,style,archive,'main')
        tex,defs = out/'manuscript.tex',out/'structured_values.tex'
        frag = PUBLIC/'main_opendd_audit.tex'
        content = tex.read_text()
        start = content.index(r'\begin{abstract}')
        end = content.index(r'\end{abstract}',start)+len(r'\end{abstract}')
        content = content[:start]+ABSTRACT+content[end:]
        content = base.replace_once(content,r'\section{Limitations and Independent Evidence}',
                                    frag.read_text()+'\n\n'+r'\section{Limitations and Independent Evidence}')
        ref = (r'\bibitem{opendd} A. Breuer, J.-A. Term\"ohlen, S. Homoceanu, and T. Fingscheidt. '
               r'openDD: A large-scale roundabout drone dataset. In \emph{IEEE ITSC}, 2020.')
        tex.write_text(base.replace_once(content,r'\end{thebibliography}',ref+'\n\n'+r'\end{thebibliography}'))
        key,manifest = 'build_bindings','source_bindings.json'
    else:
        doc = supp.build(out,style,archive)
        tex,defs = out/'supplement.tex',out/'supplement_values.tex'
        frag = PUBLIC/'supplement_context_exchange.tex'
        tex.write_text(base.replace_once(tex.read_text(),r'\section*{LLM Usage Statement}',
                                        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}'))
        key,manifest = 'files','supplement_bindings.json'
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\Audit{k}}}{{{v}}}'
        for k,v in values.items())+'\n')
    doc['source_bindings'].update(bindings)
    doc['source_bindings'][f'{PUBLIC.name}/{frag.name}'] = base.cal.digest(frag)
    for p in (tex,defs):
        doc[key][p.name] = base.cal.digest(p)
    (out/manifest).write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=base.v22.STYLE)
    p.add_argument('--archive',type=Path,default=base.v22.supp.base.REPLAY)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/('build' if a.part=='main' else 'supplement')
    result = build(out,a.style,a.archive,a.part)
    if a.source_zip:
        if a.part=='main':
            main.v22.main.v13.package(out,a.source_zip,result['build_bindings'])
        else:
            base.v22.supp.prior.package(out,a.source_zip,result)
