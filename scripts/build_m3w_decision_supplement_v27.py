"""Add a separate OpenDD development diagnostic to the anonymous supplement."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_supplement_v26 as prior

PUBLIC = prior.PUBLIC.parent / 'evidence_manuscript_v27'
DEFAULT = prior.DEFAULT.parent / PUBLIC.name
base = prior.base


def values():
    home = PUBLIC.parent/'opendd_query_rank_v1'
    paths = {name:home/name for name in ('analysis.json','registration.json','execution.json','replay_execution.json')}
    digest = {name:base.cal.digest(p) for name,p in paths.items()}
    read = {name:json.loads(p.read_text()) for name,p in paths.items()}
    result = read['analysis.json']
    start,replay = read['execution.json'],read['replay_execution.json']
    if (result['scope']!='development' or result['independent_confirmation'] or len(result['fits'])!=63
        or result['registration']!=digest['registration.json']
        or any(r['analysis_sha256']!=digest['analysis.json'] or r['phase']!='complete' for r in (start,replay))
        or start['exact_replay'] or not replay['exact_replay'] or start['pid']==replay['pid']):
        raise ValueError('Expected registered, exactly replayed OpenDD development evidence')
    source = result['summary']
    def effect(mode,arm):
        x = source[mode][arm]
        return f'{x["lower_pp"]:+.3f}' + ' [' + ', '.join(f'{a:+.3f}' for a in x['ci95_pp']) + ']'

    a = dict(FullCV=effect('full','rank_cv'),FullDual=effect('full','rank_dual'),
        FullRankVsPoint=effect('full','rank_cv_vs_pointwise'), LowDual=effect('limited20','rank_dual'),
        LowDualEasy=f'{source["limited20"]["rank_dual"]["worst_easy_pct"]:.2f}',
        LowOracle=effect('limited20','hindsight_oracle'),FullOracle=effect('full','hindsight_oracle'),
        RecipientOracle=effect('recipient','hindsight_oracle'),CandidateOracle=effect('candidate','hindsight_oracle'),
        RecipientRank=effect('recipient','rank_cv_vs_pointwise'),CandidateRank=effect('candidate','rank_cv_vs_pointwise'))
    return a, {f'{home.name}/{k}':v for k,v in digest.items()}


def build(out, style, archive):
    macros, evidence = values()
    doc = prior.build(out,style,archive)
    tex,defs = out/'supplement.tex',out/'supplement_values.tex'
    frag = PUBLIC/'supplement_opendd.tex'
    content = base.replace_once(tex.read_text(), 'All trajectory distances are in image pixels.',
        'Unless stated otherwise, trajectory distances are in image pixels. '
        'The OpenDD development diagnostic is reported separately in its publisher coordinates.')
    content = base.replace_once(content,r'\section*{LLM Usage Statement}',
        frag.read_text()+'\n\n'+r'\section*{LLM Usage Statement}')
    citation = (r'\bibitem{opendd} A. Breuer, J.-A. Term\"ohlen, S. Homoceanu, '
                r'and T. Fingscheidt. openDD: A large-scale roundabout drone dataset. '
                r'In \emph{IEEE ITSC}, 2020.')
    tex.write_text(base.replace_once(content,r'\end{thebibliography}',
        citation+'\n\n'+r'\end{thebibliography}'))
    defs.write_text(defs.read_text()+'\n'.join(f'\\newcommand{{\\OpenDD{k}}}{{{v}}}' for k,v in macros.items())+'\n')
    doc['source_bindings'].update(evidence)
    doc['source_bindings'][f'{PUBLIC.name}/{frag.name}'] = base.cal.digest(frag)
    for p in (tex,defs):
        doc['files'][p.name] = base.cal.digest(p)
    (out/'supplement_bindings.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=DEFAULT/'supplement')
    p.add_argument('--source-zip',type=Path)
    p.add_argument('--style',type=Path,default=base.v22.STYLE)
    p.add_argument('--archive',type=Path,default=base.v22.supp.base.REPLAY)
    a = p.parse_args()
    result = build(a.output_dir,a.style,a.archive)
    if a.source_zip:
        base.v22.supp.prior.package(a.output_dir,a.source_zip,result)
