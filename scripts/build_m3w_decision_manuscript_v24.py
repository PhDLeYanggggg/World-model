"""Insert verified DroneCrowd calibration outcomes, never confirmation data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v23 as prior
from scripts import report_m3w_dronecrowd_calibration as cal

v22 = prior.prior
PUBLIC = v22.base.HOME/'evidence_manuscript_v24'
DEFAULT = v22.base.DATA/'evidence_manuscript_v24'
SECOND = PUBLIC.parent/'dronecrowd_forecast_diagnostics_v1'


def values():
    official, bindings = cal.verify_artifacts(ROOT,cal.PUBLIC,cal.MANIFEST)
    table = cal.summarize(official)
    analysis = SECOND/'analysis.json'
    for phase in ('score','replay'):
        cal.verify_receipt(ROOT,cal.read(SECOND/f'{phase}_execution.json'),analysis)
    secondary = cal.read(analysis)
    if (secondary['official_bindings'] != bindings or not secondary['calibration_only']
            or secondary['confirmation_outcome_access'] or secondary['primary_winner_unchanged'] is not None
            or table['eligible_arms'] or secondary['views'] != 120):
        raise ValueError('Expected completed calibration and no confirmation admission')
    rows = {r['arm']:r for r in table['rows']}
    gate = rows['gate_0']
    names = {'independent':'Independent', 'scene':'Scene', 'pair_additive':'Additive','joint':'Joint',
             cal.BASE:'CV', cal.OLS:'OLS'}
    fmt = lambda x: f'{x:+.3f}'
    f = {}
    for a,name in names.items():
        f['Pass'+name] = str(rows[a]['supported_views'])
        f['CvGain'+name] = fmt(rows[a]['vs_cv']['completion_lower_pp']['point']) if rows[a]['vs_cv'] else '--'
        for field in ('ade','fde'):
            x = secondary['absolute_pixel'][f'{a}__all__{field}']
            f[field.title()+name] = f'{x["point"]:.3f}'

    def effect(x, digits=3):
        point = f'{x["point"]:+.{digits}f}'
        low,high = [f'{y:+.{digits}f}' for y in x['interval95']]
        return point+r'\;['+low+', '+high+']'

    s = official['summary']['effects']
    pair = secondary['joint_gain_vs_matched_controls_pixel']
    f.update(GateSwitch=f'{gate["switches"]:,}', GatePass=str(gate['supported_views']), GateUndefined=str(gate['undefined_easy_views']),
        GateGain=effect(gate['vs_cv']['completion_lower_pp'],4).replace('[+0.0000','[0.0000'),
        JointSwitch=f'{rows["joint"]["switches"]:,}',
        JointOLSGain=effect(s['joint_vs_ols4']['completion_lower_pp']),
        JointIndependentGain=effect(s['joint_vs_independent']['completion_lower_pp']),
        JointAdditiveGain=effect(s['joint_vs_pair_additive']['completion_lower_pp']),
        PairIndependent=effect(pair['independent__grid_lower']), PairScene=effect(pair['scene__grid_lower']),
        PairAdditive=effect(pair['pair_additive__grid_lower'],5),
        HardRows=str(sum(r['support']['hard']['ade_rows'] for r in secondary['recordings'])),
        HardMissing=str(secondary['absolute_pixel'][f'{cal.BASE}__hard__ade']['missing_source_seed_cells']),
        Rows=f'{bindings["calibration_counts"]["eligible"]:,}', Queries=f'{bindings["calibration_counts"]["queries"]:,}')
    paths = [cal.PUBLIC/x for x in ('analysis.json','forecast_freeze.json','replay_execution.json')]+[
             analysis, SECOND/'replay_execution.json']
    return f, {str(p.relative_to(PUBLIC.parent)):cal.digest(p) for p in paths}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Missing unique manuscript anchor: {old}')
    return text.replace(old,new)


def build(out,style,archive,part):
    doc = prior.build(out,style,archive,part)
    f, bindings = values()
    block = PUBLIC/f'{part}_calibration.tex'
    doc['source_bindings'].update(bindings)
    doc['source_bindings'][f'{PUBLIC.name}/{block.name}'] = v22.base.sha(block)
    if part == 'main':
        tex, protocol, macros = [out/name for name in ('manuscript.tex','protocol.tex','structured_values.tex')]
        body = replace_once(tex.read_text(),
            r'\noindent\textbf{Working evidence manuscript. No independent results are included.}',
            r'\noindent\textbf{Working evidence manuscript. Confirmation remains sealed.}')
        body = replace_once(body,'results do not establish deployable improvement or generalization.',
            'results are followed by a frozen transfer to 40 DroneCrowd calibration recordings. At equal intervention counts, joint choices improve over independent choices, but add little over additive pair costs. OLS4 remains much stronger, and no threshold meets the calibration rule. Independent confirmation is not run.')
        body = replace_once(body,r'\section{Limitations and Independent Evidence}',
            block.read_text()+'\n\n'+r'\section{Limitations and Independent Evidence}')
        body = replace_once(body,'These are exposed source-held-out experiments, not independent\nconfirmation.',
            'The development evidence is exposed and the separate release was used for calibration, not independent confirmation.')
        old = ('An independent evaluation protocol fixes separate DroneCrowd\ncalibration and confirmation groups on the separate release\n'
               'of Wen et al.~\\cite{wen}. This draft contains no\nexternal result. Confirmation is conditional on the preregistered\n'
               'calibration criteria; none of the exposed comparisons can be\nsubstituted for it.')
        body = replace_once(body,old,'None of the calibration or development comparisons substitutes for the 30 sealed confirmation groups.')
        tex.write_text(body)
        protocol.write_text(replace_once(protocol.read_text(),'This manuscript contains no external scores.',
            'We report calibration outcomes only; confirmation is sealed.'))
        files, key, manifest = (tex,protocol,macros),'build_bindings',out/'source_bindings.json'
    else:
        tex,macros = [out/name for name in ('supplement.tex','supplement_values.tex')]
        body = replace_once(tex.read_text(),'No calibration or confirmation scores are\nreported here.',
            'Calibration is reported below; no confirmation score was computed.')
        tex.write_text(replace_once(body,'The de-identified aggregate replay preserves',
            block.read_text()+'\n\nThe de-identified aggregate replay preserves'))
        files,key,manifest = (tex,macros),'files',out/'supplement_bindings.json'
    macros.write_text(macros.read_text()+'\n'.join(f'\\newcommand{{\\DrCal{k}}}{{{v}}}' for k,v in f.items())+'\n')
    for p in files:
        doc[key][p.name] = v22.base.sha(p)
    manifest.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('main','supplement'))
    p.add_argument('--archive',type=Path,default=v22.supp.base.REPLAY)
    p.add_argument('--style',type=Path,default=v22.STYLE)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--source-zip',type=Path)
    a = p.parse_args()
    out = a.output_dir or DEFAULT/('build' if a.part == 'main' else 'supplement')
    d = build(out,a.style,a.archive,a.part)
    if a.source_zip:
        if a.part == 'main':
            v22.main.v13.package(out,a.source_zip,d['build_bindings'])
        else:
            v22.supp.prior.package(out,a.source_zip,d)
