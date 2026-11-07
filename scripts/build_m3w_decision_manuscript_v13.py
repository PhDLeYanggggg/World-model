"""Integrate the fixed-recipient neural-increment experiment into the working paper."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_m3w_decision_manuscript_v12 as prior

PUBLIC = prior.HOME/'evidence_manuscript_v13'
DEFAULT = prior.DATA/'evidence_manuscript_v13/build'


def verified_analysis(name, bindings):
    rec = prior.read(prior.HOME/name/'replay_execution.json')
    path = prior.source(name+'/analysis.json', rec['artifact']['sha256'], bindings)
    result = prior.read(path)
    if result['external_access'] or result['independent_confirmation']:
        raise ValueError('Expected only exposed development evidence')
    return result


def values(capacity, menu, increment):
    counts = capacity['summary']['arms']
    effect = capacity['summary']['comparisons']['native_with_learned_vs_causal_only']
    rate = effect['source_balanced_positive_rate_change_pp']
    rate = dict(point=rate['point'], interval95=rate['ci95'])
    am, im = (x['grouped'] for x in (menu, increment))
    if (menu['mismatched_queries'] or menu['learned_switches']['mixed_common_screen'] != 0
            or any(im['risk'][k]['switches'] != am['risk']['causal_joint']['switches']
                   for k in ('anchor', 'cv_difference', 'direct_difference'))):
        raise ValueError('Matched recipient or switch-count premise changed')
    fmt = prior.citation_metric
    control = am['effects']['mixed_joint_vs_causal_joint']
    inc = im['effects']
    fields = dict(CausalPositive=counts['causal_only']['positive'],
        MixedPositive=counts['native_with_learned']['positive'], CapacityGain=fmt(rate, 3)+' pp',
        MenuSwitches=f'{am["risk"]["mixed_joint"]["switches"]:,}',
        MenuObserved=fmt(control['known_pp'], 3)+' pp',
        MenuLower=fmt(control['completion_lower_pp'], 3)+' pp',
        MenuPair=fmt(menu['primary_pair_geometry']['observed_ade_reduction'], 3),
        CommonScreenSwitches=am['risk']['mixed_common_screen']['switches'],
        DirectVsCV=fmt(inc['direct_difference_vs_cv_difference']['completion_lower_pp'], 3)+' pp',
        DirectKnown=fmt(inc['direct_difference_vs_cv_difference']['known_pp'], 3)+' pp',
        DirectMissing=f'{inc["direct_difference_vs_cv_difference"]["known_pp"]["point"]-inc["direct_difference_vs_cv_difference"]["completion_lower_pp"]["point"]:.3f}',
        DirectVsAnchor=fmt(inc['direct_difference_vs_anchor']['completion_lower_pp'], 3)+' pp',
        DirectPass=im['risk']['direct_difference']['supported_views'],
        DifferencePass=im['risk']['cv_difference']['supported_views'],
        AnchorPass=im['risk']['anchor']['supported_views'],
        DirectScreenSwitches=im['risk']['direct_difference_screen']['switches'],
        DirectScreenPass=im['risk']['direct_difference_screen']['supported_views'],
        DirectPair=fmt(increment['relative_pair']['direct_vs_anchor']['observed_ade_reduction'], 3))
    return '\n'.join("\\newcommand{\\"+k+"}{"+str(v)+"}" for k, v in fields.items())+'\n'


def build(out, style):
    result = prior.build(out, style)
    bindings = result['source_bindings']
    measures = [verified_analysis(k, bindings) for k in ('native_neural_bank_ablation_v1',
                'native_causal_menu_control_v1', 'native_direct_increment_v1')]
    block = PUBLIC/'menu_evidence.tex'
    shutil.copyfile(block, out/block.name)
    bindings[f'evidence_manuscript_v13/{block.name}'] = prior.sha(block)
    tex = out/'manuscript.tex'
    document = prior.replace_once(tex.read_text(), r'\begin{document}', r'\input{menu_values.tex}'+'\n')
    old_abstract = ('These findings\nidentify where candidate capacity is lost. They do not establish a\n'
                    'deployable improvement or independent generalization.')
    if document.count(old_abstract) != 1:
        raise ValueError('Expected the v12 abstract ending')
    document = document.replace(old_abstract, (
        'At identical recipients,\nlearning the neural increment directly worsens conservative utility\n'
        'relative to a CV-difference critic. These experiments identify\n'
        'decision bottlenecks, not deployable improvement or independent\n'
        'generalization.'))
    document = prior.replace_once(document, r'\section{Limitations and Independent Evidence}',
                                  r'\input{menu_evidence.tex}'+'\n\n')
    tex.write_text(document)
    (out/'menu_values.tex').write_text(values(*measures))
    for name in ('manuscript.tex', block.name, 'menu_values.tex'):
        result['build_bindings'][name] = prior.sha(out/name)
    (out/'source_bindings.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


def package(out, path, bindings):
    files = {name: (out/name).read_bytes() for name in bindings if name != 'cvpr.sty'}
    if any(prior.sha(out/name) != bindings[name] for name in files):
        raise ValueError('Cannot package modified draft sources')
    files['README.txt'] = b'Anonymous working draft. Supply cvpr.sty from the CVPR 2027 template and compile manuscript.tex.\n'
    with ZipFile(path, 'w') as z:
        for name, contents in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, contents, compress_type=ZIP_DEFLATED)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--style', type=Path, default=prior.DATA/'evidence_manuscript_v10/paper_build/cvpr.sty')
    p.add_argument('--source-zip', type=Path)
    args = p.parse_args()
    result = build(args.output_dir, args.style)
    if args.source_zip:
        package(args.output_dir, args.source_zip, result['build_bindings'])
