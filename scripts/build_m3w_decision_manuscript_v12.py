"""Assemble the anonymous native decision manuscript from verified scalar evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = HOME/'evidence_manuscript_v12'
DATA = ROOT/'data/stage_cvpr2027_experiments'
DEFAULT = DATA/'evidence_manuscript_v12/build'
PINNED = {
    'moving_visual_risk_v1/analysis.json': '4375e369a675ff2f396e5502a8e3bf43cb33dbce173142cb8ff2e4c8ecd7025f',
    'moving_visual_risk_v1/failure_case.json': '5eb109a4a98f9f3076bfe3b95e94120b0572f66bc81360e90bce04d2a3b47523',
}
REPLAY = ('dronecrowd_transport_candidate_v1', 'moving_joint_path_audit_v1',
          'native_postsearch_risk_v1', 'native_postsearch_mass_v1', 'native_history_radius_v1')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(relative, expected, bindings):
    p = HOME/relative
    actual = sha(p)
    if actual != expected:
        raise ValueError(f'Changed paper evidence: {relative}')
    bindings[relative] = actual
    return p


def read(p):
    return json.loads(p.read_text())


def citation_metric(obj, digits):
    point = f'{obj["point"]:+.{digits}f}'
    lower, upper = (f'{v:+.{digits}f}' for v in obj['interval95'])
    return f'${point}$ [${lower}$, ${upper}$]'


def macros(data):
    visual, case = (data['moving_visual_risk_v1/'+k] for k in ('analysis.json', 'failure_case.json'))
    pair = data['moving_joint_path_audit_v1/analysis.json']['comparisons']['joint_vs_pair_additive']
    risk = data['dronecrowd_transport_candidate_v1/analysis.json']['grouped']['risk']
    post = data['native_postsearch_risk_v1/analysis.json']['grouped']['effects']
    mass = data['native_postsearch_mass_v1/analysis.json']['grouped']['effects']
    if visual['required_record_views_per_policy'] != 72 or case['known_future_steps'] != 12:
        raise ValueError('Visual study scope changed')
    if risk['constant_velocity_causal_fd']['worst_recording_p95_known_ade_pixel'] != risk[
            'gated_joint']['worst_recording_p95_known_ade_pixel']:
        raise ValueError('Draft tail comparison no longer matches')
    table = []
    for label, key in (('No image', 'no_image'), ('Bounds', 'image_bounds'),
                       ('Quality and geometry', 'quality_geometry'), ('Motion and geometry', 'motion_geometry')):
        row = visual['policies'][key]['joint_free']
        table.append(f'{label} & {row["utility"]["known"]["point"]:.3f} & {row["hard_gain_percent"]:.3f} & '
                     f'{row["easy_gain_percent"]:.3f} & {row["safety"]["supported_record_views"]} '+r'\\')
    definitions = dict(
        ContextRows='\n'.join(table),
        ContextContrast=citation_metric(visual['contrasts']['primary']['completion_lower_pp'], 3)+' pp',
        FailureBaseline=f'{case["floor_available_ADE_pixel"]:.3f}',
        FailureCandidate=f'{case["arms"]["motion_geometry"]["available_ADE_pixel"]:.3f}',
        PairObserved=citation_metric(pair['observed_ade_reduction'], 3),
        PairLower=citation_metric(pair['grid_ade_lower'], 3),
        PairUpper=citation_metric(pair['grid_ade_upper'], 3),
        PostRiskContrast=citation_metric(post['learned_vs_matched_margin']['completion_lower_pp'], 6)+' pp',
        PostMassContrast=citation_metric(mass['record_mass_vs_matched_point']['completion_lower_pp'], 6)+' pp',
        GateSwitches=str(risk['gated_joint']['switches']),
        TailCV=f'{risk["constant_velocity_causal_fd"]["worst_recording_p95_known_ade_pixel"]:.3f}',
        TailOLS=f'{risk["history_ols4_velocity"]["worst_recording_p95_known_ade_pixel"]:.3f}')
    return '\n'.join("\\newcommand{\\"+key+"}{"+value+"}" for key, value in definitions.items())+'\n'


def replace_once(text, marker, inserted):
    if text.count(marker) != 1:
        raise ValueError(f'Expected one manuscript insertion marker: {marker}')
    return text.replace(marker, inserted+marker, 1)


def build(out, style):
    bindings = {}
    tex_v11 = source('evidence_manuscript_v11/decision_core.tex',
                     read(HOME/'evidence_manuscript_v11/build_verification.json')['tex_sha256'], bindings)
    evidence = {}
    for relative, digest in PINNED.items():
        evidence[relative] = read(source(relative, digest, bindings))
    for k in REPLAY:
        v = read(HOME/k/'replay_execution.json')
        digest = v['analysis_sha256'] if 'analysis_sha256' in v else v['artifact']['sha256']
        relative = k+'/analysis.json'
        evidence[relative] = read(source(relative, digest, bindings))
    tex = tex_v11.read_text()
    tex = replace_once(tex, r'\section{Matched Decisions and Capacity}', r'\input{protocol.tex}'+'\n\n')
    tex = replace_once(tex, r'\section{Limitations and Independent Evidence}', r'\input{ablations.tex}'+'\n\n')
    tex = replace_once(tex, r'\begin{document}', r'\input{evidence_values.tex}'+'\n')
    marker = r'\maketitle'
    if tex.count(marker) != 1:
        raise ValueError('Expected an anonymous title block')
    tex = tex.replace(marker, marker+'\n'+r'\noindent\textbf{Working evidence manuscript. No independent results are included.}'+'\n')
    if sha(style) != read(HOME/'evidence_manuscript_v11/build_verification.json')['cvpr_style_sha256']:
        raise ValueError('Require the already verified CVPR 2027 style')
    out.mkdir(parents=True, exist_ok=True)
    (out/'manuscript.tex').write_text(tex)
    (out/'evidence_values.tex').write_text(macros(evidence))
    for name in ('protocol.tex', 'ablations.tex'):
        shutil.copyfile(PUBLIC/name, out/name)
    shutil.copyfile(style, out/'cvpr.sty')
    subprocess.run([sys.executable, ROOT/'scripts/plot_m3w_decision_bottlenecks.py', '--output',
                    out/'decision_bottlenecks.pdf'], check=True)
    figure = out/'decision_bottlenecks.pdf'
    expected = read(HOME/'evidence_manuscript_v10/decision_bottlenecks_figure_verification.json')['local_pdf_sha256']
    if sha(figure) != expected:
        raise ValueError('Reproduced figure differs from the checked vector figure')
    result = dict(source_bindings=bindings, build_bindings={n: sha(out/n) for n in (
        'manuscript.tex', 'evidence_values.tex', 'protocol.tex', 'ablations.tex', 'cvpr.sty', 'decision_bottlenecks.pdf')},
        external_score_present=False, submission_ready=False)
    (out/'source_bindings.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


def package(out, target, bindings):
    names = ('ablations.tex', 'decision_bottlenecks.pdf', 'evidence_values.tex',
             'manuscript.tex', 'protocol.tex')
    files = {n: (out/n).read_bytes() for n in names}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != bindings[name]:
            raise ValueError(f'Cannot package modified source: {name}')
    files['README.txt'] = (
        'Anonymous working evidence manuscript. No independent results are included.\n'
        'Provide cvpr.sty from the official CVPR 2027 author kit beside manuscript.tex.\n'
        'Then compile manuscript.tex with Tectonic or the official template toolchain.\n'
        'This package reproduces the draft typesetting, not upstream training or scoring.\n'
    ).encode('ascii')
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as bundle:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            bundle.writestr(info, data, compress_type=ZIP_DEFLATED)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=DEFAULT)
    p.add_argument('--style', type=Path, default=DATA/'evidence_manuscript_v10/paper_build/cvpr.sty')
    p.add_argument('--source-zip', type=Path)
    args = p.parse_args()
    result = build(args.output_dir.resolve(), args.style)
    if args.source_zip:
        package(args.output_dir, args.source_zip, result['build_bindings'])
        result['source_zip_sha256'] = sha(args.source_zip)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
