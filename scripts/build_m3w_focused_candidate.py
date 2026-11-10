"""Build a shorter local paper from the verified v171 assets and scalar receipts."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_completion_candidate as prior

scope = prior.scope
main, supplement = scope.main, scope.supplement
DESTINATION = prior.DESTINATION.parent/'evidence_manuscript_v172_focused'
RECEIPT = main.PUBLIC/'evidence_manuscript_v172_focused/receipt.json'
MENU_DESTINATION = DESTINATION.parent/'evidence_manuscript_v173_songdo_menu'
MENU_BUILD_RECEIPT = main.PUBLIC/'evidence_manuscript_v173_songdo_menu/receipt.json'
MENU_EVIDENCE = main.PUBLIC/'songdo_frozen_menu_capacity_v1/receipt.json'
BODY_START = r'\section{Forecast Replacement}'
BODY_END = r'\section*{LLM Usage Statement}'
BIB_START = '{\\small\n\\begin{thebibliography}'


def substitute_body(original, body):
    if original.count(BODY_START) != 1 or original.count(BODY_END) != 1:
        raise ValueError('Expected the two verified v171 boundaries')
    if not body.startswith(r'\section{Introduction}') or re.search(
            r'\\(?:input|include\b|begin\{abstract\}|end\{document\}|title\b|author\b|bibitem\b)', body):
        raise ValueError('Unexpected command outside the body revision scope')
    a, b = original.index(BODY_START), original.index(BODY_END)
    if a >= b:
        raise ValueError('Unexpected body and disclosure order')
    return original[:a]+body.rstrip()+'\n\n'+original[b:]


def verified_values():
    songdo = prior.parent.verified_evidence()
    moving = supplement.read_receipt('moving_completion_envelope_v1')
    wep = supplement.read_receipt('wepdtof_missing_label_interval_v1')
    readout = songdo['matched_joint_minus']['independent']['lower_pp']
    independent, joint = (songdo['released'][a]['risk'] for a in ('independent', 'joint'))
    if (not songdo['no_fitting_on_confirmation'] or not songdo['original_2pct_unchanged']
            or (songdo['calibration_flights'], songdo['flights'], songdo['source_components'],
                songdo['flight_seed_views']) != (40, 160, 14, 480)
            or moving['views'] != 72 or moving['independent_confirmation'] or moving['external_outcomes_read']
            or not wep['original_2pct_unchanged'] or wep['original_fixed_joint_dual_passes'] != 0):
        raise ValueError('Unexpected role, risk rule or evaluation roster')
    levels = moving['completion_envelope']['record']
    lo, hi = wep['comparisons']['fixed_independent']['identification_envelope_pp']
    ci = readout['ci95']
    def available(model, metric):
        return f"{songdo['matched'][model]['metrics'][f'available_{metric}_native']['point']:.6f}"

    return dict(CompletionObserved=levels['observed_steps_only']['excluded_views'],
                CompletionFavorable=levels['favorable_completion']['excluded_views'],
                WepCompletionLower=f'{lo:+.5f}', WepCompletionUpper=f'{hi:+.5f}',
                SongdoPoint=f'{readout["point"]:+.6f}', SongdoLow=f'{ci[0]:+.6f}', SongdoHigh=f'{ci[1]:+.6f}',
                SongdoJointReleased=joint['edited'], SongdoIndependentReleased=independent['edited'],
                SongdoIndependentRisk=independent['original_risk_pass'], SongdoIndependentDual=independent['dual_pass'],
                SongdoCV=available('cv', 'ade'), SongdoOLS=available('ols4', 'ade'),
                SongdoCVFDE=available('cv', 'fde'), SongdoOLSFDE=available('ols4', 'fde'))


def add_menu_result(body, report):
    if (report['views'] != 480 or report['source_components'] != 14 or not report['original_2pct_unchanged']
            or report['test_threshold_selected'] or report['outcomes_reopened'] or report['learned_policy']):
        raise ValueError('Expected only the frozen, post-hoc menu readout')
    label = r'\section{Conclusion and Limitations}'
    if body.count(label) != 1:
        raise ValueError('Expected the unchanged conclusion boundary')
    full = report['menu']['all']
    n = full['dual_opportunity']['views']
    value = full['conservative_cv_gain_ceiling_pp']['point']
    extra = report['joint_added_to_other_arms']
    extra_n = extra['additional_dual_opportunity']['views']
    increment = extra['delta_of_conservative_cv_ceiling_pp']['point']
    paragraph = rf'''\paragraph{{How much does the Songdo menu leave unused?}}
A hindsight check chooses among the same four frozen proposals or
keeps CV. At least one proposal meets the original risk and both
utility checks in {n}/480 views. Choosing the best qualifying proposal
could improve the source-averaged conservative CV score by at most
+{value:.6f} pp. Adding joint to the other three arms makes only
{extra_n} additional views qualify and adds +{increment:.8f} pp to
that ceiling. Thus calibration leaves useful options unused, but at
this edit budget even perfect selection has little effect on the
whole flight. These are outcome-aware bounds for the frozen finite
menu, not a fitted policy or another confirmation test.

'''
    return body.replace(label, paragraph+label), (f'{n}/480', f'{value:.6f}', f'{increment:.8f}')


def check_render(raw, original, extra_markers=()):
    def normalize(text):
        text = re.sub(r'(?m)^[ \t]*(?:CVPR(?:[ \t]+CVPR)?|CVPR 2027 Submission\..*|[0-9]{1,2})[ \t]*$', '', text)
        text = re.sub(r'(?m)[ \t]*[0-9]{3}$', '', text)
        return ' '.join(re.sub(r'(?<=[a-z])[-\u00ad][ \t]*\n[ \t]*(?=[a-z])', '', text).split())

    pages = len(PdfReader(BytesIO(raw)).pages)
    if not 4 <= pages <= 8:
        raise ValueError(f'Unexpected main draft length: {pages} pages')
    text = normalize(supplement.check_pdf(raw, pages))
    before = normalize(supplement.check_pdf(original, len(PdfReader(BytesIO(original)).pages)))
    statement = lambda t: t.split('LLM Usage Statement', 1)[1].split('References', 1)[0].strip()
    if statement(text) != statement(before):
        raise ValueError(f'Rendered disclosure changed: {statement(text)!r} != {statement(before)!r}')
    markers = ('Anonymous CVPR submission', '67/72', '36/72', '9/72', '330/606', '0/45', '0/480',
               '1.373135', '1.603831', '3.416751', '3.808624', '0.000442', '0.01760', *extra_markers)
    if any(x not in text for x in markers):
        raise ValueError(f'Missing rendered results: {[x for x in markers if x not in text]}')
    return pages


def build(destination, compiler, body_path, receipt, *, menu_evidence=None):
    parent = supplement.read_receipt('evidence_manuscript_v171_completion')['artifacts']
    input_files = {k: supplement.checked(prior.DESTINATION/k, sha) for k, sha in parent.items()}
    keep = ('README.txt', 'decision_bottlenecks.pdf', 'capacity_discrimination.pdf', 'direct_count_values.tex',
            'evidence_values.tex', 'menu_values.tex', 'structured_values.tex')
    with ZipFile(BytesIO(input_files['main_source.zip'])) as z:
        if len(set(z.namelist())) != len(z.namelist()) or z.read('manuscript.tex') != input_files['manuscript.tex']:
            raise ValueError('The verified source archive and manuscript must match')
        files = {k: z.read(k) for k in keep}
    body = body_path.read_bytes()
    original = input_files['manuscript.tex'].decode()
    body_text, markers = body.decode('ascii'), ()
    menu_raw = None
    if menu_evidence:
        menu_raw = menu_evidence.read_bytes()
        body_text, markers = add_menu_result(body_text, json.loads(menu_raw))
    revised = substitute_body(original, body_text)
    if revised.count(BIB_START) != 1:
        raise ValueError('Expected one bibliography after the disclosure')
    revised = revised.replace(BODY_END, r'\begin{samepage}'+'\n'+BODY_END, 1).replace(
        BIB_START, r'\end{samepage}'+'\n'+BIB_START, 1)
    definitions = '\n'.join(r'\newcommand{\%s}{%s}' % (k, v) for k, v in verified_values().items())
    files['manuscript.tex'] = revised.replace(r'\begin{document}', definitions+'\n'+r'\begin{document}', 1).encode()
    citations = scope.citation.used_citations(files)
    style = supplement.checked(scope.citation.main.DRAFT/'cvpr.sty', scope.citation.STYLE_SHA)
    with TemporaryDirectory(prefix='focused-candidate-') as tmp:
        root = Path(tmp)
        for name, raw in {**files, 'cvpr.sty': style}.items():
            (root/name).write_bytes(raw)
        p = subprocess.run([str(compiler), '-X', 'compile', '--untrusted', '--only-cached', '--keep-logs',
                            'manuscript.tex'], cwd=root, env={**os.environ, 'SOURCE_DATE_EPOCH': '0'},
                           capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stdout+'\n'+p.stderr)
        if re.search(r'(?i)undefined (citation|reference)|Citation .* undefined|Overfull \\hbox',
                     (root/'manuscript.log').read_text()):
            raise ValueError('Unresolved citation/reference or overfull horizontal box')
        pdf = (root/'manuscript.pdf').read_bytes()
        pages = check_render(pdf, input_files['manuscript.pdf'], markers)
        scope.citation.archive.encode(root/'main_source.zip', files)
        output = {'manuscript.tex': files['manuscript.tex'], 'manuscript.pdf': pdf,
                  'main_source.zip': (root/'main_source.zip').read_bytes()}
    result = dict(pages=pages, prior_manuscript_sha256=parent['manuscript.pdf'], body_sha256=main.digest(body),
                  artifacts={k: main.digest(v) for k, v in output.items()}, bibliography_entries=len(citations),
                  title_abstract_llm_statement_and_references_preserved=True, original_risk_rule_unchanged=True,
                  result_source='fresh_manuscript_build_cached_verified_evidence', new_outcomes_read=False,
                  uploaded=False, submission_ready=False)
    if menu_raw is not None:
        result['songdo_frozen_menu_receipt_sha256'] = main.digest(menu_raw)
    deliver = {**{destination/k: v for k, v in output.items()}, receipt: main.as_json(result)}
    if any(k.exists() and k.read_bytes() != v for k, v in deliver.items()):
        raise FileExistsError('Refusing to replace a different candidate artifact')
    for k, v in deliver.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--body', type=Path, default=DESTINATION/'focused_body.tex')
    p.add_argument('--output', type=Path)
    p.add_argument('--receipt', type=Path)
    p.add_argument('--songdo-menu-capacity', action='store_true')
    a = p.parse_args()
    output = a.output or (MENU_DESTINATION if a.songdo_menu_capacity else DESTINATION)
    receipt = a.receipt or (MENU_BUILD_RECEIPT if a.songdo_menu_capacity else RECEIPT)
    build(output, a.compiler.resolve(), a.body, receipt,
          menu_evidence=MENU_EVIDENCE if a.songdo_menu_capacity else None)
