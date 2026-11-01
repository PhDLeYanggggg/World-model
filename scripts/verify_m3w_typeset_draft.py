"""Record source-bound compilation/layout checks, not scientific readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/evidence_manuscript_v8'
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/european_own_decision_cost_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(build, compile_log, visual_sha=None):
    source = json.loads((PUBLIC / 'build_sources.json').read_text())
    tex, pdf = build / 'main.tex', build / 'main.pdf'
    report = json.loads(compile_log.read_text())
    if (source['tex_sha256'] != sha(tex)
            or source['tex_sha256'] != sha(PUBLIC / 'manuscript.tex')
            or source['markdown_sha256'] != sha(PUBLIC / 'manuscript.md')
            or source['official_template_sha256'] != sha(build / 'cvpr.sty')):
        raise ValueError('Manuscript or official style changed after source generation')
    if (not report['pdfExists'] or not report['attempts']
            or report['attempts'][-1]['exitCode'] != 0
            or Path(report['pdfPath']).resolve() != pdf.resolve()
            or Path(report['rootFile']).resolve() != tex.resolve()
            or min(pdf.stat().st_mtime, compile_log.stat().st_mtime) < tex.stat().st_mtime):
        raise ValueError('Current-source successful compilation required')
    log = report['attempts'][-1]['log']
    final_pass = log.rsplit('note: Rerunning TeX', 1)[-1]
    prohibited = ('Overfull', 'Font shape', 'Missing character',
                  'undefined citations', 'There were undefined references')
    if any(x in final_pass for x in prohibited):
        raise ValueError('Unresolved final-pass typesetting issue')
    main_pages = {int(x) for x in re.findall(r'M3W_MAIN_PAGES=(\d+)', log)}
    if len(main_pages) != 1 or not 1 <= next(iter(main_pages)) <= 8:
        raise ValueError('Main-body page count missing, unstable or above eight')
    reader = PdfReader(pdf)
    texts = [p.extract_text() or '' for p in reader.pages]
    text = '\n'.join(texts)
    if (any(len(t.strip()) < 80 for t in texts)
            or any(abs(float(p.mediabox.width) - 612) > .1
                   or abs(float(p.mediabox.height) - 792) > .1 for p in reader.pages)):
        raise ValueError('Nonempty letter-format pages required')
    if len(re.findall(r'^References\s*\d*\s*$', text, re.M)) != 1:
        raise ValueError('Exactly one reference heading required')
    if text.count('LLM Usage Statement') != 1 or text.index('LLM Usage Statement') > text.index('References'):
        raise ValueError('LLM statement must precede references')
    if any(x in text for x in ('/Users/', 'PhDLeYang', '@kcl.ac.uk', 'Yangyue')):
        raise ValueError('Known identifying path/account string in PDF')
    if reader.metadata and reader.metadata.get('/Author', '') not in ('', 'Anonymous'):
        raise ValueError('PDF author metadata is not anonymous')
    digest = sha(pdf)
    if visual_sha is not None and visual_sha != digest:
        raise ValueError('Visual review must refer to this exact PDF')
    for page in range(1, len(reader.pages) + 1):
        screenshot = build / f'page-{page}.png'
        if not screenshot.exists() or screenshot.stat().st_mtime < pdf.stat().st_mtime:
            raise ValueError('Render every page of the current PDF before recording review')
    result = dict(result_source='fresh_compile_and_layout_check_only',
                  tex_sha256=sha(tex), markdown_sha256=source['markdown_sha256'],
                  pdf_sha256=digest, pdf_bytes=pdf.stat().st_size,
                  main_body_pages=next(iter(main_pages)), total_pages=len(reader.pages),
                  compile_exit_code=0, final_pass_prohibited_issues=0,
                  reference_headings=1, letter_format=True,
                  known_identity_string_check_pass=True,
                  visual_reviewed_pages=list(range(1, len(reader.pages) + 1)) if visual_sha else [],
                  visual_review_bound_to_pdf_sha256=visual_sha,
                  template_sha256=source['official_template_sha256'],
                  template_and_PDF_stored_private=True,
                  third_party_lineno_UTF8_warning_retained='Invalid UTF-8' in final_pass,
                  scientific_confirmation=False, submission_ready=False,
                  note='Layout checks and a recorded visual review do not certify scientific evidence or full anonymity.')
    (PUBLIC / 'render_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    source.update(compiled=True, pdf_sha256=digest,
                  render_verification_sha256=sha(PUBLIC / 'render_verification.json'))
    (PUBLIC / 'build_sources.json').write_text(json.dumps(source, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=PRIVATE / 'paper_build')
    parser.add_argument('--compile-log', type=Path, default=PRIVATE / 'paper_compile.log')
    parser.add_argument('--visual-reviewed-sha', help='Only after inspecting every rendered page of this exact PDF')
    args = parser.parse_args()
    print(json.dumps(verify(args.build, args.compile_log, args.visual_reviewed_sha), indent=2))
