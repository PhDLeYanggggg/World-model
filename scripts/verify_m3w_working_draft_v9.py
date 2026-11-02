"""Source-bound PDF checks for v9; no claim of scientific submission readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/evidence_manuscript_v9'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v9'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(build, compile_log, visual_sha=None):
    source = json.loads((PUBLIC/'build_sources.json').read_text())
    report = json.loads(compile_log.read_text()); figures = json.loads((PUBLIC/'figures/figure_sources.json').read_text())
    pdf,tex = build/'main.pdf',build/'main.tex'
    if (source['tex_sha256'] != sha(tex) or source['tex_sha256'] != sha(PUBLIC/'manuscript.tex')
            or source['markdown_sha256'] != sha(PUBLIC/'manuscript.md')
            or source['template_sha256'] != sha(build/'cvpr.sty')
            or source['figure_sources_sha256'] != sha(PUBLIC/'figures/figure_sources.json')
            or figures['private_PDF_sha256'] != sha(build/'cost_loss.pdf')
            or figures['public_svg_sha256'] != sha(PUBLIC/'figures/cost_loss.svg')):
        raise ValueError('Current source/style/figure identities required')
    for experiment,ref in source['source_bindings'].items():
        home = PUBLIC.parent/experiment
        if ref['summary_sha256'] != sha(home/'summary.json') or ref['verification_sha256'] != sha(home/ref['verification_file']):
            raise ValueError('Linked scalar evidence changed')
    if (not report['pdfExists'] or report['attempts'][-1]['exitCode'] != 0
            or Path(report['pdfPath']).resolve() != pdf.resolve()
            or Path(report['rootFile']).resolve() != tex.resolve()
            or min(pdf.stat().st_mtime,compile_log.stat().st_mtime) < max(tex.stat().st_mtime,(build/'cost_loss.pdf').stat().st_mtime)):
        raise ValueError('Successful compilation of current inputs required')
    log = report['attempts'][-1]['log']; final = log.rsplit('note: Rerunning TeX',1)[-1]
    if any(token in final for token in ('Overfull','Font shape','Missing character','There were undefined','undefined citations')):
        raise ValueError('Unresolved final-pass layout or reference problem')
    main_pages = {int(n) for n in re.findall(r'M3W_MAIN_PAGES=(\d+)',log)}
    if len(main_pages) != 1 or not 1 <= next(iter(main_pages)) <= 8:
        raise ValueError('Stable main-body page count up to eight required')
    reader = PdfReader(pdf); texts = [p.extract_text() or '' for p in reader.pages]
    text = '\n'.join(texts)
    if (any(len(t.strip()) < 80 for t in texts) or
            any(abs(float(p.mediabox.width)-612) > .1 or abs(float(p.mediabox.height)-792) > .1 for p in reader.pages)):
        raise ValueError('Nonempty letter-format pages required')
    if len(re.findall(r'^References\s*\d*\s*$',text,re.M)) != 1 or text.count('LLM Usage Statement') != 1:
        raise ValueError('One reference heading and one assistance disclosure required')
    if text.index('LLM Usage Statement') > text.index('References'):
        raise ValueError('Disclosure must precede references')
    metadata = reader.metadata or {}
    if '/Users/' in text or 'openreview.net/profile' in text or metadata.get('/Author','') not in ('','Anonymous'):
        raise ValueError('Private path/profile or nonanonymous PDF author')
    digest = sha(pdf)
    if visual_sha is not None and visual_sha != digest:
        raise ValueError('Visual review must bind this exact PDF')
    for index in range(1,len(reader.pages)+1):
        p = build/f'page-{index}.png'
        if not p.exists() or p.stat().st_mtime < pdf.stat().st_mtime:
            raise ValueError('Render every current PDF page before recording checks')
    result = dict(result_source='fresh_typesetting_checks_not_scientific_confirmation',
        pdf_sha256=digest,pdf_bytes=pdf.stat().st_size,tex_sha256=sha(tex),main_body_pages=next(iter(main_pages)),
        total_pages=len(reader.pages),compile_exit_code=0,overfull_or_final_reference_issues=0,
        upstream_lineno_UTF8_warning_retained='Invalid UTF-8' in final,
        letter_format=True,PDF_author_anonymous=True,known_path_profile_check_pass=True,
        reviewed_pages=list(range(1,len(reader.pages)+1)) if visual_sha else [],
        review_method='assistant inspection of every page PNG' if visual_sha else 'not_run',
        visual_review_bound_to_pdf_sha256=visual_sha,template_sha256=source['template_sha256'],
        source_bindings=source['source_bindings'],PDF_and_template_private=True,
        scientific_confirmation=False,complete_anonymity_certified=False,submission_ready=False)
    (PUBLIC/'render_verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    source.update(compiled=True,pdf_sha256=digest,render_verification_sha256=sha(PUBLIC/'render_verification.json'))
    (PUBLIC/'build_sources.json').write_text(json.dumps(source,indent=2,sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=PRIVATE/'paper_build')
    parser.add_argument('--compile-log',type=Path,default=PRIVATE/'paper_compile_01.log')
    parser.add_argument('--visual-reviewed-sha')
    args = parser.parse_args()
    print(json.dumps(verify(args.build,args.compile_log,args.visual_reviewed_sha),indent=2))
