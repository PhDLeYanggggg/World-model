"""Re-typeset the verified supplement as a monochrome academic appendix."""

from io import BytesIO
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import FloatObject
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_reader_supplement as reader
from scripts import build_m3w_wepdtof_penalty_candidate as corrected
from scripts import build_m3w_integrated_reader_candidate as integrated

HANDOFF = reader.ADMIN/'local_handoff_v218_serif'
PUBLIC = reader.PUBLIC/'serif_supplement_v218'
LEFT, WIDTH = 56, 500
BLACK = colors.black
SECTIONS = {
    'Reading the replacement experiments': 'S1',
    'Capacity, ranking and label support': 'S2',
    'Frozen comparisons and actual releases': 'S3',
    'From a history query to a scored decision': 'S5',
    'Allocation at the same spend': 'S6',
    'Neighbor input and label timing': 'S7',
    'WEPDTOF: movement and pair penalty': 'S9',
}


def para(pdf, text, y, *, size=11.3, lead=15.2, bold=False, x=LEFT, width=WIDTH, space=8, **_):
    text = text.replace('The final page shows', 'Section S4 shows')
    p = Paragraph(text, ParagraphStyle('body', fontName='Times-Bold' if bold else 'Times-Roman',
        fontSize=size, leading=lead, textColor=BLACK))
    _, h = p.wrap(width, 700)
    p.drawOn(pdf, x, y-h)
    return y-h-space


def header(pdf, _, title):
    y = 742
    if SECTIONS[title] == 'S1':
        y = para(pdf, 'Supplementary Material', y, size=14, lead=17, bold=True, space=13)
    return para(pdf, f'{SECTIONS[title]}. {title}', y, size=16.3, lead=20, bold=True, space=13)


def table(pdf, data, widths, y):
    def numeric(s):
        return bool(re.match(r'^[+\-\[0-9]', str(s)))

    right = [j for j in range(1, len(data[0])) if all(numeric(r[j]) for r in data[1:])]
    styles = {align: ParagraphStyle('cell', fontName='Times-Roman', fontSize=10.4,
                                    leading=12.3, alignment=align) for align in (TA_LEFT, TA_RIGHT)}
    rows = [[Paragraph(f'<b>{v}</b>' if i == 0 else str(v), styles[TA_RIGHT if j in right else TA_LEFT])
             for j, v in enumerate(row)] for i, row in enumerate(data)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.8), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.8),
        ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('LINEABOVE', (0, 0), (-1, 0), .75, BLACK), ('LINEBELOW', (0, 0), (-1, 0), .40, BLACK),
        ('LINEBELOW', (0, -1), (-1, -1), .75, BLACK)]))
    _, h = t.wrap(WIDTH, 700)
    t.drawOn(pdf, LEFT, y-h)
    return y-h-11


def finish(pdf, page, y):
    if y < 58:
        raise ValueError(f'Supplement page {page} would collide with its footer')
    pdf.setFillColor(BLACK)
    pdf.setFont('Times-Roman', 10)
    pdf.drawCentredString(306, 33, str(page))
    pdf.showPage()


def figure_page(original, *, title, caption, page_no, top_crop, placed_top):
    # Preserve every plotted point and line; change only color and the page framing.
    source = PdfWriter()
    fig = source.add_page(original)
    contents = fig.get_contents()
    ops = []
    for v, op in contents.operations:
        if op in (b'rg', b'RG'):
            r, g, b = map(float, v)
            gray = .2126*r+.7152*g+.0722*b
            if .13 < gray < .68:
                if b > r+.08:
                    gray = .23
                elif r > b+.15:
                    gray = .57
            ops.append(([FloatObject(gray)], b'g' if op == b'rg' else b'G'))
        else:
            ops.append((v, op))
    contents.operations = ops
    fig.replace_contents(contents)
    fig.cropbox.top = top_crop

    base = BytesIO()
    pdf = canvas.Canvas(base, pagesize=(612, 792), invariant=1)
    y = para(pdf, title, 742, size=16.3, lead=20, bold=True, space=12)
    para(pdf, caption, y)
    finish(pdf, page_no, 65)
    pdf.save()
    page = PdfReader(BytesIO(base.getvalue())).pages[0]
    factor = min(500/float(fig.mediabox.width), 1.)
    page.merge_transformed_page(fig, Transformation().scale(factor).translate(
        (612-float(fig.mediabox.width)*factor)/2, placed_top-float(top_crop)*factor))
    return page


def read_pdf(raw):
    return PdfReader(BytesIO(raw))


def run():
    current = json.loads((corrected.PUBLIC/'review_receipt.json').read_bytes())
    old_supplement = reader.checked(corrected.HANDOFF/'supplement.pdf', current['reader_pdf_sha256'])
    paper = reader.checked(corrected.HANDOFF/'paper.pdf', current['companion_pdf_sha256'])
    old_replay = reader.checked(corrected.HANDOFF/'replay.zip', current['archive_sha256'])
    d, inputs = reader.load()
    for k, name in (('easy', 'songdo_full_grid_easy_risk_v1'), ('wepdtof_grid', 'wepdtof_full_grid_risk_v1'),
                    ('opendd_exchange', 'opendd_full_grid_exchange_v1')):
        raw = (reader.PUBLIC/name/'readout.json').read_bytes()
        inputs[k] = reader.digest(raw)
        item = json.loads(raw)
        d[k] = item['summary'] if k == 'opendd_exchange' else item
    case = reader.checked(reader.LOCAL/'songdo_failure_cases_v1/failure_cases.pdf', d['cases']['figure_sha256'])
    v216 = json.loads((integrated.PUBLIC/'review_receipt.json').read_bytes())
    with ZipFile(BytesIO(reader.checked(integrated.HANDOFF/'replay.zip', v216['archive_sha256']))) as z:
        values = json.loads(z.read('reader_values.json'))
    facts, hashes = corrected.inputs()
    reader.layout = SimpleNamespace(header=header, para=para, table=table)
    reader.finish = finish
    original = read_pdf(old_supplement)
    old_text_pages = read_pdf(reader.render(d, case)).pages
    middle = read_pdf(integrated.extra_pages(values)).pages
    first = read_pdf(corrected.overview()).pages[0]
    p4 = figure_page(original.pages[3], title='S4. Three Songdo cases',
        caption='I and J indicate the independent and joint edit choices. Grey in the support grid '
                'marks a missing target; the band on the right shows missing-label bounds.',
        page_no=4, top_crop=float(original.pages[3].mediabox.height)*.925, placed_top=651)
    p8 = figure_page(original.pages[7], title='S8. Label timing at equal coverage',
        caption='OpenDD development, 3,711 originally complete queries. Points and intervals '
                'compare masks with the same retained pair-step count.',
        page_no=8, top_crop=680, placed_top=661)
    v216_supplement = reader.checked(integrated.HANDOFF/'supplement.pdf', v216['reader_pdf_sha256'])
    end = read_pdf(corrected.append(v216_supplement, facts)).pages[-1]
    writer = PdfWriter()
    for p in [first, *old_text_pages[1:3], p4, *middle, p8, end]:
        if p.get('/Annots'):
            raise ValueError('Appendix contains an unexpected link or annotation')
        writer.add_page(p)
    writer.add_metadata({'/Title': 'Supplementary Material', '/Author': 'Anonymous'})
    out = BytesIO()
    writer.write(out)
    pdf = out.getvalue()
    rebuilt = read_pdf(pdf)
    text = ' '.join(' '.join(p.extract_text().split()) for p in rebuilt.pages)
    if (len(rebuilt.pages) != 9 or reader.PRIVATE.search(text)
            or any(x not in text for x in ('49.29', '0.11013', '36 edited views', 'Label timing at equal coverage', '0/45'))):
        raise ValueError('Restyled appendix lost an expected result')

    with ZipFile(BytesIO(old_replay)) as z:
        parts = {n: z.read(n) for n in z.namelist() if n != 'manifest.json'}
    parts['supplement.pdf'] = pdf
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    replay = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='m3w-serif-') as tmp:
        path = Path(tmp)/'replay.zip'
        path.write_bytes(replay)
        reader.unpack(path, Path(tmp)/'extracted')
        result = reader.run_replay(Path(tmp)/'extracted')['wepdtof_penalty_breakpoints']
        if result != current['replay']:
            raise ValueError('Typography changed the scientific replay')
    files = {'paper.pdf': paper, 'supplement.pdf': pdf, 'replay.zip': replay}
    manifest = {k: reader.digest(v) for k, v in files.items()}
    result = dict(files=manifest, pages=9, source_pdf_sha256=current['reader_pdf_sha256'],
                  figure_geometry_unchanged=True, primary_pdf_unchanged=True, llm_statement_unchanged=True,
                  reviewer_replay=result, input_hashes=inputs | hashes, uploaded=False)
    targets = {**{HANDOFF/k: v for k, v in files.items()}, HANDOFF/'manifest.json': reader.as_json(manifest),
               PUBLIC/'receipt.json': reader.as_json(result)}
    if any(k.exists() and k.read_bytes() != v for k, v in targets.items()):
        raise FileExistsError('Refusing to replace a different serif handoff')
    for k, v in targets.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(result)


if __name__ == '__main__':
    run()
