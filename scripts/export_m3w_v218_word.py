"""Export the verified v218 LaTeX source as a local, editable Word draft."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'data/stage_cvpr2027_experiments'
SOURCE = 'evidence_manuscript_v218_wepdtof_penalty'
RECEIPT = ROOT/'outputs/publication_readiness_2026_09'/SOURCE/'receipt.json'
OUTPUT = STAGE/'submission_admin/local_handoff_v218_color_figures/paper_editable.docx'


def prepare(files):
    tex = files['manuscript.tex'].decode('ascii')
    header, body = tex.split(r'\begin{document}', 1)
    definitions = '\n'.join([header, *(files[p].decode('ascii') for p in
        re.findall(r'\\input\{([^}]+)\}', header))])
    macros = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}\n]+)\}', definitions))
    body = re.sub(r'\\([A-Za-z]+)(?:\{\})?', lambda m: macros.get(m[1], m[0]), body)
    main, bib = re.split(r'\{\\small\s*\\begin\{thebibliography\}\{99\}', body)
    bib = bib.split(r'\end{thebibliography}', 1)[0]
    entries = re.split(r'\\bibitem\{([^}]+)\}\s*', bib)[1:]
    citations = {key: i for i, key in enumerate(entries[::2], 1)}
    main = re.sub(r'\\cite\{([^}]+)\}', lambda m: '[' +
                  ', '.join(str(i) for i in sorted(citations[k] for k in m[1].split(','))) + ']', main)
    refs = {}
    for kind, env in (('Figure', r'figure\*?'), ('Table', 'table'), ('Equation', 'equation')):
        pattern = r'\\begin\{'+env+r'\}(.*?)\\end\{'+env+r'\}'

        def number(m, counter=iter(range(1, 100))):
            n, block = next(counter), m[1]
            for key in re.findall(r'\\label\{([^}]+)\}', block):
                refs[key] = n
            block = re.sub(r'\\label\{[^}]+\}', '', block)
            if kind == 'Equation':
                block = block.rstrip() + rf'\qquad ({n})' + '\n'
            else:
                block = block.replace(r'\caption{', rf'\caption{{{kind} {n}. ', 1)
            return m[0][:m.start(1)-m.start()] + block + m[0][m.end(1)-m.start():]

        main = re.sub(pattern, number, main, flags=re.S)
    main = re.sub(r'\\ref\{([^}]+)\}', lambda m: str(refs[m[1]]), main)
    main = re.sub(r'\\(?:begin|end)\{samepage\}', '', main)
    main = re.sub(r'\\(?:centering|footnotesize|small)\b|\\setlength\{\\tabcolsep\}\{[^}]*\}', '', main)
    main = main.replace('figure*}', 'figure}')
    main = re.sub(r'\\includegraphics\[[^\]]+\]\{([\w_-]+)\.pdf\}',
                  r'\\includegraphics[width=6.55in]{\1.png}', main)
    references = '\n\n'.join(f'[{i}] {entry.strip()}' for i, entry in enumerate(entries[1::2], 1))
    front = '\n'.join(line for line in header.splitlines() if line.startswith((r'\title{', r'\author{')))
    return front + '\n' + r'\begin{document}' + main + '\n' + r'\section*{References}' + '\n' + references + '\n' + r'\end{document}', len(citations)


def layout(raw, output, reference_count):
    doc = Document(raw)
    doc.core_properties.author = 'Anonymous'
    doc.core_properties.last_modified_by = ''
    for style in doc.styles:
        if style.type in (WD_STYLE_TYPE.PARAGRAPH, WD_STYLE_TYPE.CHARACTER):
            style.font.name = 'Times New Roman'
            style.font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles['Normal']
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.13
    normal.paragraph_format.space_after = Pt(6)
    doc.styles['Title'].font.size = Pt(21)
    doc.styles['Title'].font.bold = True
    doc.styles['Title'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i in range(1, 10):
        style = doc.styles[f'Heading {i}']
        style.font.size = Pt(14 if i == 1 else 11.5)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(10 if i == 1 else 7)
        style.paragraph_format.space_after = Pt(4)
    for name in ('Image Caption', 'Table Caption', 'Caption'):
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = normal
        style.font.size = Pt(10)
        style.font.italic = False
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.left_margin = section.right_margin = Inches(.85)
        section.top_margin = section.bottom_margin = Inches(.75)
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement('w:fldSimple')
        field.set(qn('w:instr'), 'PAGE')
        p._p.append(field)
    for table in doc.tables:
        table.style = None
        borders = OxmlElement('w:tblBorders')
        for name in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            edge = OxmlElement('w:'+name)
            edge.set(qn('w:val'), 'single')
            edge.set(qn('w:color'), 'B9B9B9')
            edge.set(qn('w:sz'), '4')
            borders.append(edge)
        table._tbl.tblPr.append(borders)
        widths = [1.65]+[5.15/6]*6 if len(table.columns) == 7 else [2.55, 2.05, 2.2]
        table.autofit = False
        for col, width in zip(table.columns, widths):
            col.width = Inches(width)
        head = table.rows[0]._tr.get_or_add_trPr()
        if head.find(qn('w:tblHeader')) is None:
            head.append(OxmlElement('w:tblHeader'))
        for idx, row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(4)
                    p.paragraph_format.space_after = Pt(4)
                    p.paragraph_format.keep_with_next = idx < len(table.rows)-1
                    for run in p.runs:
                        run.bold = idx == 0
                        run.font.size = Pt(10)
    paragraphs = doc.paragraphs
    for idx, p in enumerate(paragraphs):
        if p._p.xpath('.//w:drawing') and idx+1 < len(paragraphs):
            p.paragraph_format.keep_with_next = True
        if p.style.name == 'Table Caption' and p.text.startswith('Table '):
            p.paragraph_format.keep_with_next = True
    display = doc.element.xpath('.//m:oMathPara')
    if (len(doc.inline_shapes), len(doc.tables), len(display)) != (3, 3, 3):
        raise ValueError(f'Expected 3 figures, 3 tables and 3 display equations; got '
                         f'{len(doc.inline_shapes)}, {len(doc.tables)}, {len(display)}')
    refs_index = next(i for i, p in enumerate(paragraphs) if p.text == 'References')
    if len([p for p in paragraphs[refs_index+1:] if re.match(r'^\[\d+\]', p.text)]) != reference_count:
        raise ValueError('Reference count changed')
    if any(doc.element.xpath('.//w:lnNumType')):
        raise ValueError('Word draft must not have review line numbers')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        doc.save(f)


def main(output):
    expected = json.loads(RECEIPT.read_text())['artifacts']['main_source.zip']
    archive = (STAGE/SOURCE/'main_source.zip').read_bytes()
    if sha256(archive).hexdigest() != expected:
        raise ValueError('Manuscript source hash mismatch')
    with ZipFile(BytesIO(archive)) as z:
        files = {k: z.read(k) for k in z.namelist()}
    tex, reference_count = prepare(files)
    with TemporaryDirectory(prefix='m3w-word-') as tmp:
        temp = Path(tmp)
        (temp/'paper.tex').write_text(tex)
        for name in re.findall(r'\{([\w_-]+)\.png\}', tex):
            pdf = temp/f'{name}.pdf'
            pdf.write_bytes(files[pdf.name])
            subprocess.run(['pdftoppm', '-r', '260', '-singlefile', '-png', pdf, temp/name], check=True)
        raw = temp/'paper.docx'
        subprocess.run(['pandoc', 'paper.tex', '-f', 'latex', '-t', 'docx', '-s',
                        '--fail-if-warnings', '-o', str(raw)], check=True, cwd=temp)
        layout(raw, output, reference_count)
    print(f'{output}: 3 editable tables, 3 display equations, 3 figures, {reference_count} references')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    main(parser.parse_args().output)
