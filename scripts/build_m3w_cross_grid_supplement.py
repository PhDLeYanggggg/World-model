"""Pair the cross-dataset supplement with its exact main PDF."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_reader_supplement as reader
from scripts import build_m3w_full_grid_candidate as paper
from scripts import build_m3w_full_grid_exchange_candidate as exchange_paper
from reproducibility.m3w_reader_supplement_cross_grid.verify import MEMBERS, check_wep
from reproducibility.m3w_reader_supplement_exchange import verify as exchange

TEMPLATE = reader.TEMPLATE.with_name('m3w_reader_supplement_cross_grid')


def main(with_exchange=False):
    d, _ = reader.load()
    audits, inputs = paper.inputs()
    main = json.loads((paper.PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((reader.OUT.with_name('reader_supplement_v2')/'receipt.json').read_bytes())
    if (main['source_readout_sha256'] != inputs or main['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or parent['input_sha256']['easy'] != inputs['songdo']):
        raise ValueError('Draft and risk readouts do not match')
    reader.checked(paper.OUT/'manuscript.pdf', main['artifacts']['manuscript.pdf'])
    version, template, members, dest = 43, TEMPLATE, MEMBERS, paper.PUBLIC
    replay_key, result = 'wepdtof_full_grid', check_wep(audits['wepdtof'])
    filename, evidence = 'wepdtof_risk.json', audits['wepdtof']
    if with_exchange:
        data, inputs = exchange_paper.inputs()
        desc = json.loads((exchange_paper.PUBLIC/'receipt.json').read_bytes())
        parent = json.loads((paper.PUBLIC/'review_receipt.json').read_bytes())
        if (desc['source_readout_sha256'] != inputs or desc['prior_manuscript_sha256'] != main['artifacts']['manuscript.pdf']
                or parent['companion_pdf_sha256'] != main['artifacts']['manuscript.pdf']):
            raise ValueError('The main and supplemental exchange evidence differ')
        main = desc
        reader.checked(exchange_paper.OUT/'manuscript.pdf', main['artifacts']['manuscript.pdf'])
        version, template, members, dest = 44, TEMPLATE.with_name('m3w_reader_supplement_exchange'), exchange.MEMBERS, exchange_paper.PUBLIC
        d['opendd_exchange'] = data['summary']
        filename, evidence = 'exchange.json', dict(summary=data['summary'], sites=[[
            dict(views=r['counts']['views'], outcomes=r['outcomes']) for r in group]
            for group in data['by_location'].values()],
            draws=np.random.default_rng(817).integers(7, size=(3000, 7)).tolist())
        replay_key, result = 'opendd_full_grid_exchange', exchange.replay(evidence)
    old = reader.checked(reader.ADMIN/f'review_supplement_v{version-1}.zip', parent['archive_sha256'])
    case = reader.checked(reader.LOCAL/'songdo_failure_cases_v1/failure_cases.pdf', d['cases']['figure_sha256'])
    d.update(easy=audits['songdo'], wepdtof_grid=audits['wepdtof'])
    pdf = reader.render(d, case)
    document = PdfReader(BytesIO(pdf))
    page = document.pages[2].extract_text()
    if (len(document.pages) != 4 or any(p.get('/Annots') or reader.PRIVATE.search(p.extract_text()) for p in document.pages)
            or reader.PRIVATE.search(str(document.metadata))
            or any(s not in page for s in ('4 of 36', '3 of those views', '20 of all 36', '229', '144'))
            or (with_exchange and any(s not in document.pages[1].extract_text() for s in ('175', '46.73', 'OpenDD one exchange')))):
        raise ValueError('Unexpected supplement text, annotations or identity markers')
    parts = {k: (template/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': pdf, filename: reader.as_json(evidence), 'prior_evidence.zip': old})
    if set(parts) != members or reader.PRIVATE.search(parts[filename].decode()):
        raise ValueError('Unexpected archive members or identity markers')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='cross-grid-reader-') as tmp:
        temp = Path(tmp)
        (temp/'review.zip').write_bytes(payload)
        reader.unpack(temp/'review.zip', temp/'replay')
        stats = reader.run_replay(temp/'replay')
        if stats[replay_key] != result:
            raise ValueError('Anonymous replay does not reproduce the reported counts')
    receipt = dict(companion_pdf_sha256=main['artifacts']['manuscript.pdf'], pdf_sha256=reader.digest(pdf), pages=4,
                   archive_sha256=reader.digest(payload), prior_archive_sha256=parent['archive_sha256'],
                   archive_bytes=len(payload), readout_sha256=inputs, original_2pct=True, uploaded=False,
                   submission_ready=False, includes_raw_labels=False, includes_checkpoints=False)
    for path, raw in ((reader.ADMIN/f'review_supplement_v{version}.pdf', pdf), (reader.ADMIN/f'review_supplement_v{version}.zip', payload),
                      (dest/'review_receipt.json', reader.as_json(receipt))):
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen supplement')
        if not path.exists():
            with path.open('xb') as f:
                f.write(raw)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exchange', action='store_true', help='Build v44 paired with the OpenDD exchange manuscript')
    main(parser.parse_args().exchange)
