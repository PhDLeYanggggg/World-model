"""Bind the v211 paper, unchanged reader PDF and expanded anonymous replay."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_exchange_exposition_candidate as paper
from scripts import build_m3w_reader_supplement as reader
from reproducibility.m3w_exchange_exposition.verify import MEMBERS, replay
from reproducibility.m3w_reader_supplement_exchange.verify import quantile

TEMPLATE = reader.TEMPLATE.with_name('m3w_exchange_exposition')


def main():
    d, hashes = paper.inputs()
    published = json.loads((paper.PUBLIC/'receipt.json').read_bytes())
    parent = json.loads((paper.prior.PUBLIC/'review_receipt.json').read_bytes())
    if (published['source_readout_sha256'] != hashes or published['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or parent['readout_sha256']['opendd_exchange'] != hashes['opendd_exchange']):
        raise ValueError('The paper, figure and inherited OpenDD comparison differ')
    pdf = published['artifacts']['manuscript.pdf']
    reader.checked(paper.OUT/'manuscript.pdf', pdf)
    old = reader.checked(reader.ADMIN/'review_supplement_v44.zip', parent['archive_sha256'])
    supplement = reader.checked(reader.ADMIN/'review_supplement_v44.pdf', parent['pdf_sha256'])
    data = dict(counts=d['summary']['counts'], arms=d['summary']['arms'],
                contrasts=d['summary']['full_grid_contrasts'], sites=[[
                    dict(counts=s['counts'], arms=s['arms']) for s in group] for group in d['by_location'].values()],
                draws=np.random.default_rng(817).integers(7, size=(3000, 7)).tolist())
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, 'listwise.json': reader.as_json(data), 'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['listwise.json'].decode()):
        raise ValueError('Unexpected archive members or identifying fields')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v211-review-replay-') as tmp:
        p = Path(tmp)/'replay.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'replay')
        if reader.run_replay(Path(tmp)/'replay')['opendd_listwise_full_grid'] != replay(data, quantile):
            raise ValueError('Anonymous replay differs from the new table')
    result = dict(companion_pdf_sha256=pdf, reader_pdf_sha256=parent['pdf_sha256'], reader_pdf_reused='v44',
                  archive_sha256=reader.digest(payload), archive_bytes=len(payload), prior_archive_sha256=parent['archive_sha256'],
                  source_readout_sha256=hashes, standard_library_replay=True, includes_raw_labels=False,
                  includes_checkpoints=False, original_2pct=True, uploaded=False, submission_ready=False)
    for p, raw in ((reader.ADMIN/'review_supplement_v45.zip', payload),
                   (paper.PUBLIC/'review_receipt.json', reader.as_json(result))):
        if p.exists() and p.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(raw)
    print(result)


if __name__ == '__main__':
    main()
