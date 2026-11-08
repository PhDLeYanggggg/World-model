from io import BytesIO
from zipfile import ZipFile

from src.data_unification.m3w_public_zip_range import RangeArchive


def test_range_reader_works_with_zip_seek_and_bounded_member_reads(monkeypatch):
    raw = BytesIO()
    with ZipFile(raw,'w') as z:
        z.writestr('first.csv',b'a'*7000)
        z.writestr('tail.csv',b'held')
    blob = raw.getvalue()
    calls = []
    def fetch(_self, a, b):
        calls.append((a,b))
        return blob[a:b+1]

    monkeypatch.setattr(RangeArchive,'_fetch',fetch)
    with RangeArchive('https://example.test',len(blob),block_size=256) as f, ZipFile(f) as z:
        assert z.namelist()==['first.csv','tail.csv']
        assert z.read('tail.csv')==b'held'
        assert len(calls) < 10
