from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts.audit_m3w_visdrone_mot_train import parse, support
from src.evaluation.m3w_remote_zip import RangeFile


class Response:
    history = []
    url = 'https://test.hf.co/object'

    def __init__(self, raw, bounds, status):
        self.raw = BytesIO(raw)
        self.status_code = status
        self.headers = {'Content-Range': bounds}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.raw.close()


class Session:
    def __init__(self, blob, status=206):
        self.blob, self.status = blob, status

    def get(self, url, *, headers, stream, timeout):
        low, high = map(int, headers['Range'].split('=')[1].split('-'))
        return Response(self.blob[low:high+1], f'bytes {low}-{high}/{len(self.blob)}', self.status)


def test_zipfile_reads_sparse_ranges_and_reuses_cached_bytes():
    b = BytesIO()
    with ZipFile(b, 'w') as z:
        z.writestr('data.txt', 'first')
        z.writestr('not-read.jpg', b'x'*50_000)
    data = b.getvalue()
    reader = RangeFile('https://huggingface.co/data', len(data), budget=5000, session=Session(data))
    with ZipFile(reader) as z:
        assert z.read('data.txt') == b'first'
        reads = reader.requests
        assert z.read('data.txt') == b'first'
        assert reader.requests == reads
        assert reader.transferred < 500


def test_range_refuses_full_download_or_over_budget():
    bad = RangeFile('https://huggingface.co/data', 30, budget=10, session=Session(b'x'*30, 200))
    with pytest.raises(OSError, match='exact bounded'):
        bad.span(0, 4)
    with pytest.raises(ValueError, match='byte budget'):
        bad.span(0, 11)


def test_label_support_distinguishes_missing_and_occluded_futures():
    rows = []
    for f in range(1, 21):
        rows.append(f'{f},1,3,4,10,10,1,1,0,{int(f==20)}')
        rows.append(f'{f},4,3,4,10,10,1,2,0,0')
        if f != 19:
            rows.append(f'{f},2,3,4,10,10,1,1,0,0')
    frames = list(range(1, 21))
    labels = parse(('\n'.join(rows)+'\n').encode(), frames)
    assert support(labels, frames, 1) == dict(windows=1, history_instances=2, complete=1, partial=1,
        no_future=0, multi_complete_windows=0, complete_untruncated_unoccluded=0)


def test_included_duplicate_pedestrians_and_out_of_roster_frames_refused():
    row = b'1,1,3,4,10,10,1,1,0,0\n'
    with pytest.raises(ValueError, match='duplicate'):
        parse(row*2, [1])
    with pytest.raises(ValueError, match='absent'):
        parse(row, [2])
