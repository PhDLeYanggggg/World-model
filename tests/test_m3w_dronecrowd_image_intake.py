from collections import OrderedDict
import io
import zipfile

import pytest

from src.evaluation.m3w_dronecrowd_image_intake import (
    DownloadForm, RangeArchive, content_range, image_identity, require_image_permission,
)


def test_provider_form_is_bound_to_archive_and_destination():
    form = DownloadForm()
    form.feed('<input type="hidden" name="id" value="evil">'
              '<form id="download-form" action="https://drive.usercontent.google.com/download">'
              '<input type="hidden" name="id" value="abc">'
              '<input type="hidden" name="export" value="download">'
              '<input type="hidden" name="confirm" value="t"></form>')
    form.validate("abc")
    with pytest.raises(ValueError):
        form.validate("other")
    form.action = "https://example.org/"
    with pytest.raises(ValueError):
        form.validate("abc")


@pytest.mark.parametrize("value", [None, "", "bytes */123", "bytes 10-0/123", "bytes 0-123/123"])
def test_bad_range_refused(value):
    with pytest.raises(ValueError):
        content_range(value)


def test_exact_range():
    assert content_range("bytes 10-19/100") == (10, 19, 100)


def test_canonical_image_mapping():
    assert image_identity("train_data/images/img001001.jpg") == ("00001", 1)
    assert image_identity("test_data/images/img112300.jpg") == ("00112", 300)


@pytest.mark.parametrize("name", ["../img001001.jpg", "/img001001.jpg", "img113001.jpg",
                                  "img001000.jpg", "img001301.jpg", "dir//img001001.jpg",
                                  "dir\\img001001.jpg", "img001001.exe"])
def test_unexpected_image_path_refused(name):
    with pytest.raises(ValueError):
        image_identity(name)


def test_old_annotation_permission_does_not_implicitly_authorize_images():
    with pytest.raises(ValueError):
        require_image_permission({"annotation_download_and_audit": True})
    receipt = dict(official_dronecrowd_image_acquisition=True,
                   execute_downloaded_third_party_code=False,
                   stage5c_execution=False, smc_enabled=False, user_message="Approved")
    require_image_permission(receipt)
    receipt["stage5c_execution"] = True
    with pytest.raises(ValueError):
        require_image_permission(receipt)


class MemoryRanges(RangeArchive):
    def __init__(self, data):
        io.RawIOBase.__init__(self)
        self.data, self.size = data, len(data)
        self.block_size, self.cache_blocks = 7, 2
        self.cache, self.position = OrderedDict(), 0
        self.tail_start = max(0, len(data)-12)
        self.tail = data[self.tail_start:]
        self.ranges = []

    def _fetch(self, request):
        start, end = map(int, request[6:].split("-"))
        self.ranges.append((start, end))
        return self.data[start:end+1]


def test_ranges_support_cross_block_seek_tail_and_eof():
    data = bytes(range(100))
    with MemoryRanges(data) as source:
        assert source.read(15) == data[:15]
        source.seek(-5, 1)
        assert source.read(80) == data[10:90]
        source.seek(-15, 2)
        assert source.read() == data[-15:]
        assert source.read(10) == b""
        assert len(source.cache) <= 2
        with pytest.raises(ValueError):
            source.seek(-1)


def test_zip_crc_checked_through_seekable_ranges():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.txt", b"synthetic-test-data" * 100)
    with MemoryRanges(memory.getvalue()) as source, zipfile.ZipFile(source) as archive:
        assert archive.read("fixture.txt") == b"synthetic-test-data" * 100


def test_ranges_refuse_unbounded_archive_materialization():
    with MemoryRanges(b"x" * (33 * 1024**2)) as source:
        with pytest.raises(ValueError, match="memory bound"):
            source.read()


class Response:
    def __init__(self, status, byte_range, payload, url="https://drive.usercontent.google.com/download"):
        self.status_code, self.headers, self.url = status, {"Content-Range":byte_range}, url
        self.raw = io.BytesIO(payload)

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response

    def close(self):
        pass


@pytest.mark.parametrize("status,byte_range,payload,url", [
    (200,"bytes 10-19/100",b"0123456789","https://drive.usercontent.google.com/download"),
    (206,"bytes 11-20/100",b"0123456789","https://drive.usercontent.google.com/download"),
    (206,"bytes 10-19/101",b"0123456789","https://drive.usercontent.google.com/download"),
    (206,"bytes 10-19/100",b"short","https://drive.usercontent.google.com/download"),
    (206,"bytes 10-19/100",b"01234567890","https://drive.usercontent.google.com/download"),
    (206,"bytes 10-19/100",b"0123456789","https://example.org/download"),
])
def test_transport_rejects_ignored_range_changed_source_or_bad_body(status,byte_range,payload,url):
    source = MemoryRanges(bytes(range(100)))
    source.session = Session(Response(status,byte_range,payload,url))
    source.url, source.params = "https://drive.usercontent.google.com/download", {}
    source.transferred_bytes = source.range_requests = 0
    with pytest.raises(ValueError):
        RangeArchive._fetch(source,"bytes=10-19")
    assert source.transferred_bytes == source.range_requests == 0


def test_transport_exact_response_and_suffix_are_counted():
    source = MemoryRanges(bytes(range(100)))
    source.session = Session(Response(206,"bytes 90-99/100",b"0123456789"))
    source.url, source.params = "https://drive.usercontent.google.com/download", {}
    source.transferred_bytes = source.range_requests = 0
    assert RangeArchive._fetch(source,"bytes=-10") == b"0123456789"
    assert source.transferred_bytes==10 and source.range_requests==1
