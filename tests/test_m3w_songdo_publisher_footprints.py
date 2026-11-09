import pytest

from scripts import audit_m3w_songdo_publisher_footprints as audit


def test_cutout_overlap_components_keep_transitive_links_and_boundary():
    edges, groups = audit.footprints(dict(A=(0, 0), B=(15000, 0), C=(29999, 14999), D=(50000, 0)))
    assert edges == [['A', 'B'], ['B', 'C']]
    assert groups == [['A', 'B', 'C'], ['D']]


def test_range_reader_checks_response_and_reuses_block(monkeypatch):
    calls = []

    def response(url, limit, *options):
        calls.append(options)
        return b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/4\r\n\r\nabcd'

    monkeypatch.setattr(audit, 'curl', response)
    f = audit.RemoteZip('source', 4)
    f.seek(1)
    assert f.read(2) == b'bc'
    f.seek(0)
    assert f.read() == b'abcd'
    assert f.fetched == 4 and len(calls) == 1
    monkeypatch.setattr(audit, 'curl', lambda *a: b'HTTP/1.1 200 OK\r\n\r\nabcd')
    with pytest.raises(ValueError, match='exact requested range'):
        audit.RemoteZip('source', 4).read()
