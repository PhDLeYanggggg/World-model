import io
import tarfile

import numpy as np
import pytest

from scripts.audit_m3w_pdestre import components, directory, identifiers, pairs, supports
from scripts.probe_m3w_pdestre_video import span


def test_range_index_skips_payloads_and_pairs_video_with_annotations():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as z:
        for name, payload in (("annotations/A.txt", b"0,2,opaque\n"), ("videos/A.mp4", b"v"*1536)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            z.addfile(info, io.BytesIO(payload))
    raw, requests = buf.getvalue(), []

    def read(offset, length):
        requests.append(offset)
        return raw[offset:offset+length]

    p = pairs(directory(read, len(raw)))
    assert requests == [0, 1024, 3072, 3584]
    assert p["A"][".txt"]["offset"] == 512
    assert p["A"][".mp4"]["bytes"] == 1536


def test_http_member_ranges_cannot_spill_into_the_next_tar_member():
    assert span("bytes=512-", 2048) == (512, 2047)
    assert span("bytes=512-3000", 2048) == (512, 2047)
    assert span(None, 2048) == (0, 2047)
    with pytest.raises(ValueError):
        span("bytes=2048-", 2048)


def test_id_parser_ignores_nonstructural_fields():
    frame, ids, missing = identifiers(b"0,5,not parsed\n1,-1,ignored\n\n7,4,unparsed\n")
    assert frame.tolist() == [0, 7] and ids.tolist() == [5, 4] and missing == 1
    with pytest.raises(ValueError):
        identifiers(b"0,9\n")


def test_query_admission_does_not_require_future_targets():
    z = supports([np.arange(20), np.arange(8)], 1)
    assert z["first_selected_anchor"] == 7
    assert z["history_targets"] == 2 and z["complete_targets"] == z["absent_targets"] == 1
    assert z["queries_with_two_complete"] == 0


def test_strided_support_requires_exact_ticks_and_preserves_transitive_groups():
    z = supports([np.arange(0, 240, 12), np.arange(0, 84, 12), np.arange(0, 240, 12)], 12)
    assert z["history_targets"] == z["complete_targets"] == 2
    assert z["first_selected_anchor"] == 84
    assert components({"a": [1], "b": [2], "c": [1, 2], "d": [3]}) == [
        {"clips": ["a", "b", "c"], "tracks": 2}, {"clips": ["d"], "tracks": 1}]
