from xml.etree import ElementTree as XML

import numpy as np
import pytest

from scripts import acquire_m3w_av2_development as acquire
from scripts import run_m3w_av2_data_scale as run


def receipt(*paths):
    root = XML.Element("ListBucketResult")
    for path in paths:
        x = XML.SubElement(root, "Contents")
        XML.SubElement(x, "Key").text = path
        XML.SubElement(x, "Size").text = "1234"
        XML.SubElement(x, "ETag").text = '"test-tag"'
    XML.SubElement(root, "NextContinuationToken").text = "next"
    return XML.tostring(root)


def test_train_listing_ignores_maps_and_other_splits():
    name = "abcdef12-0123-0123-0123-0123456789ab"
    valid = f"{run.av.PREFIX}{name}/scenario_{name}.parquet"
    hidden = valid.replace("/train/", "/test/")
    map_file = f"{run.av.PREFIX}{name}/log_map_archive_{name}.json"
    rows, token = acquire.listing(receipt(hidden, valid, map_file), "a")
    assert rows == {name: dict(key=valid, bytes=1234, etag='"test-tag"')}
    assert token == "next"
    with pytest.raises(ValueError, match="duplicated"):
        acquire.listing(receipt(valid, valid), "a")
    with pytest.raises(ValueError, match="Unexpected"):
        acquire.listing(receipt(valid), "b")


def test_duplicate_observed_roster_hash_ignores_agent_order_not_time():
    h = np.arange(48).reshape(3, 8, 2)
    assert run.history_fingerprint(h) == run.history_fingerprint(h[[2, 0, 1]])
    assert run.history_fingerprint(h) != run.history_fingerprint(h[:, ::-1])
    assert run.history_fingerprint(h) != run.history_fingerprint(h+1)
