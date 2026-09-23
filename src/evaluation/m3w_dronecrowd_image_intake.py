"""Bounded public-archive reads for source auditing, never model features."""
from __future__ import annotations

from collections import OrderedDict
from html.parser import HTMLParser
import io
import re
from urllib.parse import urlparse

import requests

IMAGE_ARCHIVES = {
    "train": "1piwSmJ5ySlQJsKHXSSG2gicbE86Ib4HV",
    "test": "1C9PGUZ9GPB_NnUBVGbkJQFXhRxD3bfT9",
}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}
        self.inside = False

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "form" and values.get("id") == "download-form":
            self.action = values.get("action")
            self.inside = True
        if tag == "input" and self.inside and values.get("type") == "hidden":
            self.params[values["name"]] = values.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False

    def validate(self, file_id):
        if (self.action != "https://drive.usercontent.google.com/download"
                or self.params.get("id") != file_id
                or self.params.get("export") != "download"
                or self.params.get("confirm") != "t"):
            raise ValueError("Unexpected download form; no unreviewed destination")


def require_image_permission(receipt):
    if not (receipt.get("official_dronecrowd_image_acquisition") is True
            and receipt.get("execute_downloaded_third_party_code") is False
            and receipt.get("stage5c_execution") is False
            and receipt.get("smc_enabled") is False
            and receipt.get("user_message")):
        raise ValueError("Image-audit authorization receipt required")


def content_range(value):
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    if not match:
        raise ValueError("Missing exact HTTP Content-Range")
    start, end, size = map(int, match.groups())
    if not 0 <= start <= end < size:
        raise ValueError("Invalid HTTP byte range")
    return start, end, size


def image_identity(name):
    if "\\" in name or any(p in ("..", "") for p in name.split("/")) or name.startswith("/"):
        raise ValueError("Unsafe archive member path")
    match = re.fullmatch(r"img(\d{3})(\d{3})\.jpg", name.split("/")[-1])
    if not match:
        raise ValueError("Unexpected image member name")
    scene, frame = map(int, match.groups())
    if not (1 <= scene <= 112 and 1 <= frame <= 300):
        raise ValueError("Image identity outside annotation bounds")
    return f"{scene:05d}", frame


class RangeArchive(io.RawIOBase):
    """Seekable read-only HTTP view; refuses servers ignoring ranges.

    Member CRC plus a repeated central-directory hash binds selected payloads.
    This does not claim a SHA256 of the entire undownloaded archive.
    """
    def __init__(self, file_id, *, block_size=262144, cache_blocks=96):
        if file_id not in IMAGE_ARCHIVES.values():
            raise ValueError("Only verified official image archives are allowed")
        super().__init__()
        self.file_id = file_id
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.cache = OrderedDict()
        self.session = requests.Session()
        self.position = 0
        self.size = None
        self.transferred_bytes = 0
        self.range_requests = 0
        with self.session.get("https://drive.google.com/uc",
                              params={"export": "download", "id": file_id},
                              stream=True, timeout=(10, 45)) as response:
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                raise ValueError("Expected provider confirmation page")
            data = response.raw.read(65537)
        if len(data) > 65536:
            raise ValueError("Oversized download confirmation page")
        form = DownloadForm()
        form.feed(data.decode("utf-8"))
        form.validate(file_id)
        self.url, self.params = form.action, form.params
        tail = self._fetch("bytes=-65557")
        self.tail_start = self.size - len(tail)
        self.tail = tail

    def _fetch(self, requested):
        with self.session.get(self.url, params=self.params, stream=True,
                              headers={"Range": requested, "Accept-Encoding": "identity"},
                              timeout=(10, 60)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError("Range unsupported; refusing accidental full download")
            if urlparse(response.url).hostname != "drive.usercontent.google.com":
                raise ValueError("Unexpected archive destination")
            start, end, size = content_range(response.headers.get("Content-Range"))
            if self.size is not None and size != self.size:
                raise ValueError("Remote archive size changed")
            if requested.startswith("bytes=-"):
                expected_start, expected_end = max(0, size - int(requested[7:])), size - 1
            else:
                expected_start, expected_end = map(int, requested[6:].split("-"))
            if (start, end) != (expected_start, expected_end):
                raise ValueError("Server returned wrong byte range")
            length = end - start + 1
            data = response.raw.read(length + 1)
            if len(data) != length:
                raise ValueError("Truncated or oversized range response")
            self.size = size
            self.transferred_bytes += length
            self.range_requests += 1
            return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}.get(whence)
        if base is None or base + offset < 0:
            raise ValueError("Invalid seek")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        end = min(self.position + size, self.size)
        if end - self.position > 32 * 1024**2:
            raise ValueError("Read exceeds sparse-audit memory bound")
        pieces = []
        while self.position < end:
            if self.position >= self.tail_start:
                pieces.append(self.tail[self.position-self.tail_start:end-self.tail_start])
                self.position = end
                break
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                self.cache[block] = self._fetch(f"bytes={start}-{min(start+self.block_size,self.size)-1}")
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            data = self.cache[block]
            self.cache.move_to_end(block)
            offset = self.position % self.block_size
            count = min(end-self.position, len(data)-offset)
            pieces.append(data[offset:offset+count])
            self.position += count
        return b"".join(pieces)

    def close(self):
        if hasattr(self, "session"):
            self.session.close()
        super().close()
