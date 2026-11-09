"""Bounded random reads from a pinned HTTP archive, never a full download."""

import io
from urllib.parse import urlsplit

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, *, budget=64_000_000, max_read=8_000_000, session=None):
        self.url, self.size, self.budget, self.max_read = url, size, budget, max_read
        self.position, self.transferred, self.requests = 0, 0, 0
        self.blocks = []
        self.session = session or requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        position = (0, self.position, self.size)[whence]+offset
        if position < 0:
            raise OSError('Negative archive seek')
        self.position = position
        return position

    def span(self, start, length):
        if start < 0 or length < 0 or start+length > self.size:
            raise ValueError('Requested byte range outside archive')
        if not length:
            return b''
        end = start+length
        for offset, raw in reversed(self.blocks):
            if offset <= start and end <= offset+len(raw):
                return raw[start-offset:end-offset]
        if length > self.max_read or self.transferred+length > self.budget:
            raise ValueError('Archive transfer exceeds the approved byte budget')
        headers = {'Range': f'bytes={start}-{end-1}', 'Accept-Encoding': 'identity'}
        with self.session.get(self.url, headers=headers, stream=True, timeout=(10, 30)) as r:
            hosts = [urlsplit(z.url) for z in (*r.history, r)]
            if any(z.scheme != 'https' or not (z.hostname == 'huggingface.co' or
                                               (z.hostname or '').endswith('.hf.co')) for z in hosts):
                raise ValueError('Unexpected archive redirect host')
            if (r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.size}'
                    or r.headers.get('Content-Encoding', 'identity') != 'identity'):
                raise OSError('Server did not honor the exact bounded archive range')
            raw = r.raw.read(length+1)
            if len(raw) != length:
                raise OSError('Archive range returned the wrong byte count')
        self.transferred += length
        self.requests += 1
        self.blocks.append((start, raw))
        return raw

    def read(self, size=-1):
        n = max(0, self.size-self.position)
        if size >= 0:
            n = min(n, size)
        raw = self.span(self.position, n) if n else b''
        self.position += len(raw)
        return raw
