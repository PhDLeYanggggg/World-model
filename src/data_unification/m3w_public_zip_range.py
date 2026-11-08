"""Bounded, verified byte-range reader for ZIP publisher archives."""
from collections import OrderedDict
from io import RawIOBase
import ssl
from urllib.request import Request, urlopen


class RangeArchive(RawIOBase):
    def __init__(self, url, length, *, context=None, block_size=1 << 20, byte_limit=64 << 20):
        super().__init__()
        self.url, self.length, self.block_size, self.byte_limit = url,length,block_size,byte_limit
        self.context = context or ssl.create_default_context()
        self.pos, self.total = 0, 0
        self.version = None
        self.blocks = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        bases = {0:0,1:self.pos,2:self.length}
        if whence not in bases or offset+bases[whence]<0:
            raise ValueError('Invalid archive seek')
        self.pos = offset+bases[whence]
        return self.pos

    def _fetch(self, first, last):
        expected = last-first+1
        if self.total+expected>self.byte_limit:
            raise ValueError('Archive range budget exceeded')
        req = Request(self.url,headers={'Range':f'bytes={first}-{last}','Accept-Encoding':'identity'})
        with urlopen(req,context=self.context,timeout=30) as res:
            key = res.headers.get('x-amz-version-id') or res.headers.get('ETag')
            if (res.status!=206 or res.headers.get('Content-Range')!=f'bytes {first}-{last}/{self.length}'
                    or not key or self.version not in (None,key)):
                raise ValueError('Range response does not bind the same archive version')
            block = res.read(expected+1)
        if len(block)!=expected:
            raise ValueError('Truncated or oversized archive range')
        self.total += expected
        self.version = key
        return block

    def read(self, size=-1):
        available = max(0,self.length-self.pos)
        remaining = available if size<0 else min(size,available)
        out = bytearray()
        while remaining:
            start = self.pos//self.block_size*self.block_size
            if start not in self.blocks:
                self.blocks[start] = self._fetch(start,min(start+self.block_size,self.length)-1)
                if len(self.blocks)>8:
                    self.blocks.popitem(last=False)
            else:
                self.blocks.move_to_end(start)
            start_in_block = self.pos-start
            piece = self.blocks[start][start_in_block:start_in_block+remaining]
            out.extend(piece)
            self.pos += len(piece)
            remaining -= len(piece)
        return bytes(out)
