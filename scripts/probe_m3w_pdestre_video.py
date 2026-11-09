"""Verify actual first-eight-frame decoding through a bounded tar-member stream."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_pdestre as audit

CHUNK = 512*1024
MAX_TRANSFER = 24*1024**2


def span(header, size):
    if header is None:
        return 0, size-1
    m = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
    if not m:
        raise ValueError("Unsupported request range")
    a = int(m[1])
    b = min(int(m[2]), size-1) if m[2] else size-1
    if not 0 <= a <= b:
        raise ValueError("Unsatisfiable request range")
    return a, b


class MemberStream:
    def __init__(self, member):
        self.size = member["bytes"]
        self.start = member["offset"]
        self.transferred = 0
        self.lock = threading.Lock()

    def handler(self):
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path != "/clip.mp4":
                    self.send_error(404)
                    return
                header = self.headers.get("Range")
                try:
                    lo, hi = span(header, stream.size)
                except ValueError:
                    self.send_error(416)
                    return
                self.send_response(206 if header else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(hi-lo+1))
                self.send_header("Accept-Ranges", "bytes")
                if header:
                    self.send_header("Content-Range", f"bytes {lo}-{hi}/{stream.size}")
                self.end_headers()
                try:
                    for offset in range(lo, hi+1, CHUNK):
                        n = min(hi+1-offset, CHUNK)
                        with stream.lock:
                            if stream.transferred+n > MAX_TRANSFER:
                                raise ValueError("Smoke exceeded media transfer cap")
                            stream.transferred += n
                        self.wfile.write(audit.byte_range(stream.start+offset, n))
                except (BrokenPipeError, ConnectionResetError):
                    return

        return Handler


def main():
    import cv2

    index = audit.OUT/"range_index.json"
    data = audit.load(index)
    if (data["url"], data["size"], data["etag"]) != (audit.URL, audit.SIZE, audit.ETAG):
        raise ValueError("Wrong publisher snapshot")
    clip = min(data["clips"])
    entry = data["clips"][clip][".mp4"]
    stream = MemberStream(entry)
    with ThreadingHTTPServer(("127.0.0.1", 0), stream.handler()) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        camera = cv2.VideoCapture()
        try:
            if not camera.open(f"http://127.0.0.1:{server.server_port}/clip.mp4", cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 45000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 45000]):
                raise ValueError("MP4 stream is not decodable")
            fps = camera.get(cv2.CAP_PROP_FPS)
            frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            shape, fingerprint = None, hashlib.sha256()
            for _ in range(8):
                ok, frame = camera.read()
                if not ok:
                    raise ValueError("Incomplete eight-frame decode")
                if shape is not None and shape != list(frame.shape):
                    raise ValueError("Inconsistent raster shape")
                shape = list(frame.shape)
                fingerprint.update(frame.tobytes())
        finally:
            camera.release()
            server.shutdown()
            worker.join()
    if not 0 < fps or frames < 8:
        raise ValueError("Missing video clock or frame count")
    r = dict(index_sha256=audit.digest(index.read_bytes()), opencv=cv2.__version__, source=entry,
             prefix_frames_decoded=8, raster_shape=shape, prefix_sha256=fingerprint.hexdigest(),
             stream_reported_fps=fps, stream_reported_frames=frames,
             compressed_fetch_cap_bytes=MAX_TRANSFER, images_saved=False,
             track_inference=False, scientific_roles_assigned=False, reserved_outcomes_opened=False)
    audit.save(audit.OUT/"video_prefix_probe.json", r)
    print(f"Decoded 8 frames at {fps:.6g} reported fps, {shape}; requested {stream.transferred:,} bytes.", flush=True)


if __name__ == "__main__":
    main()
