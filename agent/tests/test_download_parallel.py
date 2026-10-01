import hashlib
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests

from agent.io import API, read_json


PAYLOAD = bytes(range(256)) * 2048
ETAG = '"replay-one"'
MODIFIED = "Fri, 25 Sep 2026 18:14:47 GMT"


class ReplayHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        requested = self.headers.get("Range")
        with server.lock:
            server.requests.append(dict(self.headers))
        mode = server.mode
        if requested and mode == "mixed_failure":
            server.barrier.wait(timeout=3)
        if requested and (mode == "throttled" or mode == "mixed_failure" and requested != "bytes=0-131071"):
            if mode == "mixed_failure":
                threading.Event().wait(0.03)
            self.send_response(429)
            self.send_header("Retry-After", "120")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = PAYLOAD
        start, end = 0, len(body) - 1
        ranged = bool(requested) and mode != "ignored"
        if ranged:
            start, end = map(int, requested.removeprefix("bytes=").split("-"))
            body = body[start:end + 1]
        self.send_response(206 if ranged else 200)
        self.send_header("Accept-Ranges", "bytes")
        if mode != "no_validator":
            value = 'W/"replay-one"' if mode in ("weak_validator", "modified") else ETAG
            self.send_header("ETag", '"replay-two"' if ranged and mode == "changed" else value)
        if mode == "modified":
            self.send_header("Last-Modified", MODIFIED)
        if ranged and mode != "missing_range":
            self.send_header("Content-Range", f"bytes {start + (mode in ('bad_range', 'mixed_failure'))}-{end}/{len(PAYLOAD)}")
        if ranged and mode == "encoded_range":
            self.send_header("Content-Encoding", "gzip")
        if not (ranged and mode in ("oversized", "no_length")):
            self.send_header("Content-Length", str(len(body) + (ranged and mode == "bad_length")))
        self.end_headers()
        if ranged and mode == "truncated":
            body = body[:len(body) // 2]
        if ranged and mode == "oversized":
            body += b"unexpected extra bytes"
        with server.lock:
            server.active += 1
            server.max_active = max(server.max_active, server.active)
        try:
            for offset in range(0, len(body), 16384):
                if server.delay:
                    time.sleep(server.delay)
                self.wfile.write(body[offset:offset + 16384])
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with server.lock:
                server.active -= 1


class ParallelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
        cls.server.daemon_threads = True
        cls.server.lock = threading.Lock()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = API(self.root)
        self.addCleanup(self.api.session.close)
        self.target = self.root / "replay.dem.bz2"
        self.url = f"http://127.0.0.1:{self.server.server_port}/replay"
        self.server.mode = "normal"
        self.server.requests = []
        self.server.delay = 0
        self.server.barrier = threading.Barrier(4)
        self.server.active = self.server.max_active = 0
        threshold = patch("agent.io.PARALLEL_DOWNLOAD_MIN_BYTES", 4096)
        threshold.start()
        self.addCleanup(threshold.stop)

    def assert_download(self):
        self.assertEqual(self.api.download(self.url, self.target), self.target)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        record = read_json(self.target.with_suffix(".bz2.provenance.json"))
        self.assertEqual(record["sha256"], hashlib.sha256(PAYLOAD).hexdigest())
        self.assertEqual(record["bytes"], len(PAYLOAD))
        self.assertEqual(record["source"], self.url)
        self.assertFalse(self.target.with_suffix(".bz2.part").exists())

    def test_ranges_preserve_bytes_provenance_and_cached_reuse(self):
        self.assert_download()
        ranges = [entry for entry in self.server.requests if "Range" in entry]
        self.assertEqual(len(ranges), 4)
        self.assertTrue(all(entry["If-Range"] == ETAG and entry["Accept-Encoding"] == "identity"
                            for entry in ranges))
        count = len(self.server.requests)
        self.assert_download()
        self.assertEqual(len(self.server.requests), count)

    def test_malformed_or_unsupported_ranges_fall_back(self):
        for mode in ("ignored", "missing_range", "bad_range", "bad_length", "changed", "encoded_range", "oversized"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.server.requests = []
                self.target.unlink(missing_ok=True)
                self.assert_download()
                self.assertTrue(any("Range" in entry for entry in self.server.requests))
                self.assertNotIn("Range", self.server.requests[-1])

    def test_missing_and_weak_validators_use_single_stream(self):
        for mode in ("no_validator", "weak_validator"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                self.server.requests = []
                self.target.unlink(missing_ok=True)
                self.assert_download()
                self.assertEqual(len(self.server.requests), 1)

    def test_last_modified_can_validate_ranges(self):
        self.server.mode = "modified"
        self.assert_download()
        ranges = [entry for entry in self.server.requests if "Range" in entry]
        self.assertEqual(len(ranges), 4)
        self.assertTrue(all(entry["If-Range"] == MODIFIED for entry in ranges))

    def test_small_files_do_not_probe_or_fork(self):
        with patch("agent.io.PARALLEL_DOWNLOAD_MIN_BYTES", len(PAYLOAD) + 1):
            self.assert_download()
        self.assertEqual(len(self.server.requests), 1)

    def test_range_lengths_checked_without_content_length(self):
        self.server.mode = "no_length"
        self.assert_download()
        self.assertEqual(len(self.server.requests), 5)

    def test_worker_count_is_bounded(self):
        for workers in (0, -1, 9, 1.5, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.api.download(self.url, self.target, workers=workers)
        self.assertEqual(self.server.requests, [])

    def test_truncated_range_retries_in_one_stream(self):
        self.server.mode = "truncated"
        with patch("agent.io.time.sleep") as sleep:
            self.assert_download()
        sleep.assert_called_once_with(2)
        self.assertNotIn("Range", self.server.requests[-1])

    def test_retry_after_is_not_bypassed_by_fallback(self):
        self.server.mode = "throttled"
        self.target.write_bytes(b"previous replay")
        with patch("agent.io.time.sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                self.api.download(self.url, self.target)
        sleep.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"previous replay")
        self.assertFalse(self.target.with_suffix(".bz2.part").exists())
        self.assertEqual(sum("Range" not in entry for entry in self.server.requests), 1)

    def test_simultaneous_throttle_precedes_malformed_range_fallback(self):
        self.server.mode = "mixed_failure"
        with patch("agent.io.time.sleep") as sleep:
            with self.assertRaises(requests.HTTPError):
                self.api.download(self.url, self.target)
        sleep.assert_not_called()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix(".bz2.part").exists())
        self.assertEqual(sum("Range" not in entry for entry in self.server.requests), 1)

    def test_fork_does_not_share_sessions_or_request_records(self):
        self.api.session.headers["X-Research"] = "example"
        self.api.requests.append({"url": "example"})
        child = self.api.fork()
        self.addCleanup(child.session.close)
        self.assertIsNot(child.session, self.api.session)
        self.assertEqual(child.requests, [])
        self.assertEqual(child.root, self.api.root)
        self.assertEqual(child.download_workers, self.api.download_workers)
        self.assertEqual(child.session.headers["X-Research"], "example")

    def test_parallel_download_overlaps_throttled_connections(self):
        self.server.delay = 0.025
        start = time.perf_counter()
        self.api.download(self.url, self.target, workers=1)
        sequential = time.perf_counter() - start
        self.target.unlink()
        start = time.perf_counter()
        self.assert_download()
        parallel = time.perf_counter() - start
        self.assertGreaterEqual(self.server.max_active, 2)
        self.assertLess(parallel, sequential * 0.75)
        print(f"Local replay benchmark: one stream {sequential:.3f}s, four ranges {parallel:.3f}s "
              f"({sequential / parallel:.2f}x)")


if __name__ == "__main__":
    unittest.main()
