import hashlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock, call, patch

import requests

from agent.io import API, RETRY_DELAYS_SECONDS, read_json


def response(status=200, body=b'{"ok":true}', headers=None, method="GET"):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.invalid/data"
    result._content = body
    result._content_consumed = True
    result.headers.update(headers or {})
    result.request = requests.Request(method, result.url).prepare()
    return result


class HTTPRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.api = API(self.root)
        self.addCleanup(self.api.session.close)
        self.request = patch.object(self.api.session, "request").start()
        self.addCleanup(patch.stopall)
        self.sleep = patch("agent.io.time.sleep").start()
        patch("agent.io.LOG").start()

    def test_json_recovers_from_503_and_archives_only_success(self):
        self.request.side_effect = [response(503, b'{"error":"retry later"}'), response()]
        payload, provenance = self.api.json("/metadata", "metadata.json")
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(read_json(self.root / "metadata.json"), payload)
        self.assertEqual([record["status"] for record in self.api.requests], [503, 200])
        saved = read_json(self.root / "metadata.json.provenance.json")
        self.assertEqual(saved["status"], 200)
        self.assertEqual(saved["retrieved_at"], provenance["retrieved_at"])
        self.sleep.assert_called_once_with(2)

    def test_repeated_503_is_bounded_and_preserves_previous_archive(self):
        archive = self.root / "metadata.json"
        archive.write_bytes(b'{"previous":true}')
        self.request.side_effect = [response(503) for _ in range(4)]
        with self.assertRaises(requests.HTTPError):
            self.api.json("/metadata", "metadata.json")
        self.assertEqual(self.request.call_count, 4)
        self.assertEqual(self.sleep.call_args_list, [call(delay) for delay in RETRY_DELAYS_SECONDS])
        self.assertEqual(archive.read_bytes(), b'{"previous":true}')
        self.assertEqual([record["status"] for record in self.api.requests], [503] * 4)

    def test_404_is_not_retried(self):
        self.request.return_value = response(404)
        with self.assertRaises(requests.HTTPError):
            self.api.json("/missing")
        self.assertEqual(self.request.call_count, 1)
        self.sleep.assert_not_called()

    def test_post_is_not_retried(self):
        self.request.return_value = response(503, method="POST")
        with self.assertRaises(requests.HTTPError):
            self.api.json("/submit", body={"query": "SELECT 1"})
        self.assertEqual(self.request.call_count, 1)
        self.sleep.assert_not_called()

    def test_retry_after_seconds_is_respected(self):
        self.request.side_effect = [response(429, headers={"Retry-After": "9"}), response()]
        self.api.json("/metadata")
        self.sleep.assert_called_once_with(9)

    def test_retry_after_http_date_is_respected(self):
        deadline = datetime.now(timezone.utc) + timedelta(seconds=20)
        self.request.side_effect = [response(503, headers={"Retry-After": format_datetime(deadline, usegmt=True)}),
                                    response()]
        self.api.json("/metadata")
        self.sleep.assert_called_once()
        delay = self.sleep.call_args.args[0]
        self.assertGreater(delay, 17)
        self.assertLessEqual(delay, 20)

    def test_long_retry_after_defers_without_retrying_early(self):
        self.request.return_value = response(429, headers={"Retry-After": "120"})
        with self.assertRaises(requests.HTTPError):
            self.api.json("/metadata")
        self.assertEqual(self.request.call_count, 1)
        self.sleep.assert_not_called()

    def test_timeout_can_recover(self):
        self.request.side_effect = [requests.Timeout("interrupted"), response()]
        self.assertEqual(self.api.json("/metadata")[0], {"ok": True})
        self.assertEqual(self.request.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_stream_reset_restarts_entire_file(self):
        def interrupted(_):
            yield b"discard this partial download"
            raise requests.exceptions.ChunkedEncodingError("connection reset")

        first = response()
        first.iter_content = interrupted
        second = response()
        second.iter_content = Mock(return_value=iter([b"complete", b" replay"]))
        self.request.side_effect = [first, second]
        target = self.root / "replay.dem.bz2"
        self.assertEqual(self.api.download("https://example.invalid/replay", target), target)
        self.assertEqual(target.read_bytes(), b"complete replay")
        saved = read_json(target.with_suffix(".bz2.provenance.json"))
        self.assertEqual(saved["sha256"], hashlib.sha256(b"complete replay").hexdigest())
        self.assertEqual(saved["bytes"], len(b"complete replay"))
        self.assertFalse(target.with_suffix(".bz2.part").exists())
        self.sleep.assert_called_once_with(2)

    def test_failed_stream_cleans_partial_and_preserves_existing_file(self):
        def interrupted(_):
            yield b"partial"
            raise requests.exceptions.ChunkedEncodingError("connection reset")

        attempts = [response() for _ in range(4)]
        for item in attempts:
            item.iter_content = interrupted
        self.request.side_effect = attempts
        target = self.root / "replay.dem.bz2"
        target.write_bytes(b"previous archive")
        with self.assertRaises(requests.exceptions.ChunkedEncodingError):
            self.api.download("https://example.invalid/replay", target)
        self.assertEqual(self.request.call_count, 4)
        self.assertEqual(target.read_bytes(), b"previous archive")
        self.assertFalse(target.with_suffix(".bz2.part").exists())
        self.assertFalse(target.with_suffix(".bz2.provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
