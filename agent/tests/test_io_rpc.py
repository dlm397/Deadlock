import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from agent.io import API


def response(message=None, text=None, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://api.deadlock-api.com/v1/mcp"
    result.encoding = "utf-8"
    result._content = (text if text is not None else json.dumps(message)).encode()
    result.headers.update(headers or {})
    return result


def rpc_result(result):
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def query_result(rows):
    return {"structuredContent": {"result": json.dumps({
        "success": True, "columns": ["match_id"], "rows": rows,
    })}}


class LakeRPCTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        session = patch("agent.io.requests.Session")
        self.addCleanup(session.stop)
        self.session = session.start().return_value
        self.api = API(Path(temporary.name))

    def test_sse_handshake_initializes_once_and_keeps_session_headers(self):
        initialized = rpc_result({"protocolVersion": "2025-03-26"})
        event = "event: message\r\n" + "\r\n".join(
            "data: " + line for line in json.dumps(initialized, indent=2).splitlines()
        ) + "\r\n\r\n"
        self.session.post.side_effect = [
            response(text=event, headers={"Mcp-Session-Id": "test-session"}),
            response(text="", status=202),
            response(rpc_result(query_result([[10], [9]]))),
            response(rpc_result(query_result([[8]]))),
        ]
        self.assertEqual(self.api.lake_query("SELECT first_page"), [{"match_id": 10}, {"match_id": 9}])
        self.assertEqual(self.api.lake_query("SELECT next_page"), [{"match_id": 8}])
        calls = self.session.post.call_args_list
        self.assertEqual([call.kwargs["json"]["method"] for call in calls],
                         ["initialize", "notifications/initialized", "tools/call", "tools/call"])
        self.assertNotIn("id", calls[1].kwargs["json"])
        self.assertEqual(self.api.mcp_headers["Mcp-Session-Id"], "test-session")
        self.assertEqual(self.api.mcp_headers["MCP-Protocol-Version"], "2025-03-26")
        self.assertEqual(calls[-1].kwargs["json"]["params"]["arguments"]["sql"], "SELECT next_page")
        self.assertEqual(len(self.api.requests), 4)

    def test_sse_query_ignores_notification_and_reads_text_content(self):
        self.api.mcp_ready = True
        table = {"success": True, "columns": ["match_id"], "rows": [[2**53 + 1]]}
        message = rpc_result({"content": [{"type": "text", "text": json.dumps(table)}]})
        event = 'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
        event += "data: " + json.dumps(message) + "\n\n"
        self.session.post.return_value = response(text=event)
        self.assertEqual(self.api.lake_query("SELECT match_id"), [{"match_id": 2**53 + 1}])

    def test_rpc_error_is_reported(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response({
            "jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Unknown tool"},
        })
        with self.assertRaisesRegex(RuntimeError, "Unknown tool"):
            self.api.lake_query("SELECT match_id")

    def test_tool_error_is_not_treated_as_an_empty_page(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result({
            "isError": True, "content": [{"type": "text", "text": "Query resource limit exceeded"}],
        }))
        with self.assertRaisesRegex(RuntimeError, "resource limit"):
            self.api.lake_query("SELECT match_id")

    def test_structured_query_failure_is_reported(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result({
            "structuredContent": {"result": json.dumps({"success": False, "error": "No such table"})},
        }))
        with self.assertRaisesRegex(RuntimeError, "No such table"):
            self.api.lake_query("SELECT match_id")

    def test_empty_query_result_is_valid(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result(query_result([])))
        self.assertEqual(self.api.lake_query("SELECT match_id"), [])


if __name__ == "__main__":
    unittest.main()
