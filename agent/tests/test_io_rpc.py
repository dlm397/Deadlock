import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import requests

from agent.io import API, RETRY_DELAYS_SECONDS
from agent.lake_response import LakeUnavailable, query_rows


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

    def test_live_toon_shape_preserves_types_large_ids_and_escaped_text(self):
        text = ('success: true\ncolumns[6]: id,text,active,missing,label,value\n'
                'columnTypes[6]: UBIGINT,VARCHAR,BOOLEAN,INTEGER,VARCHAR,"DECIMAL(2,1)"\n'
                'rows[2]{id,text,active,missing,label,value}:\n'
                '  18446744073709551615,"comma, quote \\" and slash \\\\ and line\\n雪",true,null,"null",0.5\n'
                '  9007199254740993,"",false,null,"001",-2e-3\nrowCount: 2')
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result({"structuredContent": {"result": text}}))
        result = self.api.lake_query("SELECT values")
        self.assertEqual(result[0], {"id": 2**64 - 1, "text": 'comma, quote " and slash \\ and line\n雪',
                                    "active": True, "missing": None, "label": "null", "value": 0.5})
        self.assertEqual(result[1], {"id": 2**53 + 1, "text": "", "active": False,
                                    "missing": None, "label": "001", "value": -0.002})

    def test_toon_text_only_empty_rows(self):
        for empty in ("rows[0]:", "rows: []"):
            payload = f"success: true\ncolumns[1]: x\n{empty}\nrowCount: 0"
            self.assertEqual(query_rows({"structuredContent": {}, "content": [{"type": "text", "text": payload}]}), [])

    def test_toon_pipe_and_tab_delimiters(self):
        for delimiter in ("|", "\t"):
            payload = (f"success: true\ncolumns[2{delimiter}]: id{delimiter}label\n"
                       f"rows[1{delimiter}]{{id{delimiter}label}}:\n  5{delimiter}plain,comma\nrowCount: 1")
            self.assertEqual(query_rows({"structuredContent": {"result": payload}}), [{"id": 5, "label": "plain,comma"}])

    def test_native_structured_result_and_json_object_rows(self):
        table = {"success": True, "columns": ["id"], "rows": [{"id": 2**64 - 1}], "rowCount": 1}
        for payload in (table, {"result": table}, {"result": json.dumps(table)}):
            self.assertEqual(query_rows({"structuredContent": payload}), [{"id": 2**64 - 1}])

    def test_incomplete_or_malformed_toon_is_rejected_without_retry(self):
        prefixes = "success: true\ncolumns[2]: id,label\n"
        bad = [
            prefixes + 'rows[2]{id,label}:\n  1,only one row\nrowCount: 2',
            prefixes + 'rows[1]{id,label}:\n  1\nrowCount: 1',
            prefixes + 'rows[1]{id,label}:\n  1,x,extra\nrowCount: 1',
            prefixes + 'rows[1]{id,label}:\n  1,"unterminated\nrowCount: 1',
            prefixes + 'rows[1]{id,label}:\n  1,x\nrowCount: 2',
            prefixes + 'rows[1]{id,other}:\n  1,x\nrowCount: 1',
            prefixes + 'rows[1]{id,label}:\n  1,x\nrowCount: 1\ntruncated: true',
            'success: true\ncolumns[2]: id,id\nrows[0]:\nrowCount: 0',
            'success: true\nsuccess: true\ncolumns[1]: id\nrows[0]:',
        ]
        self.api.mcp_ready = True
        for text in bad:
            with self.subTest(text=text), patch("agent.io.time.sleep") as sleep:
                self.session.post.reset_mock()
                self.session.post.return_value = response(rpc_result({"structuredContent": {"result": text}}))
                with self.assertRaises(RuntimeError):
                    self.api.lake_query("SELECT partial")
                sleep.assert_not_called()
                self.session.post.assert_called_once()

    def test_snapshot_loading_retries_before_returning_results(self):
        self.api.mcp_ready = True
        for loading in (
            {"isError": True, "content": [{"type": "text", "text": "Error calling tool 'execute_query': success: false\nerror: snapshot catalog is still loading\nerrorType: NotReadyError"}]},
            {"structuredContent": {"result": 'success: false\nerror: "Please wait"\nerrorType: NotReadyError'}},
            {"structuredContent": {"result": json.dumps({"success": False, "errorType": "NotReadyError", "error": "Please wait"})}},
        ):
            self.session.post.side_effect = [response(rpc_result(loading)), response(rpc_result(query_result([[7]])))]
            with patch("agent.io.time.sleep") as sleep, patch("agent.io.LOG"):
                self.assertEqual(self.api.lake_query("SELECT match_id"), [{"match_id": 7}])
                sleep.assert_called_once_with(2)

    def test_snapshot_loading_retry_limit_is_bounded(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result({"isError": True, "content": [
            {"type": "text", "text": "NotReadyError: snapshot catalog is still loading"}]}))
        with patch("agent.io.time.sleep") as sleep, patch("agent.io.LOG"):
            with self.assertRaisesRegex(LakeUnavailable, "still loading"):
                self.api.lake_query("SELECT match_id")
            self.assertEqual(sleep.call_args_list, [call(seconds) for seconds in RETRY_DELAYS_SECONDS])
            self.assertEqual(self.session.post.call_count, 4)

    def test_read_only_query_http_retry_after_is_respected(self):
        self.api.mcp_ready = True
        self.session.post.side_effect = [response(status=429, headers={"Retry-After": "9"}),
                                         response(rpc_result(query_result([[7]])))]
        with patch("agent.io.time.sleep") as sleep, patch("agent.io.LOG"):
            self.assertEqual(self.api.lake_query("SELECT match_id"), [{"match_id": 7}])
            sleep.assert_called_once_with(9)

    def test_toon_query_error_is_clear_and_not_retried(self):
        self.api.mcp_ready = True
        self.session.post.return_value = response(rpc_result({"structuredContent": {"result":
            'success: false\nerror: "No such table"\nerrorType: BinderException'}}))
        with patch("agent.io.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "No such table"):
                self.api.lake_query("SELECT missing")
            sleep.assert_not_called()

    def test_empty_or_malformed_rpc_and_empty_tool_response_can_recover(self):
        self.api.mcp_ready = True
        for bad in (response(text=""), response(text="{broken"), response(rpc_result({"content": []}))):
            self.session.post.side_effect = [bad, response(rpc_result(query_result([[8]])))]
            with patch("agent.io.time.sleep") as sleep, patch("agent.io.LOG"):
                self.assertEqual(self.api.lake_query("SELECT match_id"), [{"match_id": 8}])
                sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
