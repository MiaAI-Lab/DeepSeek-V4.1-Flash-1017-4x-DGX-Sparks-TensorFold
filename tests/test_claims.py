"""The claim scripts in tools/claims (0042, 0043, 0044) against a loopback stand-in for the server.

Offline: the stand-in answers /v1/chat/completions (0042, 0043) or /v1/models and /v1/responses (0044) the way a build
with the patch does, unless a test makes it answer otherwise (an error object beside good data, a reply without its
statistics, a warning on an earlier stream event only, a body left in the socket). Each case pins the script's exit:
0 the claim holds, 1 it does not, 2 the script cannot judge.
Run: python3 -m unittest tests.test_claims       SCRIPTS_DIR=<a repository copy> tests the scripts there instead.
"""
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import threading
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)
CLAIMS = ROOT / "tools" / "claims"
WARN = "orphan_dsml_closers"
CLOSERS = {"</｜DSML｜ parameter>", "</｜DSML｜ invoke>", "</｜DSML｜ calls>",
           "</｜DSML｜parameter>", "</｜DSML｜invoke>", "</｜DSML｜tool_calls>"}
ASK_PREFIX = "Reply with exactly this text and nothing else: "


def closers_only(text):
    rest = text
    while rest:
        for c in CLOSERS:
            if rest.startswith(c):
                rest = rest[len(c):]
                break
        else:
            return False
    return True


class Server:
    """A reply as a build with 0042 and 0043 gives it; `mode` names the one way it differs."""

    def __init__(self):
        self.mode = "valid"
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                events, whole = test.reply(body)
                if body.get("stream"):
                    data = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
                    self.send(data.encode(), "text/event-stream")
                else:
                    self.send(json.dumps(whole).encode(), "application/json")

            def send(self, data, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()

    def reply(self, body):
        tools = bool(body.get("tools"))
        ask = body["messages"][-1]["content"]
        if "guided_regex" in body:  # 0043: the forced text
            content, calls, finish = ask[len(ASK_PREFIX):], None, "stop"
        elif "response_format" in body or "guided_json" in body:  # 0042: the JSON stays content
            content, calls, finish = json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}}), None, "stop"
            if self.mode == "bad_content":
                content = "not json"
        else:  # 0042's control: an ordinary tool call
            content, calls, finish = None, [{"id": "call_1", "type": "function",
                                             "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}], "tool_calls"
        tf = {"token_sha": hashlib.sha256((content or "").encode()).hexdigest()[:12]}
        usage = {"prompt_tokens": 10, "completion_tokens": len(content or "x") // 3}
        if tools and content and closers_only(content) and self.mode != "missing_warning":
            tf["tool_parse_warning"] = WARN
        if self.mode == "missing_stats":
            tf.pop("token_sha")
            usage.pop("completion_tokens")
        rid = "chatcmpl-" + hashlib.sha256(ask.encode()).hexdigest()[:8]
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        whole = {"id": rid, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                 "usage": usage, "tensorfold": tf}
        delta = {"role": "assistant", "content": content or ""}
        if calls:
            delta["tool_calls"] = [dict(c, index=0) for c in calls]
        events = [{"id": rid, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  {"id": rid, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "tensorfold": tf}]
        if self.mode == "stream_mismatch":
            events[-1]["usage"] = dict(usage, completion_tokens=usage.get("completion_tokens", 0) + 1)
        else:
            events.append({"id": rid, "choices": [], "usage": usage})
        if self.mode == "early_warning" and "tool_parse_warning" in tf:
            events[0]["tensorfold"] = events[1].pop("tensorfold")  # the final event carries none
        failure = {"message": "the engine failed this request", "type": "server_error"}
        if self.mode == "error_envelope" and calls is None:  # (0042's control call stays good)
            whole["error"] = failure
            events[1]["error"] = failure
        if self.mode == "stream_error" and calls is None:  # the whole replies are good, the streams report a failure
            events[1]["error"] = failure
        if self.mode == "error_null":  # an explicit "no error" is a good reply
            whole["error"] = None
            for e in events:
                e["error"] = None
        return events, whole


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class ClaimScripts(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.addCleanup(self.server.close)

    def run_claim(self, name, mode):
        self.server.mode = mode
        r = subprocess.run(["bash", str(CLAIMS / name), self.server.api, "DeepSeek-V4.1-Flash-1017"],
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def assertExit(self, name, mode, want, line):
        code, out = self.run_claim(name, mode)
        self.assertEqual(code, want, out)
        self.assertIn(line, out.strip().splitlines()[-1], out)
        return out

    # 0042: a JSON answer stays JSON
    def test_0042_holds_on_a_good_build(self):
        self.assertExit("0042-json-answer-channel.sh", "valid", 0, "CLAIM: holds")

    def test_0042_an_error_object_beside_good_choices_is_not_a_good_reply(self):
        out = self.assertExit("0042-json-answer-channel.sh", "error_envelope", 1, "CLAIM: does not hold")
        for case in ("json_schema whole", "json_schema stream", "guided_json whole"):
            self.assertRegex(out, rf"{case}: status=200 with an error .* -> FAIL")

    def test_0042_an_error_only_in_the_stream_fails_that_case(self):
        out = self.assertExit("0042-json-answer-channel.sh", "stream_error", 1, "CLAIM: does not hold")
        self.assertRegex(out, r"json_schema stream: status=200 with an error .* -> FAIL")
        self.assertRegex(out, r"json_schema whole: status=200 .* -> ok")

    def test_0042_error_null_is_no_error(self):
        self.assertExit("0042-json-answer-channel.sh", "error_null", 0, "CLAIM: holds")

    def test_0042_wrong_content_does_not_hold(self):
        self.assertExit("0042-json-answer-channel.sh", "bad_content", 1, "CLAIM: does not hold")

    # 0043: orphan DSML closers are flagged
    def test_0043_holds_on_a_good_build(self):
        self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "valid", 0, "CLAIM: holds")

    def test_0043_without_the_statistics_it_cannot_judge(self):
        # no token_sha or completion_tokens on either side: equal by absence is no evidence
        out = self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "missing_stats", 2, "CLAIM: error")
        self.assertIn("no token_sha or completion_tokens", out)

    def test_0043_different_statistics_do_not_hold(self):
        self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "stream_mismatch", 1, "CLAIM: does not hold")

    def test_0043_a_warning_only_before_the_final_event_is_not_a_pass(self):
        # the final event (finish_reason) carries no tensorfold object: neither its warning nor its token_sha is there
        out = self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "early_warning", 2, "CLAIM: error")
        self.assertIn("FAIL V4 stream: final event tool_parse_warning", out)
        self.assertNotIn("FAIL V4 whole", out)

    def test_0043_no_warning_does_not_hold(self):
        self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "missing_warning", 1, "CLAIM: does not hold")

    def test_0043_an_error_only_in_the_stream_cannot_be_judged(self):
        out = self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "stream_error", 2, "CLAIM: error")
        self.assertIn("the stream carries an error", out)

    def test_0043_error_null_is_no_error(self):
        self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "error_null", 0, "CLAIM: holds")

    def test_0043_an_error_object_beside_good_choices_cannot_be_judged(self):
        out = self.assertExit("0043-orphan-dsml-closer-diagnostic.sh", "error_envelope", 2, "CLAIM: error")
        self.assertIn("carries an error", out)



class ResponsesServer:
    """/v1/models and /v1/responses on kept HTTP/1.1 connections, as a build with 0044 answers them (a GET or DELETE
    body is read and dropped); `mode` names the one way it differs."""

    def __init__(self):
        self.mode = "valid"
        self.stored = set()
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def do_GET(self):
                with_body = bool(test.mode != "no_drain" and self.body()) or (
                    test.mode == "no_drain" and int(self.headers.get("Content-Length") or 0) > 0)
                path = self.path.removeprefix("/v1")
                if path == "/models":
                    reply = {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "tensorfold"}]}
                    if (test.mode == "body_get_error" and with_body) or (test.mode == "control_error" and not with_body):
                        reply["error"] = FAILURE
                    return self.answer(200, reply)
                if path.startswith("/responses/"):
                    rid = path.split("/")[2]
                    if rid in test.stored:
                        return self.answer(200, {"id": rid, "object": "response", "status": "completed"})
                    return self.not_stored(rid)
                self.answer(404, {"error": {"message": "no route", "type": "invalid_request_error"}})

            def do_DELETE(self):
                with_body = bool(test.mode != "no_drain" and self.body()) or (
                    test.mode == "no_drain" and int(self.headers.get("Content-Length") or 0) > 0)
                rid = self.path.removeprefix("/v1/responses/")
                if rid not in test.stored:
                    return self.not_stored(rid)
                test.stored.discard(rid)
                reply = {"id": rid, "object": "response", "deleted": True}
                if test.mode == "body_delete_error" and with_body:
                    reply["error"] = FAILURE
                self.answer(200, reply)

            def do_POST(self):
                json.loads(self.body())
                rid = "resp_" + hashlib.sha256(str(len(test.stored)).encode()).hexdigest()[:16]
                test.stored.add(rid)
                reply = {"id": rid, "object": "response", "status": "completed", "output_text": "ok"}
                if test.mode == "create_error":
                    reply["error"] = FAILURE
                self.answer(200, reply)

            def not_stored(self, rid):
                self.answer(404, {"error": {"message": f"no stored response has id '{rid}'", "type": "invalid_request_error",
                                            "param": "response_id", "code": None}})

            def answer(self, code, reply):
                if test.mode == "error_null" and code == 200:
                    reply["error"] = None
                data = json.dumps(reply).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


MODEL = "DeepSeek-V4.1-Flash-1017"
FAILURE = {"message": "the engine failed this request", "type": "server_error"}
SCRIPT_0044 = "0044-server-get-delete-body-discard.sh"


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class Claim0044(unittest.TestCase):
    def setUp(self):
        self.server = ResponsesServer()
        self.addCleanup(self.server.close)

    def assertExit(self, mode, want, line):
        self.server.mode = mode
        r = subprocess.run(["bash", str(CLAIMS / SCRIPT_0044), self.server.api, MODEL],
                           capture_output=True, text=True, timeout=120)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, want, out)
        self.assertIn(line, out.strip().splitlines()[-1], out)
        return out

    def test_holds_on_a_good_build(self):
        out = self.assertExit("valid", 0, "CLAIM: holds")
        # the expected 404s (an unknown id, and the stored one once deleted) are still what the claim promises
        self.assertRegex(out, r"DELETE unknown with body: #1 DELETE /responses/<unknown> \+ body -> 404 .*\[as promised\]")
        self.assertRegex(out, r"DELETE stored with body: #2 GET /responses/<stored> -> 404 .*\[as promised\]")

    def test_error_null_is_no_error(self):
        self.assertExit("error_null", 0, "CLAIM: holds")

    def test_a_body_left_in_the_socket_does_not_hold(self):
        out = self.assertExit("no_drain", 1, "CLAIM: does not hold")
        self.assertIn("the reply carries the previous request's body bytes", out)

    def test_an_error_object_beside_the_model_list_after_a_get_body_does_not_hold(self):
        # the bodyless controls and the create stay good; only the body-carrying GET's 200 carries an error object
        out = self.assertExit("body_get_error", 1, "CLAIM: does not hold")
        self.assertRegex(out, r"GET with body: #1 GET /models \+ body -> 200 .*\[WRONG\]")
        self.assertNotRegex(out, r"control .*\[WRONG\]")

    def test_an_error_object_beside_deleted_true_after_a_delete_body_does_not_hold(self):
        out = self.assertExit("body_delete_error", 1, "CLAIM: does not hold")
        self.assertRegex(out, r"DELETE stored with body: #1 DELETE /responses/<stored> \+ body -> 200 .*\[WRONG\]")
        self.assertRegex(out, r"GET with body: #1 GET /models \+ body -> 200 .*\[as promised\]")

    def test_an_error_object_on_a_bodyless_control_cannot_be_judged(self):
        self.assertExit("control_error", 2, "CLAIM: error")

    def test_an_error_object_on_the_create_cannot_be_judged(self):
        out = self.assertExit("create_error", 2, "CLAIM: error")
        self.assertIn("could not create a stored response (status 200 with an error)", out)


if __name__ == "__main__":
    unittest.main()
