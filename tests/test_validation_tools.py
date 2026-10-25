"""Offline regression tests for tools/validate.py and tools/needle_bench.py: each test starts a mock OpenAI-compatible
server on 127.0.0.1 and runs the real tool against it as a subprocess, checking its output and exit code. No GPU,
weights, Docker or Sparks needed.

    python3 -m unittest discover -s tests -v
    TOOLS_DIR=/path/to/other/tools python3 -m unittest discover -s tests -v   # run the same tests against other tools
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = os.environ.get("TOOLS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
PROMPTS = ["Write a haiku about the sea.", "List the first ten prime numbers.", "Explain TCP in two sentences.",
           "Write a Python function that reverses a string.", "Name five rivers in Europe.",
           "What causes the seasons on Earth? Two sentences.", "Translate 'good morning' into French, Spanish and German.",
           "Give three tips for writing clear documentation."]
DEBUG_MARK = "DEBUG-PROMPT-MARK"


def reply_to(p):
    return f"Reply to: {p}"


def completion(content, **message):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content, **message}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class Mock:
    """A mock server whose replies come from respond(body) -> (status, json object or str)."""

    def __init__(self, respond):
        mock = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                status, out = mock.respond(body)
                data = (out if isinstance(out, str) else json.dumps(out)).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.respond = respond
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def run_tool(name, args, env=None):
    e = {k: v for k, v in os.environ.items() if k != "TF_DEBUG_PROMPT_FILE"}
    e.update(env or {})
    p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, name), *args], capture_output=True, text=True,
                       timeout=120, env=e)
    return p.returncode, p.stdout + p.stderr


# ---------------------------------------------------------------- validate.py

def validate_server(overrides=None, debug=None):
    """A server that answers every validate.py check correctly; overrides maps a test prompt to its reply text,
    debug is the (status, body) answer to the TF_DEBUG_PROMPT_FILE prompt."""
    overrides = overrides or {}

    def respond(body):
        msgs = body["messages"]
        last = msgs[-1]
        c = last.get("content")
        if isinstance(c, list):
            kinds = {part.get("type") for part in c}
            return completion("Red at first, then blue." if "video_url" in kinds else "A yellow circle.")
        if last["role"] == "tool":
            return completion("It is 21 degrees and clear in Tokyo.")
        if DEBUG_MARK in c:
            return debug or completion("Debug reply.")
        if "capital of France" in c:
            return completion("Paris")
        if "17 * 23" in c:
            return completion("391", reasoning_content="17 * 23 = 391.")
        if "weather in Tokyo" in c:
            # a valid tool call: empty content, the call in tool_calls
            return completion("", tool_calls=[{"id": "call_0", "type": "function",
                                               "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}])
        if "item 1234" in c:
            return completion("8638")
        if c in PROMPTS:
            return completion(overrides.get(c, reply_to(c)))
        return 400, {"error": {"message": "unexpected prompt"}}

    return respond


class ValidateCompare(unittest.TestCase):
    def run_compare(self, ref, overrides=None, raw=None):
        m = Mock(validate_server(overrides))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(raw if raw is not None else json.dumps(ref))
            code, out = run_tool("validate.py", [m.url, "--compare", f.name])
        finally:
            m.close()
            os.unlink(f.name)
        line = next((l for l in out.splitlines() if "equal to the saved replies" in l), "")
        return code, out, line

    def assertCompare(self, ok, ref=None, overrides=None, raw=None):
        code, out, line = self.run_compare(ref, overrides, raw)
        if ok:
            self.assertTrue(line.startswith("PASS"), out)
            self.assertEqual(code, 0, out)
        else:
            self.assertFalse(line.startswith("PASS"), out)
            self.assertNotEqual(code, 0, out)

    def test_full_correct_reference_passes(self):
        self.assertCompare(True, [reply_to(p) for p in PROMPTS])

    def test_empty_reference_fails(self):
        self.assertCompare(False, [])

    def test_short_reference_fails(self):
        self.assertCompare(False, [reply_to(PROMPTS[0])])

    def test_long_reference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS] + ["one more"])

    def test_reference_not_a_list_fails(self):
        self.assertCompare(False, {reply_to(p): 1 for p in PROMPTS})

    def test_reference_with_non_string_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [None])

    def test_reference_not_json_fails(self):
        self.assertCompare(False, raw="not json")

    def test_wrong_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + ["something else"])

    def test_empty_reference_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""])

    def test_empty_replies_on_both_sides_fail(self):
        # the server's reply and the saved one are both empty: equal, but not a reply
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""], overrides={PROMPTS[-1]: ""})


class ValidateDebugPrompt(unittest.TestCase):
    def run_debug(self, debug):
        m = Mock(validate_server(debug=debug))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(f"{DEBUG_MARK} a long debug prompt")
            code, out = run_tool("validate.py", [m.url], {"TF_DEBUG_PROMPT_FILE": f.name})
        finally:
            m.close()
            os.unlink(f.name)
        line = next((l for l in out.splitlines() if "debug prompt" in l), "")
        return code, out, line

    def test_valid_reply_passes(self):
        code, out, line = self.run_debug(completion("Debug reply."))
        self.assertTrue(line.startswith("PASS"), out)
        self.assertEqual(code, 0, out)

    def test_http_error_fails(self):
        code, out, line = self.run_debug((500, {"error": {"message": "internal error"}}))
        self.assertFalse(line.startswith("PASS"), out)
        self.assertNotEqual(code, 0, out)

    def test_empty_content_fails(self):
        code, out, line = self.run_debug(completion(""))
        self.assertFalse(line.startswith("PASS"), out)
        self.assertNotEqual(code, 0, out)


# ---------------------------------------------------------------- needle_bench.py

def needle_server(mode, calib=(200, None)):
    """Calibration answers normally (unless calib says otherwise); every needle prompt is answered by mode."""

    def respond(body):
        prompt = body["messages"][-1]["content"]
        if prompt.endswith("Say OK."):
            status, out = calib
            return (status, out) if out is not None else (200, {
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 3200, "completion_tokens": 1}})
        code = re.search(r"The secret code is: (\d+)\.", prompt).group(1)
        usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": 3}
        if mode == "correct":
            return 200, {"choices": [{"message": {"role": "assistant", "content": code}}], "usage": usage}
        if mode == "wrong":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "000000"}}], "usage": usage}
        if mode == "http400_echo":
            return 400, {"error": {"message": "invalid request", "prompt": prompt}}
        if mode == "error_object":
            return 200, {"error": {"message": f"engine failed on code {code}"}}
        if mode == "no_choices":
            return 200, {"usage": usage, "detail": f"no choices, code {code}"}
        if mode == "empty_choices":
            return 200, {"choices": [], "usage": usage, "detail": code}
        if mode == "null_content":
            return 200, {"choices": [{"message": {"role": "assistant", "content": None}}], "usage": usage, "detail": code}
        raise ValueError(mode)

    return respond


class Needle(unittest.TestCase):
    def run_needle(self, mode, **kw):
        m = Mock(needle_server(mode, **kw))
        try:
            code, out = run_tool("needle_bench.py", ["--url", m.url, "--lens", "64,128", "--depths", "0,50,100"])
        finally:
            m.close()
        return code, out

    def assertAllFail(self, mode):
        code, out = self.run_needle(mode)
        self.assertNotEqual(code, 0, out)
        self.assertIn("0/6 passed", out)
        self.assertNotIn("ok=True", out)

    def test_correct_code_passes(self):
        code, out = self.run_needle("correct")
        self.assertEqual(code, 0, out)
        self.assertIn("6/6 passed", out)

    def test_wrong_code_fails(self):
        self.assertAllFail("wrong")

    def test_http_400_echoing_the_code_fails(self):
        self.assertAllFail("http400_echo")

    def test_error_object_fails(self):
        self.assertAllFail("error_object")

    def test_missing_choices_fails(self):
        self.assertAllFail("no_choices")

    def test_empty_choices_fails(self):
        self.assertAllFail("empty_choices")

    def test_null_content_fails(self):
        self.assertAllFail("null_content")

    def test_failed_calibration_stops(self):
        code, out = self.run_needle("correct", calib=(400, {"error": {"message": "bad"}}))
        self.assertNotEqual(code, 0, out)
        self.assertIn("Calibration request failed", out)


if __name__ == "__main__":
    unittest.main()
