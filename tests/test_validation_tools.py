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


def check_line(out, name):
    return next((l for l in out.splitlines() if name in l and l[:4] in ("PASS", "FAIL")), "")


class ToolTest(unittest.TestCase):
    def assertPassed(self, code, out, name):
        self.assertTrue(check_line(out, name).startswith("PASS"), out)
        self.assertEqual(code, 0, out)

    def assertFailedCleanly(self, code, out, name):
        """A deliberate FAIL of check <name>: its FAIL line, exit code 1, and no traceback (not a crash)."""
        self.assertTrue(check_line(out, name).startswith("FAIL"), out)
        self.assertEqual(code, 1, out)
        self.assertNotIn("Traceback", out)


# ---------------------------------------------------------------- validate.py

def validate_server(overrides=None, debug=None, features=None):
    """A server that answers every validate.py check correctly; overrides maps a test prompt to its reply text or a
    whole (status, body) answer, debug is the (status, body) answer to the TF_DEBUG_PROMPT_FILE prompt, and
    features(status, body) -> (status, body) rewrites the answers to the feature checks (text to long prompt)."""
    overrides = overrides or {}

    def respond(body):
        msgs = body["messages"]
        c = msgs[-1].get("content")
        answer = feature_answer(body)
        if answer is not None:
            return features(*answer) if features else answer
        if isinstance(c, str) and DEBUG_MARK in c:
            return debug or completion("Debug reply.")
        if c in PROMPTS:
            o = overrides.get(c, reply_to(c))
            return o if isinstance(o, tuple) else completion(o)
        return 400, {"error": {"message": "unexpected prompt"}}

    return respond


def feature_answer(body):
    """The correct answer to one of validate.py's feature checks, or None for any other prompt."""
    msgs = body["messages"]
    last = msgs[-1]
    c = last.get("content")
    if isinstance(c, list):
        kinds = {part.get("type") for part in c}
        return completion("Red at first, then blue." if "video_url" in kinds else "A yellow circle.")
    if last["role"] == "tool":
        return completion("It is 21 degrees and clear in Tokyo.")
    if DEBUG_MARK in c:
        return None
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
    return None


class ValidateCompare(ToolTest):
    NAME = "equal to the saved replies"

    def run_compare(self, ref, overrides=None, raw=None):
        m = Mock(validate_server(overrides))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(raw if raw is not None else json.dumps(ref))
            return run_tool("validate.py", [m.url, "--compare", f.name])
        finally:
            m.close()
            os.unlink(f.name)

    def assertCompare(self, ok, ref=None, overrides=None, raw=None, why=None):
        code, out = self.run_compare(ref, overrides, raw)
        if ok:
            self.assertPassed(code, out, self.NAME)
        else:
            self.assertFailedCleanly(code, out, self.NAME)
        if why:
            self.assertIn(why, check_line(out, self.NAME))

    def test_full_correct_reference_passes(self):
        self.assertCompare(True, [reply_to(p) for p in PROMPTS])

    def test_empty_reference_fails(self):
        self.assertCompare(False, [], why="holds 0 replies, not 8")

    def test_short_reference_fails(self):
        self.assertCompare(False, [reply_to(PROMPTS[0])], why="holds 1 replies, not 8")

    def test_long_reference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS] + ["one more"])

    def test_reference_not_a_list_fails(self):
        self.assertCompare(False, {reply_to(p): 1 for p in PROMPTS})

    def test_reference_null_fails(self):
        self.assertCompare(False, raw="null")

    def test_reference_with_non_string_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [None], why="empty or non-text reply")

    def test_reference_not_json_fails(self):
        self.assertCompare(False, raw="not json")

    def test_missing_reference_file_fails(self):
        m = Mock(validate_server())
        try:
            code, out = run_tool("validate.py", [m.url, "--compare", os.path.join(tempfile.gettempdir(), "no-such-ref.json")])
        finally:
            m.close()
        self.assertFailedCleanly(code, out, self.NAME)
        self.assertIn("cannot read", check_line(out, self.NAME))

    def test_reordered_reference_fails(self):
        # the same replies in another order: each reply is compared with its own prompt's saved one
        self.assertCompare(False, [reply_to(p) for p in reversed(PROMPTS)], why="0 of 8 equal")

    def test_wrong_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + ["something else"])

    def test_whitespace_difference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [reply_to(PROMPTS[-1]) + " "])

    def test_line_ending_difference_fails(self):
        ref = [reply_to(p) for p in PROMPTS]
        ref[0] += "\n"
        self.assertCompare(False, ref, overrides={PROMPTS[0]: reply_to(PROMPTS[0]) + "\r\n"})

    def test_case_difference_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [reply_to(PROMPTS[-1]).upper()])

    def test_empty_reference_reply_fails(self):
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""], why="empty or non-text reply")

    def test_empty_replies_on_both_sides_fail(self):
        # the server's reply and the saved one are both empty: equal, but not a reply
        self.assertCompare(False, [reply_to(p) for p in PROMPTS[:-1]] + [""], overrides={PROMPTS[-1]: ""})

    def test_error_marked_replies_fail(self):
        # every reply carries correct-looking choices and an error: an error is never a reply
        bad = {p: (200, {**completion(reply_to(p))[1], "error": {"message": "engine failed"}}) for p in PROMPTS}
        self.assertCompare(False, [reply_to(p) for p in PROMPTS], overrides=bad)


class ValidateConcurrent(ToolTest):
    NAME = "concurrent == solo"

    def run_validate(self, overrides):
        m = Mock(validate_server(overrides))
        try:
            return run_tool("validate.py", [m.url])
        finally:
            m.close()

    def test_all_replies_pass(self):
        self.assertPassed(*self.run_validate({}), self.NAME)

    def test_empty_reply_fails_without_compare(self):
        self.assertFailedCleanly(*self.run_validate({PROMPTS[3]: ""}), self.NAME)

    def test_error_marked_reply_fails_without_compare(self):
        bad = {PROMPTS[3]: (200, {**completion(reply_to(PROMPTS[3]))[1], "error": {"message": "engine failed"}})}
        self.assertFailedCleanly(*self.run_validate(bad), self.NAME)


class ValidateFeatures(ToolTest):
    CHECKS = ["text", "thinking", "tool call", "image", "long prompt"]

    def run_validate(self, features):
        m = Mock(validate_server(features=features))
        try:
            return run_tool("validate.py", [m.url])
        finally:
            m.close()

    def assertFeaturesFail(self, features, checks=None):
        code, out = self.run_validate(features)
        for name in checks or self.CHECKS:
            self.assertTrue(check_line(out, name).startswith("FAIL"), f"{name}:\n{out}")
        self.assertEqual(code, 1, out)
        self.assertNotIn("Traceback", out)

    def test_correct_features_pass(self):
        code, out = self.run_validate(None)
        for name in self.CHECKS + ["tool result"]:
            self.assertTrue(check_line(out, name).startswith("PASS"), f"{name}:\n{out}")
        self.assertEqual(code, 0, out)

    def test_error_marked_feature_replies_fail(self):
        # correct-looking replies that also carry an error: an error is never a reply
        self.assertFeaturesFail(lambda st, b: (st, {**b, "error": {"message": "engine failed"}}))

    def test_wrong_role_feature_replies_fail(self):
        def user_role(st, b):
            msg = {**b["choices"][0]["message"], "role": "user"}
            return st, {**b, "choices": [{"message": msg}]}
        self.assertFeaturesFail(user_role)

    def test_malformed_tool_call_fails_cleanly(self):
        def broken_call(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                return st, {**b, "choices": [{"message": {**msg, "tool_calls": [{"function": None}]}}]}
            return st, b
        self.assertFeaturesFail(broken_call, ["tool call"])

    def test_tool_call_arguments_not_json_fail_cleanly(self):
        def bad_args(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                call = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{city: Tokyo"}}
                return st, {**b, "choices": [{"message": {**msg, "tool_calls": [call]}}]}
            return st, b
        self.assertFeaturesFail(bad_args, ["tool call"])

    def test_malformed_tool_calls_fail(self):
        good = {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}
        cases = {
            "type not function": [{**good, "type": "not-a-function"}],
            "no id": [{k: v for k, v in good.items() if k != "id"}],
            "empty id": [{**good, "id": ""}],
            "no type": [{k: v for k, v in good.items() if k != "type"}],
            "a null call after a good one": [good, None],
            "arguments not an object": [{**good, "function": {"name": "get_weather", "arguments": "[\"Tokyo\"]"}}],
            "city not a string": [{**good, "function": {"name": "get_weather", "arguments": "{\"city\": [\"Tokyo\"]}"}}],
            "a nameless call after a good one": [good, {**good, "id": "call_1", "function": {"name": "", "arguments": "{}"}}],
            "a name that is not a string": [good, {**good, "id": "call_1", "function": {"name": 7, "arguments": "{}"}}],
            "a call to a tool not offered after a good one": [good, {**good, "id": "call_1", "function": {"name": "nonexistent_tool", "arguments": "{\"city\": \"Oslo\"}"}}],
            "a second call without the required city": [good, {**good, "id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}],
            "NaN in the arguments": [{**good, "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\", \"x\": NaN}"}}],
        }
        for case, calls in cases.items():
            with self.subTest(case=case):
                def bad(st, b, calls=calls):
                    msg = b["choices"][0]["message"]
                    if msg.get("tool_calls"):
                        return st, {**b, "choices": [{"message": {**msg, "tool_calls": calls}}]}
                    return st, b
                self.assertFeaturesFail(bad, ["tool call"])

    def test_tool_call_with_null_content_passes(self):
        def null_content(st, b):
            msg = b["choices"][0]["message"]
            if msg.get("tool_calls"):
                return st, {**b, "choices": [{"message": {**msg, "content": None}}]}
            return st, b
        code, out = self.run_validate(null_content)
        self.assertTrue(check_line(out, "tool call").startswith("PASS"), out)
        self.assertEqual(code, 0, out)


class ValidateArguments(unittest.TestCase):
    def test_option_without_a_value_stops(self):
        # every case names a dead URL first: a tool that misreads its options must not reach a live server
        dead = "http://127.0.0.1:9"
        for argv in ([dead, "--model"], [dead, "--compare"], [dead, "--save"],
                     [dead, "--model", "--compare", "f.json"]):
            with self.subTest(args=argv):
                code, out = run_tool("validate.py", argv)
                self.assertNotEqual(code, 0, out)
                self.assertIn("needs a value", out)
                self.assertNotIn("Traceback", out)
                self.assertNotIn("PASS", out)


class ValidateDebugPrompt(ToolTest):
    NAME = "debug prompt"

    def run_debug(self, debug):
        m = Mock(validate_server(debug=debug))
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(f"{DEBUG_MARK} a long debug prompt")
            return run_tool("validate.py", [m.url], {"TF_DEBUG_PROMPT_FILE": f.name})
        finally:
            m.close()
            os.unlink(f.name)

    def test_valid_reply_passes(self):
        self.assertPassed(*self.run_debug(completion("Debug reply.")), self.NAME)

    def test_http_error_fails(self):
        self.assertFailedCleanly(*self.run_debug((500, {"error": {"message": "internal error"}})), self.NAME)

    def test_empty_content_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion("")), self.NAME)

    def test_non_text_content_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion(42)), self.NAME)

    def test_null_choice_fails(self):
        self.assertFailedCleanly(*self.run_debug((200, {"choices": [None]})), self.NAME)

    def test_non_object_body_fails(self):
        self.assertFailedCleanly(*self.run_debug((200, [])), self.NAME)

    def test_wrong_role_fails(self):
        self.assertFailedCleanly(*self.run_debug(completion("Debug reply.", role="user")), self.NAME)


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
        if mode == "error_with_choices":
            return 200, {"choices": [{"message": {"role": "assistant", "content": code}}], "usage": usage,
                         "error": {"message": "engine failed"}}
        if mode in ("number_content", "list_content", "object_content"):
            content = {"number_content": int(code), "list_content": [code], "object_content": {"text": code}}[mode]
            return 200, {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}
        if mode == "wrong_role":
            return 200, {"choices": [{"message": {"role": "user", "content": code}}], "usage": usage}
        if mode == "list_body":
            return 200, [code]
        raise ValueError(mode)

    return respond


class Needle(unittest.TestCase):
    def run_needle(self, mode, extra=(), **kw):
        m = Mock(needle_server(mode, **kw))
        try:
            # extra first: needle_bench.py reads an option's first occurrence
            code, out = run_tool("needle_bench.py", ["--url", m.url, *extra, "--lens", "64,128", "--depths", "0,50,100"])
        finally:
            m.close()
        return code, out

    def assertAllFail(self, mode):
        """Every cell fails, reported as such: exit code 1, 0/6, no cell ok, no traceback."""
        code, out = self.run_needle(mode)
        self.assertEqual(code, 1, out)
        self.assertIn("0/6 passed", out)
        self.assertNotIn("ok=True", out)
        self.assertNotIn("Traceback", out)

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

    def test_error_with_choices_fails(self):
        self.assertAllFail("error_with_choices")

    def test_missing_choices_fails(self):
        self.assertAllFail("no_choices")

    def test_empty_choices_fails(self):
        self.assertAllFail("empty_choices")

    def test_null_content_fails(self):
        self.assertAllFail("null_content")

    def test_number_content_fails(self):
        self.assertAllFail("number_content")

    def test_list_content_fails(self):
        self.assertAllFail("list_content")

    def test_object_content_fails(self):
        self.assertAllFail("object_content")

    def test_wrong_role_fails(self):
        self.assertAllFail("wrong_role")

    def test_non_object_body_fails(self):
        self.assertAllFail("list_body")

    def test_error_diagnostics_kept(self):
        # a failed cell says why, on the console and in the --json row, with the server's own error
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            code, out = self.run_needle("http400_echo", extra=("--json", path))
            with open(path) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(code, 1, out)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertFalse(row["ok"])
            self.assertEqual(row["reply"], "")
            self.assertTrue(row["error"].startswith("error: "), row)
            self.assertIn("invalid request", row["error"])
        self.assertEqual(out.count("error: "), 6, out)

    def test_no_reply_diagnostics_kept(self):
        # a reply without choices: the cell says so, with what the server sent
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            code, out = self.run_needle("no_choices", extra=("--json", path))
            with open(path) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(code, 1, out)
        for row in rows:
            self.assertTrue(row["error"].startswith("no reply text: "), row)
            self.assertIn("no choices, code", row["error"])
        self.assertEqual(out.count("no reply text: "), 6, out)

    def assertCalibrationStops(self, calib):
        code, out = self.run_needle("correct", calib=calib)
        self.assertEqual(code, 1, out)
        self.assertIn("Calibration request failed", out)
        self.assertNotIn("passed", out)
        self.assertNotIn("Traceback", out)

    def test_failed_calibration_stops(self):
        self.assertCalibrationStops((400, {"error": {"message": "bad"}}))

    def test_error_marked_calibration_stops(self):
        self.assertCalibrationStops((200, {"error": {"message": "engine failed"}, "usage": {"prompt_tokens": 1}}))

    def test_implausible_calibration_counts_stop(self):
        for n in (1, -3200, 0, True, "3200", 3200.5):
            with self.subTest(prompt_tokens=n):
                self.assertCalibrationStops((200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                                                   "usage": {"prompt_tokens": n}}))

    def test_bad_grid_arguments_stop(self):
        for extra in (("--reps", "0"), ("--reps", "-1"), ("--depths", "150"), ("--depths", "-5"), ("--lens", "0"),
                      ("--lens", "abc"), ("--reps", "x")):
            with self.subTest(args=extra):
                code, out = self.run_needle("correct", extra=extra)
                self.assertNotEqual(code, 0, out)
                self.assertNotIn("passed", out)
                self.assertNotIn("Traceback", out)

    def test_option_without_a_value_stops(self):
        # each case but --url's own names a dead URL first: a tool that misreads its options must not reach a live
        # server (the default grid runs to 1M tokens)
        dead = ["--url", "http://127.0.0.1:9"]
        for argv in (dead + ["--lens"], dead + ["--depths"], dead + ["--reps"], dead + ["--model"], ["--url"],
                     dead + ["--reps", "--lens", "64"]):
            with self.subTest(args=argv):
                code, out = run_tool("needle_bench.py", argv)
                self.assertNotEqual(code, 0, out)
                self.assertIn("needs a value", out)
                self.assertNotIn("Traceback", out)
                self.assertNotIn("passed", out)


if __name__ == "__main__":
    unittest.main()
