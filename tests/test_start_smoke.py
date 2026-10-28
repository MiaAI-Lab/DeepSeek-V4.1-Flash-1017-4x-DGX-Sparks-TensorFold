"""Exercise the real launcher's smoke validation with loopback HTTP responses.

The existing ScriptTest fixture supplies fake SSH/Docker and isolated temporary
files. SCRIPTS_DIR selects the scripts under test, allowing an unchanged suite to
compare the original launcher with a fixed copy. No real cluster calls are made.
"""

import copy
import http.server
import json
from pathlib import Path
import runpy
import shlex
import shutil
import socket
import threading
import unittest


_FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_scripts.py")))
_CURL = shutil.which("curl")
_NAME = _FIXTURE["NAME"]
_COMPLETION = {
    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
    "usage": {"completion_tokens": 1},
}


@unittest.skipUnless(_FIXTURE["BASH"] and _CURL, "needs bash 4+ and curl")
class StartSmoke(unittest.TestCase):
    # Compose the old fixture instead of inheriting its unrelated test methods.
    def setUp(self):
        self.fixture = _FIXTURE["StartRestart"](methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.status = 200
        self.body = json.dumps(_COMPLETION).encode()
        self.abort_post = False
        self.hits = []
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def respond(self, status, data):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                test.hits.append(("GET", self.path))
                self.respond(200, b'{"object":"list","data":[]}')

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                test.hits.append(("POST", self.path))
                if test.abort_post:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return
                self.respond(test.status, test.body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()
        self.addCleanup(self.close_server)

        # The old Docker mock reads the later engine --name as a container name.
        # Failure paths inspect the container: fix this temporary mock to use
        # Docker's first --name, which appears before the image/engine arguments.
        docker = self.fixture.tmp / "bin" / "docker"
        text = docker.read_text()
        old = '[[ "$prev" != --name ]] || n=$a'
        self.assertEqual(text.count(old), 1)
        docker.write_text(text.replace(old, '[[ "$prev" != --name || -n "$n" ]] || n=$a'))

        # Use real HTTP status/transport semantics and permit only this listener.
        curl = self.fixture.tmp / "bin" / "curl"
        curl.write_text(
            "#!/bin/sh\n"
            "for arg in \"$@\"; do\n"
            " case \"$arg\" in\n"
            f"  http://127.0.0.1:{self.port}/*) ;;\n"
            "  http://*|https://*) echo 'non-test URL denied' >&2; exit 97 ;;\n"
            " esac\n"
            "done\n"
            f"exec {shlex.quote(_CURL)} -q --noproxy '*' \"$@\"\n"
        )
        curl.chmod(0o755)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def launch(self, extra_env=None):
        code, out = self.fixture.run_script(
            ["start.sh"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port),
             **(extra_env or {})},
            timeout=15,
        )
        self.assertEqual(sorted(self.fixture.launched()), ["head", "w1", "w2", "w3"], out)
        for host in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.fixture.left(host), [_NAME], out)
        self.assertIn(("GET", "/v1/models"), self.hits)
        self.assertIn(("POST", "/v1/chat/completions"), self.hits)
        self.fixture.assertNoTraceback(out)
        self.assertNotIn("Traceback (most recent call last)", out)
        return code, out

    def assertRejected(self, extra_env=None):
        code, out = self.launch(extra_env)
        self.assertEqual(code, 1, out)
        self.assertIn("smoke test request failed", out)
        self.assertNotIn("is now LIVE", out)

    def set_content(self, content):
        reply = copy.deepcopy(_COMPLETION)
        reply["choices"][0]["message"]["content"] = content
        self.body = json.dumps(reply).encode()

    def test_valid_assistant_completion_is_live(self):
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)
        self.assertIn("is now LIVE", out)

    def test_http_500_with_completion_body_is_rejected(self):
        self.status = 500
        self.assertRejected()

    def test_http_302_with_completion_body_is_rejected(self):
        self.status = 302
        self.assertRejected()

    def test_http_200_with_error_and_choices_is_rejected(self):
        reply = copy.deepcopy(_COMPLETION)
        reply["error"] = {"message": "controlled error", "type": "server_error"}
        self.body = json.dumps(reply).encode()
        self.assertRejected()

    def test_http_500_is_rejected_with_python_optimize_1(self):
        self.status = 500
        self.assertRejected({"PYTHONOPTIMIZE": "1"})

    def test_error_and_choices_are_rejected_with_python_optimize_2(self):
        reply = copy.deepcopy(_COMPLETION)
        reply["error"] = {"message": "controlled error", "type": "server_error"}
        self.body = json.dumps(reply).encode()
        self.assertRejected({"PYTHONOPTIMIZE": "2"})

    def test_valid_completion_is_live_with_python_optimize_2(self):
        code, out = self.launch({"PYTHONOPTIMIZE": "2"})
        self.assertEqual(code, 0, out)
        self.assertIn("OK:", out)
        self.assertIn("is now LIVE", out)

    def test_empty_content_is_rejected(self):
        self.set_content("")
        self.assertRejected()

    def test_whitespace_content_is_rejected(self):
        self.set_content(" \n\t ")
        self.assertRejected()

    def test_missing_content_is_rejected(self):
        reply = copy.deepcopy(_COMPLETION)
        del reply["choices"][0]["message"]["content"]
        self.body = json.dumps(reply).encode()
        self.assertRejected()

    def test_array_content_is_rejected(self):
        self.set_content(["OK"])
        self.assertRejected()

    def test_number_content_is_rejected(self):
        self.set_content(123)
        self.assertRejected()

    def test_wrong_message_role_is_rejected(self):
        reply = copy.deepcopy(_COMPLETION)
        reply["choices"][0]["message"]["role"] = "user"
        self.body = json.dumps(reply).encode()
        self.assertRejected()

    def test_malformed_json_is_rejected(self):
        self.body = b'{"choices":'
        self.assertRejected()

    def test_transport_failure_is_rejected(self):
        self.abort_post = True
        self.assertRejected()

    def test_long_single_character_reply_remains_rejected(self):
        self.set_content("A" * 8)
        self.assertRejected()

    def test_short_single_character_reply_remains_accepted(self):
        self.set_content("A" * 7)
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)


if __name__ == "__main__":
    unittest.main()
