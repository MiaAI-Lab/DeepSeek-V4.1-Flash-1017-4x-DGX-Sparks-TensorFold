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
        # /health: the launcher reads the engine's effective cache account from it (absent here, as an old engine's)
        self.health = {"status": "ok", "live": {"connections": 0, "waiting": 0}}
        self.health_status = 200
        self.health_failures = 0
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
                if self.path == "/health":
                    if test.health_failures:
                        test.health_failures -= 1
                        self.respond(500, b"{}")
                    else:
                        self.respond(test.health_status, json.dumps(test.health).encode())
                else:
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

    def launch(self, extra_env=None, infer=True, args=None):
        code, out = self.fixture.run_script(
            ["start.sh", *(args or [])],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port),
             **(extra_env or {})},
            timeout=15,
        )
        self.assertEqual(sorted(self.fixture.launched()), ["head", "w1", "w2", "w3"], out)
        for host in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.fixture.left(host), [_NAME], out)
        self.assertIn(("GET", "/v1/models"), self.hits)
        self.assertIn(("GET", "/health"), self.hits)
        if infer:  # a refusal before the smoke test makes no inference request
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


    def cache_health(self, **cache):
        """The engine's effective cache account, as the instrumented engine reports it for this launcher's settings."""
        body = {
            "schema_version": 1,
            "engine_epoch": 1793692800000000000,
            "observed_unix_ns": 1793692801000000000,
            "consistent": True,
            "context_limit_positions": 1048576,
            "stream_capacity": 16,
            "pool": {
                "capacity_positions": 2097152,
                "active_reserved_positions": 0,
                "retained_prefix_positions": 0,
                "retained_growth_positions": 0,
                "unclaimed_positions": 2097152,
                "valid_states": 0,
                "invalidations_total": 0,
                "placement_blocks_total": 0,
            },
            "prompt_state": {
                "enabled": True,
                "budget_bytes": 536870912,
                "held_bytes": 0,
                "spare_bytes": 0,
                "entries": 0,
            },
        }
        body.update(cache)
        self.health = {"status": "ok", "live": {"connections": 0, "waiting": 0, "cache": body}}
        return body

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

    def test_cache_telemetry_reports_the_effective_configuration(self):
        self.cache_health()
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("Cache telemetry: shared pool 2,097,152 positions", out)
        self.assertIn("per request 1,048,576", out)
        self.assertIn("snapshots 512 MiB", out)
        self.assertIn("is now LIVE", out)

    def test_an_old_engine_reports_unavailable_not_a_mismatch(self):
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("Cache telemetry: unavailable", out)
        self.assertIn("is now LIVE", out)

    def test_a_pool_mismatch_is_a_warning_by_default(self):
        self.cache_health()
        self.health["live"]["cache"]["pool"]["capacity_positions"] = 8388608
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("WARN:", out)
        self.assertIn("cache telemetry check failed", out)
        self.assertIn("pool 8388608 != 2097152", out)
        self.assertIn("CACHE_STRICT=1", out)
        self.assertIn("is now LIVE", out)

    def test_cache_strict_makes_an_inconsistent_account_fatal(self):
        self.cache_health(consistent=False)
        code, out = self.launch({"CACHE_STRICT": "1"}, infer=False)
        self.assertEqual(code, 1, out)
        self.assertIn("inconsistent accounting", out)
        self.assertIn("CACHE_STRICT=1", out)
        self.assertNotIn("is now LIVE", out)

    def test_a_disabled_store_with_a_configured_budget_is_a_warning(self):
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"] = {"enabled": False, "budget_bytes": 0}
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("the prompt-state store is disabled", out)
        self.assertIn("is now LIVE", out)

    def test_one_failed_health_read_is_retried(self):
        """A server that is serving but answers /health once with a 5xx does not fail the launch."""
        self.cache_health()
        self.health_failures = 1
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("Cache telemetry: shared pool 2,097,152 positions", out)
        self.assertEqual(self.hits.count(("GET", "/health")), 2, out)

    # ─── operator arguments are what the engine actually runs ────────────
    def test_a_prompt_cache_override_is_the_expected_budget(self):
        """`start.sh --prompt-cache-gib 1` wins over the 0.5 GiB default, so the check must expect 1 GiB."""
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 1 << 30
        code, out = self.launch(args=["--prompt-cache-gib", "1"])
        self.assertEqual(code, 0, out)
        self.assertIn("snapshots 1024 MiB", out)
        self.assertNotIn("MISMATCH", out)
        self.assertIn("is now LIVE", out)

    def test_the_last_prompt_cache_override_wins(self):
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 1 << 30
        code, out = self.launch(args=["--prompt-cache-gib", "0.25", "--prompt-cache-gib", "1"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("MISMATCH", out)

    def test_a_prompt_cache_override_in_equals_form_is_read(self):
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 1 << 30
        code, out = self.launch(args=["--prompt-cache-gib=1"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("MISMATCH", out)

    def test_a_context_override_is_the_expected_window(self):
        self.cache_health()
        self.health["live"]["cache"]["context_limit_positions"] = 262144
        code, out = self.launch(args=["--context", "262144"])
        self.assertEqual(code, 0, out)
        self.assertIn("per request 262,144", out)
        self.assertNotIn("MISMATCH", out)

    def test_a_parallel_override_is_the_expected_stream_count(self):
        self.cache_health()
        self.health["live"]["cache"]["stream_capacity"] = 8
        code, out = self.launch(args=["--parallel", "8"])
        self.assertEqual(code, 0, out)
        self.assertIn("8 streams", out)
        self.assertNotIn("MISMATCH", out)

    def test_an_engine_chosen_context_is_reported_not_compared(self):
        """`--context 0` means the model's window: the engine resolves it, so a matching report cannot disagree."""
        self.cache_health()
        code, out = self.launch(args=["--context", "0"])
        self.assertEqual(code, 0, out)
        self.assertIn("context 1,048,576 (the model's window)", out)
        self.assertNotIn("MISMATCH", out)

    def test_an_engine_chosen_parallel_is_reported_not_compared(self):
        self.cache_health()
        self.health["live"]["cache"]["stream_capacity"] = 7
        code, out = self.launch(args=["--parallel", "auto"])
        self.assertEqual(code, 0, out)
        self.assertIn("7 streams (engine-selected)", out)
        self.assertNotIn("MISMATCH", out)

    def test_an_override_past_the_pool_is_refused_before_any_container(self):
        code, out = self.fixture.run_script(
            ["start.sh", "--context", "4194304"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("past the model's 1,048,576-token window", out)
        self.assertEqual(self.fixture.launched(), [], out)
        self.assertNotIn(("GET", "/health"), self.hits)

    def test_an_override_above_context_is_refused_before_any_container(self):
        """The followers are launched with CONTEXT and size their buffers for it (ctx_cap); a wider rank-0 window is not theirs."""
        code, out = self.fixture.run_script(
            ["start.sh", "--context", "524288"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port), "POOL": "2097152", "CONTEXT": "262144"},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("above CONTEXT (262144)", out)
        self.assertEqual(self.fixture.launched(), [], out)

    def test_an_engine_chosen_context_above_a_narrower_context_is_refused_before_any_container(self):
        """`--context 0` asks for the model's 1,048,576-token window, which the followers were not sized for either."""
        code, out = self.fixture.run_script(
            ["start.sh", "--context", "0"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port), "POOL": "2097152", "CONTEXT": "262144"},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("above CONTEXT (262144)", out)
        self.assertIn("set CONTEXT=1048576", out)
        self.assertEqual(self.fixture.launched(), [], out)

    def test_a_parallel_auto_below_the_engines_eight_streams_is_refused_before_any_container(self):
        """auto is 8 streams to the engine and this backend never lowers them by memory: 4 slots would meet NoFreeSlot."""
        code, out = self.fixture.run_script(
            ["start.sh", "--parallel", "auto"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port), "PARALLEL": "4"},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("asks the engine for its 8 streams", out)
        self.assertEqual(self.fixture.launched(), [], out)

    def test_a_fractional_budget_is_the_bytes_the_engine_truncates_to(self):
        """0.2 GiB is 214748364 bytes to the engine's truncation, not the 214748365 a round would expect."""
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 214748364
        code, out = self.launch(args=["--prompt-cache-gib", "0.2"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("MISMATCH", out)
        self.assertIn("is now LIVE", out)

    def test_a_rounded_fractional_budget_is_a_mismatch(self):
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 214748365
        code, out = self.launch({"CACHE_STRICT": "1"}, infer=False, args=["--prompt-cache-gib", "0.2"])
        self.assertEqual(code, 1, out)
        self.assertIn("snapshot budget 214748365 != 214748364", out)
        self.assertNotIn("is now LIVE", out)

    def test_a_parallel_override_of_zero_is_refused_before_any_container(self):
        """The engine normalizes 0 to one stream, so a launch expecting 0 could never match its report."""
        code, out = self.fixture.run_script(
            ["start.sh", "--parallel", "0"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("--parallel 0", out)
        self.assertIn("the engine reads 0 as 1", out)
        self.assertEqual(self.fixture.launched(), [], out)
        self.assertNotIn(("GET", "/health"), self.hits)

    def test_a_parallel_override_above_the_launch_count_is_refused_before_any_container(self):
        code, out = self.fixture.run_script(
            ["start.sh", "--parallel", "8"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port), "PARALLEL": "4"},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("is above PARALLEL (4)", out)
        self.assertEqual(self.fixture.launched(), [], out)

    def test_a_positive_budget_below_one_byte_is_refused_before_any_container(self):
        code, out = self.fixture.run_script(
            ["start.sh", "--prompt-cache-gib", "0.0000000001"],
            {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
            timeout=15,
        )
        self.assertEqual(code, 1, out)
        self.assertIn("below one byte", out)
        self.assertEqual(self.fixture.launched(), [], out)

    def test_a_flag_without_a_value_is_refused_before_any_container(self):
        for flag in ["--context=", "--parallel=", "--prompt-cache-gib="]:
            with self.subTest(flag=flag):
                code, out = self.fixture.run_script(
                    ["start.sh", flag],
                    {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
                    timeout=15,
                )
                self.assertEqual(code, 1, out)
                self.assertIn("needs a value", out)
                self.assertEqual(self.fixture.launched(), [], out)
        for flag in ["--context", "--parallel", "--prompt-cache-gib"]:  # a trailing flag has no value either
            with self.subTest(flag=flag):
                code, out = self.fixture.run_script(
                    ["start.sh", flag],
                    {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
                    timeout=15,
                )
                self.assertEqual(code, 1, out)
                self.assertIn("needs a value", out)
                self.assertEqual(self.fixture.launched(), [], out)

    def test_a_non_decimal_count_is_refused_before_any_container(self):
        """0100 is octal 64 to Bash and 100 to the engine; both readings are wrong to launch with either way."""
        for value in ["0100", "99999999999999999999999", "-1", "1 00"]:
            with self.subTest(value=value):
                code, out = self.fixture.run_script(
                    ["start.sh", f"--context={value}"],
                    {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
                    timeout=15,
                )
                self.assertEqual(code, 1, out)
                self.assertIn("a plain decimal token count", out)
                self.assertEqual(self.fixture.launched(), [], out)

    def test_a_dotted_count_is_refused_before_any_container(self):
        for value in ["010000", "1e6"]:
            with self.subTest(value=value):
                code, out = self.fixture.run_script(
                    ["start.sh", f"--parallel={value}"],
                    {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
                    timeout=15,
                )
                self.assertEqual(code, 1, out)
                self.assertIn("a plain decimal stream count", out)
                self.assertEqual(self.fixture.launched(), [], out)

    def assertHealthErrorRejected(self, error):
        """The launcher's test is `is not None`, as step 5's is: a falsey error object is still an error."""
        self.cache_health()
        self.health["error"] = error
        code, out = self.launch(infer=False)
        self.assertEqual(code, 1, out)
        self.assertIn("health reported an error", out)
        self.assertNotIn("is now LIVE", out)

    def test_a_falsey_health_error_object_still_fails(self):
        self.assertHealthErrorRejected({})

    def test_an_empty_health_error_string_still_fails(self):
        self.assertHealthErrorRejected("")

    def test_a_false_health_error_still_fails(self):
        self.assertHealthErrorRejected(False)

    def test_a_budget_past_the_engines_ceiling_is_refused_before_any_container(self):
        for value in ["17179869184", "99999999999999"]:
            with self.subTest(value=value):
                code, out = self.fixture.run_script(
                    ["start.sh", f"--prompt-cache-gib={value}"],
                    {"PREPARE": "0", "HOST": "127.0.0.1", "PORT": str(self.port)},
                    timeout=15,
                )
                self.assertEqual(code, 1, out)
                self.assertIn("past the engine's 2^34 GiB ceiling", out)
                self.assertEqual(self.fixture.launched(), [], out)

    def test_a_leading_zero_budget_is_decimal_not_octal(self):
        """`08` is 8 GiB to the engine; Bash read it as an invalid octal and reported a ceiling that was not there."""
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 8 << 30
        code, out = self.launch(args=["--prompt-cache-gib", "08"])
        self.assertEqual(code, 0, out)
        self.assertIn("snapshots 8192 MiB", out)
        self.assertNotIn("ceiling", out)
        self.assertIn("is now LIVE", out)

    def test_a_leading_zero_fractional_budget_is_decimal_not_octal(self):
        """`09.5` is 9.5 GiB, not Bash's octal error; the whole part is 9, a digit Bash refuses as octal."""
        self.cache_health()
        self.health["live"]["cache"]["prompt_state"]["budget_bytes"] = 10200547328
        code, out = self.launch(args=["--prompt-cache-gib", "09.5"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("ceiling", out)
        self.assertNotIn("MISMATCH", out)
        self.assertIn("is now LIVE", out)

    def test_a_health_redirect_is_not_a_cache_report(self):
        self.cache_health()
        self.health_status = 302
        code, out = self.launch(infer=False)
        self.assertEqual(code, 1, out)
        self.assertIn("health answered HTTP 302", out)
        self.assertNotIn("is now LIVE", out)

    def test_a_health_error_object_with_matching_cache_data_still_fails(self):
        self.cache_health()
        self.health["error"] = "loading"
        code, out = self.launch(infer=False)
        self.assertEqual(code, 1, out)
        self.assertIn("health reported an error", out)
        self.assertNotIn("is now LIVE", out)

    def test_a_genuine_pool_mismatch_is_reported_with_an_override_in_play(self):
        self.cache_health()
        self.health["live"]["cache"]["pool"]["capacity_positions"] = 8388608
        code, out = self.launch(args=["--prompt-cache-gib", "1"])
        self.assertEqual(code, 0, out)
        self.assertIn("WARN:", out)
        self.assertIn("pool 8388608 != 2097152", out)
        self.assertIn("is now LIVE", out)

    def test_short_single_character_reply_remains_accepted(self):
        self.set_content("A" * 7)
        code, out = self.launch()
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)


if __name__ == "__main__":
    unittest.main()
