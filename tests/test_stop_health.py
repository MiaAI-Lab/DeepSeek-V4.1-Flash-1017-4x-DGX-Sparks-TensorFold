"""Check stop activity warnings using real loopback HTTP and fake SSH/Docker.

The existing StopSh fixture is composed rather than inherited, so discovery does
not duplicate its tests. SCRIPTS_DIR selects original or fixed stop.sh sources.
All containers and remote hosts are files in the fixture's temporary directory.
"""

import copy
import http.server
import json
from pathlib import Path
import runpy
import shlex
import shutil
import socket
import subprocess
import threading
import unittest


_FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_scripts.py")))
_CURL = shutil.which("curl")
_HOSTS = ["head", "w1", "w2", "w3"]
_HEALTH = {"status": "ok", "live": {"connections": 4, "waiting": 1}}
_UNKNOWN = "could not read the open requests from /health"
_ACTIVITY = "request(s) open (running or waiting) will be cut off"


@unittest.skipUnless(_FIXTURE["BASH"] and _CURL, "needs bash 4+ and curl")
class StopHealth(unittest.TestCase):
    def setUp(self):
        self.fixture = _FIXTURE["StopSh"](methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.all_up()
        self.status = 200
        self.body = json.dumps(_HEALTH).encode()
        self.abort_get = False
        self.incomplete_transfer = False
        self.hits = []
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                test.hits.append(self.path)
                if test.abort_get:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return
                self.send_response(test.status)
                self.send_header("Content-Type", "application/json")
                declared_length = len(test.body) + (10 if test.incomplete_transfer else 0)
                self.send_header("Content-Length", str(declared_length))
                self.end_headers()
                self.wfile.write(test.body)
                if test.incomplete_transfer:
                    self.close_connection = True

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()
        self.addCleanup(self.close_server)

        # Use real HTTP status and transport semantics for this listener only.
        curl = self.fixture.tmp / "bin" / "curl"
        curl.write_text(
            "#!/bin/sh\n"
            "for arg in \"$@\"; do\n"
            " case \"$arg\" in\n"
            f"  http://127.0.0.1:{self.port}/health) ;;\n"
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

    def set_health(self, health):
        self.body = json.dumps(health).encode()

    def stop(self, dry=True, extra_env=None):
        hits_before = len(self.hits)
        code, out = self.fixture.run_script(
            ["stop.sh"],
            {"HOST": "127.0.0.1", "PORT": str(self.port),
             "DRY_RUN": "1" if dry else "0", **(extra_env or {})},
            timeout=15,
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(self.hits[hits_before:], ["/health"], out)
        self.fixture.assertNoTraceback(out)
        self.assertNotIn("Traceback (most recent call last)", out)
        for host in _HOSTS:
            expected = [_FIXTURE["NAME"]] if dry else []
            self.assertEqual(self.fixture.left(host), expected, out)
        if dry:
            self.assertIn("nothing is stopped", out)
        else:
            self.assertIn("its GPU memory is free again", out)
        return out

    def assert_unknown(self, dry=True, extra_env=None):
        out = self.stop(dry, extra_env)
        self.assertIn(_UNKNOWN, out)
        self.assertNotIn(_ACTIVITY, out)
        return out

    def test_busy_count_includes_waiting_once_and_dry_run_preserves_all(self):
        out = self.stop()
        self.assertIn("4 " + _ACTIVITY, out)
        self.assertNotIn("5 " + _ACTIVITY, out)
        self.assertNotIn(_UNKNOWN, out)

    def test_idle_has_no_activity_warning(self):
        self.set_health({"status": "ok", "live": {"connections": 0, "waiting": 0}})
        out = self.stop()
        self.assertNotIn(_ACTIVITY, out)
        self.assertNotIn(_UNKNOWN, out)

    def test_only_waiting_request_is_counted(self):
        self.set_health({"status": "ok", "live": {"connections": 1, "waiting": 1}})
        out = self.stop()
        self.assertIn("1 " + _ACTIVITY, out)
        self.assertNotIn("2 " + _ACTIVITY, out)
        self.assertNotIn(_UNKNOWN, out)

    def test_connections_without_waiting_are_counted(self):
        self.set_health({"status": "ok", "live": {"connections": 3}})
        out = self.stop()
        self.assertIn("3 " + _ACTIVITY, out)
        self.assertNotIn(_UNKNOWN, out)

    def test_status_only_and_legacy_schema_are_unknown(self):
        for health in [{"status": "ok"}, {"requests_running": 4}]:
            with self.subTest(health=health):
                self.set_health(health)
                self.assert_unknown()

    def test_invalid_root_and_live_shapes_are_unknown(self):
        cases = [
            [], None, True,
            {"live": None}, {"live": []}, {"live": "bad"}, {"live": True},
        ]
        for health in cases:
            with self.subTest(health=health):
                self.set_health(health)
                self.assert_unknown()

    def test_missing_or_invalid_connections_are_unknown(self):
        missing = {"status": "ok", "live": {"waiting": 1}}
        cases = [missing]
        for count in [None, -1, True, False, "4", 4.0]:
            cases.append({"status": "ok", "live": {"connections": count}})
        for health in cases:
            with self.subTest(health=health):
                self.set_health(health)
                self.assert_unknown()

    def test_error_envelope_is_unknown_even_with_valid_live(self):
        health = copy.deepcopy(_HEALTH)
        health["error"] = {"message": "controlled error"}
        self.set_health(health)
        self.assert_unknown()

    def test_http_500_with_valid_health_is_unknown(self):
        self.status = 500
        self.assert_unknown()

    def test_http_302_with_valid_health_is_unknown(self):
        self.status = 302
        self.assert_unknown()

    def test_malformed_json_is_unknown(self):
        self.body = b'{"live":'
        self.assert_unknown()

    def test_transport_failure_is_unknown(self):
        self.abort_get = True
        self.assert_unknown()

    def test_incomplete_transfer_with_complete_json_is_unknown(self):
        self.incomplete_transfer = True
        # Prove this response still has parseable JSON and HTTP 200, while curl
        # rejects its prematurely closed body transfer. This isolates pipefail
        # handling from JSON/schema rejection in the script's activity parser.
        probe = subprocess.run(
            [_CURL, "-q", "--noproxy", "*", "-s", "--max-time", "3",
             "-w", "\n%{http_code}", f"http://127.0.0.1:{self.port}/health"],
            env=self.fixture.env, capture_output=True, text=True, timeout=5,
        )
        body, separator, status = probe.stdout.rpartition("\n")
        self.assertEqual(probe.returncode, 18, probe.stderr)
        self.assertEqual((separator, status), ("\n", "200"))
        self.assertEqual(json.loads(body), _HEALTH)
        self.assert_unknown()

    def test_busy_normal_stop_removes_all_fake_containers(self):
        out = self.stop(dry=False)
        self.assertIn("4 " + _ACTIVITY, out)
        self.assertNotIn(_UNKNOWN, out)

    def test_unknown_activity_does_not_prevent_normal_stop(self):
        self.set_health({"status": "ok"})
        self.assert_unknown(dry=False)

    def test_invalid_connections_remain_unknown_with_python_optimize_2(self):
        self.set_health({"status": "ok", "live": {"connections": True}})
        self.assert_unknown(extra_env={"PYTHONOPTIMIZE": "2"})


if __name__ == "__main__":
    unittest.main()
