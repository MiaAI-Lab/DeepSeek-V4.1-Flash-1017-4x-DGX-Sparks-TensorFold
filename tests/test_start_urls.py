"""Check launcher URL authorities without running an engine or contacting a cluster.

The upstream StartRestart fixture supplies fake SSH/Docker and temporary files.
Actual curl requests reach only per-test loopback listeners. SCRIPTS_DIR selects
original or fixed scripts; composing the fixture avoids duplicate test discovery.
"""

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


class _IPv6Server(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


@unittest.skipUnless(_FIXTURE["BASH"] and _CURL, "needs bash 4+ and curl")
class StartURLs(unittest.TestCase):
    def check_host(self, host, hostname="10.0.0.1"):
        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def respond(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                hits.append(("GET", self.path))
                self.respond({"object": "list", "data": []})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                hits.append(("POST", self.path))
                self.respond({
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"completion_tokens": 1},
                })

            def log_message(self, *args):
                pass

        explicit_ipv6 = host == "::1"
        listener_host = "::1" if explicit_ipv6 else "127.0.0.1"
        family = _IPv6Server if explicit_ipv6 else http.server.ThreadingHTTPServer
        try:
            server = family((listener_host, 0), Handler)
        except OSError as exc:
            if explicit_ipv6:
                self.skipTest(f"IPv6 loopback listener unavailable: {exc}")
            raise
        port = server.server_address[1]
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        case = _FIXTURE["StartRestart"](methodName="runTest")
        try:
            case.setUp()
            docker = case.tmp / "bin" / "docker"
            old = '[[ "$prev" != --name ]] || n=$a'
            text = docker.read_text()
            self.assertEqual(text.count(old), 1)
            docker.write_text(text.replace(old, '[[ "$prev" != --name || -n "$n" ]] || n=$a'))

            # Pin localhost to this IPv4 listener rather than trusting DNS or
            # relying on an IPv6/IPv4 resolution preference on the test machine.
            # Explicit ::1 below independently exercises actual IPv6 transport.
            curl = case.tmp / "bin" / "curl"
            curl.write_text(
                "#!/bin/sh\n"
                "for arg in \"$@\"; do\n"
                " case \"$arg\" in\n"
                f"  http://127.0.0.1:{port}/*|http://localhost:{port}/*|"
                f"http://\\[::1\\]:{port}/*|http://::1:{port}/*) ;;\n"
                "  http://*|https://*) echo 'non-test URL denied' >&2; exit 97 ;;\n"
                " esac\n"
                "done\n"
                "printf '%s\\n' \"$*\" >> \"$SIM/curl.log\"\n"
                f"exec {shlex.quote(_CURL)} -q --noproxy '*' "
                f"--resolve 'localhost:{port}:127.0.0.1' \"$@\"\n"
            )
            curl.chmod(0o755)
            fake_hostname = case.tmp / "bin" / "hostname"
            fake_hostname.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(hostname)}\n")
            fake_hostname.chmod(0o755)

            # In particular, prove ::1 is listening through a correct bracketed
            # URL before asking the launcher to construct its own probe URL.
            direct_authority = "[::1]" if explicit_ipv6 else "127.0.0.1"
            direct = subprocess.run(
                [_CURL, "-q", "--noproxy", "*", "-sf", "--max-time", "2",
                 f"http://{direct_authority}:{port}/v1/models"],
                env=case.env, capture_output=True, text=True, timeout=4,
            )
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertEqual(json.loads(direct.stdout)["object"], "list")
            hits.clear()
            code, out = case.run_script(
                ["start.sh"],
                {"PREPARE": "0", "HOST": host, "PORT": str(port), "WAIT_TIMEOUT": "0"},
                timeout=15,
            )
            self.assertEqual(code, 0, out)
            self.assertIn("is now LIVE", out)
            self.assertEqual(hits, [("GET", "/v1/models"), ("POST", "/v1/chat/completions")], out)
            case.assertNoTraceback(out)
            self.assertEqual(sorted(case.launched()), ["head", "w1", "w2", "w3"], out)
            for node in ["head", "w1", "w2", "w3"]:
                self.assertEqual(case.left(node), [_FIXTURE["NAME"]], out)

            probe_host = "127.0.0.1" if host in ["0.0.0.0", "::"] else host
            probe_authority = f"[{probe_host}]" if ":" in probe_host else probe_host
            curl_log = (case.sim / "curl.log").read_text()
            self.assertIn(f"http://{probe_authority}:{port}/v1/models", curl_log)
            self.assertIn(f"http://{probe_authority}:{port}/v1/chat/completions", curl_log)

            display_host = hostname if host in ["0.0.0.0", "::"] else host
            display_authority = f"[{display_host}]" if ":" in display_host else display_host
            self.assertIn(f"API      http://{display_authority}:{port}/v1", out)

            # URL brackets must not leak into the engine's raw bind argument.
            docker_log = (case.sim / "docker.log").read_text().splitlines()
            head_runs = [line for line in docker_log if line.startswith("head docker run ")]
            self.assertEqual(len(head_runs), 1, docker_log)
            args = shlex.split(head_runs[0])
            self.assertEqual(args[args.index("--host") + 1], host, head_runs[0])
        finally:
            case.doCleanups()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_ipv4_literal(self):
        self.check_host("127.0.0.1")

    def test_localhost_pinned_to_loopback(self):
        self.check_host("localhost")

    def test_explicit_ipv6_literal(self):
        self.check_host("::1")

    def test_ipv4_wildcard_probes_loopback(self):
        self.check_host("0.0.0.0")

    def test_ipv6_wildcard_probes_ipv4_loopback(self):
        self.check_host("::")

    def test_wildcard_with_ipv6_hostname_formats_display_url(self):
        for host in ["0.0.0.0", "::"]:
            with self.subTest(host=host):
                self.check_host(host, hostname="::1")


if __name__ == "__main__":
    unittest.main()
