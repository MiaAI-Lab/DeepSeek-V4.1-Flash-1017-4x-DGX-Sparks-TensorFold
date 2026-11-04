"""Regression tests for stop.sh (and its /health warning), ./start.sh restart and its smoke test, the ring's NCCL graph
files under DRY_RUN=1, the prepared marker and prepare.sh's weight-file loop.

Offline: no Spark, GPU, Docker or network. Each test copies the scripts into a temporary folder and puts fake `ssh`,
`docker`, `curl` and `getent` first on PATH. The fake ssh runs a worker's command locally with that worker's own HOME
and container list (under the test's folder); the fake docker keeps each Spark's containers as lines in a file.
Run: python3 -m unittest discover -s tests       SCRIPTS_DIR=<a repository copy> tests the scripts there instead.
The scripts need bash 4 or later: BASH4=<path> picks one, else the first bash on PATH (or Homebrew's) that is.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)
COPY = ["stop.sh", "start.sh", "start-switchless.sh", "scripts", "model", "zig"]


def find_bash():
    for cand in [os.environ.get("BASH4"), shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"]:
        if not cand or not os.path.exists(cand):
            continue
        out = subprocess.run([cand, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
        if out.isdigit() and int(out) >= 4:
            return cand
    return None


BASH = find_bash()

FAKE_SSH = r"""#!BASH
# fake ssh: ssh [-o opt]... host command...; a host listed in $SIM/unreachable cannot be reached, one in $SIM/flaky
# answers `true` (the reachability probe) and drops every other command
while [[ "$1" == -o ]]; do shift 2; done
host=$1; shift
echo "$host $*" >> "$SIM/ssh.log"
grep -qxF "$host" "$SIM/unreachable" 2>/dev/null && { echo "ssh: connect to host $host: No route to host" >&2; exit 255; }
grep -qxF "$host" "$SIM/flaky" 2>/dev/null && [[ "$*" != true ]] && { echo "Connection to $host closed by remote host." >&2; exit 255; }
mkdir -p "$SIM/home/$host"
SIM_HOST=$host HOME="$SIM/home/$host" exec BASH -c "$*"
"""

FAKE_DOCKER = r"""#!BASH
# fake docker: each Spark's containers are lines in $SIM/hosts/<host>/containers; a container named in
# $SIM/hosts/<host>/stuck cannot be stopped or removed; with $SIM/hosts/<host>/ps_fails `docker ps` fails; with
# $SIM/image the image exists (its identity: "image-ident"); `run -d --name N` adds N
h=${SIM_HOST:-head}; d="$SIM/hosts/$h"; mkdir -p "$d"; touch "$d/containers"
echo "$h docker $*" >> "$SIM/docker.log"
has() { grep -qxF "$1" "$d/containers"; }
stuck() { grep -qxF "$1" "$d/stuck" 2>/dev/null; }
last=${!#}
case "$1" in
  ps) [[ ! -e "$d/ps_fails" ]] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }; cat "$d/containers" ;;
  inspect) has "$last" || { echo "Error: No such object: $last" >&2; exit 1; }
           [[ "$*" != *State.Running* ]] || echo true ;;
  stop) stuck "$last" && { echo "Error response from daemon: cannot stop container: $last" >&2; exit 1; }; true ;;
  rm) stuck "$last" && { echo "Error response from daemon: cannot remove container: $last" >&2; exit 1; }
      grep -vxF "$last" "$d/containers" > "$d/c.new" || true; mv "$d/c.new" "$d/containers" ;;
  logs) echo "a log line" ;;
  image) [[ -e "$SIM/image" ]] || exit 1; echo image-ident ;;
  run) n=""; prev=""; for a in "$@"; do [[ "$prev" != --name ]] || n=$a; prev=$a; done; echo "$n" >> "$d/containers"; echo container-id ;;
  *) exit 0 ;;
esac
"""

FAKE_CURL = """#!/bin/sh
# fake curl: /health answers $SIM/health if it exists, else the native server's schema with nothing open; with -w
# its HTTP status ($SIM/health_status or 200) on the last line, as curl prints it
if [ -e "$SIM/health" ]; then cat "$SIM/health"; else printf '%s' '{"status": "ok", "live": {"connections": 0, "waiting": 0}}'; fi
case "$*" in *http_code*) printf '\n%s' "$(cat "$SIM/health_status" 2>/dev/null || echo 200)" ;; esac
"""

FAKE_GETENT = """#!/bin/sh
exit 2
"""

# macOS's paste reads stdin only when told "-" (GNU paste, on a Spark, by itself): stop.sh's `paste -sd, <<<...`
FAKE_PASTE = """#!/bin/sh
for a in "$@"; do case "$a" in -*) ;; *) exec /usr/bin/paste "$@" ;; esac; done
exec /usr/bin/paste "$@" -
"""

NAME = "dsv41-1017-tf"
WORKERS = {"WORKER": "w1", "WORKER2": "w2", "WORKER3": "w3"}


@unittest.skipIf(BASH is None, "needs bash 4 or later")
class ScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-scripts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for name in COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, self.repo / name, ignore=shutil.ignore_patterns("local.sh", "__pycache__"))
            else:
                shutil.copy2(src, self.repo / name)
        self.sim = self.tmp / "sim"
        (self.sim / "hosts").mkdir(parents=True)
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        for name, body in [("ssh", FAKE_SSH), ("docker", FAKE_DOCKER), ("curl", FAKE_CURL), ("getent", FAKE_GETENT)]:
            p = bin_ / name
            p.write_text(body.replace("BASH", BASH))
            p.chmod(0o755)
        if sys.platform == "darwin":
            (bin_ / "paste").write_text(FAKE_PASTE)
            (bin_ / "paste").chmod(0o755)
        # the scripts' #!/usr/bin/env bash: a bash 4 first on PATH
        (bin_ / "bash").symlink_to(BASH)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.env = {
            "PATH": f"{bin_}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(self.home),
            "SIM": str(self.sim),
            "LANG": "C",
            **WORKERS,
        }

    def containers(self, host, *names):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "containers").write_text("".join(n + "\n" for n in names))

    def left(self, host):
        f = self.sim / "hosts" / host / "containers"
        return f.read_text().split() if f.exists() else []

    def stuck(self, host):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "stuck").write_text(NAME + "\n")

    def unreachable(self, *hosts):
        (self.sim / "unreachable").write_text("".join(h + "\n" for h in hosts))

    def flaky(self, *hosts):
        (self.sim / "flaky").write_text("".join(h + "\n" for h in hosts))

    def ps_fails(self, host):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "ps_fails").write_text("")

    def run_script(self, args, extra_env=None, timeout=60):
        env = {**self.env, **(extra_env or {})}
        r = subprocess.run([BASH, *args], cwd=self.repo, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr

    def bash(self, body, extra_env=None):
        """Source config.sh and nodes.sh in the copy, then run body; returns (exit code, output)."""
        script = f"set -euo pipefail\ncd {self.repo}\nsource ./scripts/config.sh\nsource ./scripts/nodes.sh\n{body}\n"
        return self.run_script(["-c", script], extra_env)

    def assertNoTraceback(self, out):
        self.assertNotIn("unbound variable", out)
        self.assertNotIn("syntax error", out)
        self.assertNotIn("command not found", out)


class StopSh(ScriptTest):
    def all_up(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)

    def test_every_rank_stopped_exits_zero(self):
        self.all_up()
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("its GPU memory is free again", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], h)

    def test_nothing_to_stop_exits_zero(self):
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("nothing to stop", out)

    def test_worker_that_cannot_be_stopped_fails(self):
        self.all_up()
        self.stuck("w2")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("could not stop rank 2", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w2"), [NAME])
        for h in ["head", "w1", "w3"]:          # the others are still stopped
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_unreachable_worker_fails(self):
        self.all_up()
        self.unreachable("w3")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 3 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w3"), [NAME])
        for h in ["head", "w1", "w2"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_unreachable_worker_is_not_nothing_to_stop(self):
        self.unreachable("w2")
        self.containers("w2", NAME)
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertNotIn("nothing to stop", out)
        self.assertIn("rank 2 was left alone", out)
        self.assertNoTraceback(out)

    def test_container_list_failing_on_a_worker_fails(self):
        self.all_up()
        self.ps_fails("w2")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 2 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w2"), [NAME])
        for h in ["head", "w1", "w3"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_ssh_dropping_after_the_probe_fails(self):
        self.all_up()
        self.flaky("w1")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 1 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w1"), [NAME])
        self.assertNoTraceback(out)

    def test_container_list_failing_here_fails(self):
        self.all_up()
        self.ps_fails("head")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 0 was left alone", out)
        self.assertNotIn("nothing to stop", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("head"), [NAME])
        for h in ["w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_container_list_failing_here_alone_is_not_nothing_to_stop(self):
        self.containers("head", NAME)
        self.ps_fails("head")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertNotIn("nothing to stop", out)
        self.assertEqual(self.left("head"), [NAME])

    def test_dry_run_stops_nothing(self):
        self.all_up()
        code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("nothing is stopped", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [NAME], h)

    def test_dry_run_with_a_spark_it_cannot_list_fails(self):
        for case, setup in [("docker ps fails", lambda: self.ps_fails("w2")), ("unreachable", lambda: self.unreachable("w2"))]:
            with self.subTest(case=case):
                self.setUp()
                self.all_up()
                setup()
                code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1"})
                self.assertEqual(code, 1, out)
                self.assertIn("nothing is stopped", out)
                for h in ["head", "w1", "w2", "w3"]:
                    self.assertEqual(self.left(h), [NAME], h)


class StopHealth(ScriptTest):
    """stop.sh's warning about requests it cuts off, read from the native server's /health (live.connections counts
    the running and the waiting ones)."""

    def stop_with(self, health):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        if health is not None:
            (self.sim / "health").write_text(health)
        return self.run_script(["stop.sh"])

    def test_open_requests_warn(self):
        code, out = self.stop_with('{"status": "ok", "live": {"connections": 4, "waiting": 1}}')
        self.assertEqual(code, 0, out)
        self.assertIn("4 request(s) open (running or waiting) will be cut off", out)

    def test_nothing_open_no_warning(self):
        code, out = self.stop_with(None)
        self.assertEqual(code, 0, out)
        self.assertNotIn("cut off", out)

    def test_unreadable_health_is_unknown_not_none(self):
        cases = {"no answer": "", "an error page": "<html>401</html>", "another schema": '{"requests_running": 3}',
                 "not a count": '{"live": {"connections": "4"}}', "a negative count": '{"live": {"connections": -1}}',
                 "an error reply": '{"error": {"message": "unauthorized"}, "live": {"connections": 0}}'}
        for name, body in cases.items():
            with self.subTest(name):
                code, out = self.stop_with(body)
                self.assertEqual(code, 0, out)
                self.assertIn("could not read the open requests from /health", out)
                self.assertNoTraceback(out)

    def test_http_error_is_unknown_not_none(self):
        for status in ["500", "302", "401"]:
            with self.subTest(status):
                (self.sim / "health_status").write_text(status)
                code, out = self.stop_with('{"status": "ok", "live": {"connections": 0, "waiting": 0}}')
                self.assertEqual(code, 0, out)
                self.assertIn("could not read the open requests from /health", out)

    def test_error_null_is_no_error(self):
        # JSON null is no error: the count is read as usual
        code, out = self.stop_with('{"error": null, "live": {"connections": 2, "waiting": 0}}')
        self.assertEqual(code, 0, out)
        self.assertIn("2 request(s) open (running or waiting) will be cut off", out)

    def test_other_success_status_is_unknown(self):
        # the native server answers /health with 200; any other status, 2xx too, is not its answer
        (self.sim / "health_status").write_text("204")
        code, out = self.stop_with('{"live": {"connections": 0, "waiting": 0}}')
        self.assertEqual(code, 0, out)
        self.assertIn("could not read the open requests from /health", out)

    def test_stop_health_check_holds_with_python_optimizing(self):
        # the checks are no asserts: PYTHONOPTIMIZE would drop those
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        (self.sim / "health").write_text('{"live": {"connections": -1}}')
        code, out = self.run_script(["stop.sh"], {"PYTHONOPTIMIZE": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("could not read the open requests from /health", out)


class StartRestart(ScriptTest):
    """./start.sh with only what needs a Spark replaced: the links (detect_links), flock, ss, free, hostname, curl's
    answers (the server ready, the smoke test's reply) and scripts/prepare.sh (it records that it ran)."""
    FAKES = {
        "flock": "#!/bin/sh\nexit 0\n",
        "ss": "#!/bin/sh\nexit 0\n",
        "free": "#!/bin/sh\necho 'Mem: 128 1 1 0 0 120'\n",
        "hostname": "#!/bin/sh\necho 10.0.0.1\n",
        # the smoke test's reply: $SIM/smoke if it exists, else a completion; with -w its HTTP status ($SIM/smoke_status
        # or 200) on the last line, as curl prints it
        "curl": "#!/bin/sh\ncase \"$*\" in *completions*)\n"
                "  if [ -e \"$SIM/smoke\" ]; then cat \"$SIM/smoke\"; else printf '%s' '{\"choices\": [{\"message\": {\"role\": \"assistant\", \"content\": \"OK\"}}], \"usage\": {\"completion_tokens\": 1}}'; fi\n"
                "  case \"$*\" in *http_code*) printf '\\n%s' \"$(cat \"$SIM/smoke_status\" 2>/dev/null || echo 200)\" ;; esac\n"
                "  if [ -e \"$SIM/smoke_exit\" ]; then exit \"$(cat \"$SIM/smoke_exit\")\"; fi ;;\n"
                "*) printf '%s' '{\"status\": \"ok\", \"live\": {\"connections\": 0, \"waiting\": 0}}'\n"
                "   case \"$*\" in *http_code*) printf '\\n200' ;; esac ;; esac\n",
    }

    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        for name, body in self.FAKES.items():
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        nodes = self.repo / "scripts" / "nodes.sh"
        nodes.write_text(nodes.read_text() + """
detect_links() {
  FABRIC=mesh; MASTER_ADDR=127.0.0.1
  for r in 0 1 2 3; do NODE_DEV[r]=fake; NODE_HCAS[r]=fake; NODE_GID[r]=0; LINK_HEAD_ADDR[r]=127.0.0.1; done
}
node_order() { echo 0,1,2,3; }
""")
        (self.repo / "scripts" / "prepare.sh").write_text("#!/usr/bin/env bash\necho prepared >> \"$SIM/events\"\n")
        (self.sim / "image").write_text("")

    def launched(self):
        log = (self.sim / "docker.log").read_text() if (self.sim / "docker.log").exists() else ""
        return [l.split()[0] for l in log.splitlines() if " docker run -d " in l]

    def prepared(self):
        return (self.sim / "events").exists()

    def test_restart_starts_every_rank(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)
        self.assertIn("is now LIVE", out)

    def test_a_worker_reads_its_weight_file_from_its_disk_when_its_copy_matches(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        code, out = self.bash('echo "@$DATA_DIR|$WORKER_RANK_DIR"')
        data_dir, rank_dir = out.split("@")[-1].strip().split("|")
        cache = Path(data_dir) / "rank-cache"
        cache.mkdir(parents=True)
        for n in range(4):
            (cache / f"rank{n}of4-abc.bin").write_bytes(b"x" * (10 + n))
        w2 = Path(rank_dir.replace("$HOME", str(self.sim / "home" / "w2")))
        w2.mkdir(parents=True)
        (w2 / "rank2of4-abc.bin").write_bytes(b"x" * 12)  # the size of this Spark's node-2 file
        os.utime(w2 / "rank2of4-abc.bin", (1000, cache.joinpath("rank2of4-abc.bin").stat().st_mtime))
        w3 = Path(rank_dir.replace("$HOME", str(self.sim / "home" / "w3")))
        w3.mkdir(parents=True)
        (w3 / "rank3of4-abc.bin").write_bytes(b"x" * 5)  # a partial copy: not used

        def runs():
            log = (self.sim / "docker.log").read_text()
            (self.sim / "docker.log").unlink()
            return {l.split()[0]: l for l in log.splitlines() if " docker run -d " in l}

        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "RANK_LOCAL": "1"})
        self.assertEqual(code, 0, out)
        r = runs()
        self.assertIn(":/rank-local:ro", r["w2"], out)
        self.assertIn(" /rank-local 2 4 ", r["w2"], out)
        for h in ("w1", "w3"):
            self.assertNotIn("/rank-local", r[h], out)
        self.assertIn("rank 1 reads its weight file over NFS", out)
        self.assertIn("rank 3 reads its weight file over NFS", out)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "RANK_LOCAL": "0"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("/rank-local", "".join(runs().values()), out)
        self.assertNotIn("reads its weight file over NFS", out)
        # a copy of the same name and size baked earlier than this Spark's file holds other bytes: NFS, with a warning
        os.utime(w2 / "rank2of4-abc.bin", (1000, 1000))
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "RANK_LOCAL": "1"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("/rank-local", "".join(runs().values()), out)
        self.assertIn("rank 2 reads its weight file over NFS", out)
        # off by default: no copies, no warnings
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("/rank-local", "".join(runs().values()), out)
        self.assertNotIn("reads its weight file over NFS", out)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "RANK_LOCAL": "yes"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("RANK_LOCAL is 0 or 1", out)

    def copies(self, *workers):
        """This Spark's four weight files, and on each worker named a copy of the one it reads (name, size and time)."""
        code, out = self.bash('echo "@$DATA_DIR|$WORKER_RANK_DIR"')
        data_dir, rank_dir = out.split("@")[-1].strip().split("|")
        cache = Path(data_dir) / "rank-cache"
        cache.mkdir(parents=True, exist_ok=True)
        for n in range(4):
            (cache / f"rank{n}of4-abc.bin").write_bytes(b"x" * (10 + n))
        for h in workers:   # node_order 0,1,2,3: worker wN reads node N's file
            name = f"rank{h[1]}of4-abc.bin"
            d = Path(rank_dir.replace("$HOME", str(self.sim / "home" / h)))
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes((cache / name).read_bytes())
            os.utime(d / name, (1000, (cache / name).stat().st_mtime))
        return Path(rank_dir)

    def test_step_4_says_where_the_weights_load_from(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        rank_dir = self.copies("w1", "w2", "w3")
        cases = [({}, "Loading: ~75 GiB of weights on each Spark from its own disk, then the caches (a minute or two)"),
                 ({"gone": "w2"}, "Loading: ~75 GiB of weights on each Spark over NFS (ranks 1,3 from their own disks)"),
                 ({"gone": "w1"}, "Loading: ~75 GiB of weights on each Spark over NFS (rank 3 from its own disk)"),
                 ({"off": True}, "Loading: ~75 GiB of weights on each Spark over NFS, then the caches (a few minutes)")]
        for change, want in cases:
            if "gone" in change:
                for f in Path(str(rank_dir).replace("$HOME", str(self.sim / "home" / change["gone"]))).glob("*.bin"):
                    f.unlink()
            env = {"PREPARE": "0", "RANK_LOCAL": "0" if change.get("off") else "1"}
            code, out = self.run_script(["start.sh", "restart"], env)
            self.assertEqual(code, 0, out)
            self.assertIn(want, out)
            self.assertEqual(out.count("Loading: "), 1, out)

    def test_the_hints_name_the_launcher_this_run_was_started_as(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        self.copies()   # no copies on the workers: each rank warns, naming the script to re-prepare with
        for script, want in (("start.sh", "./start.sh"), ("start-switchless.sh", "./start-switchless.sh")):
            code, out = self.run_script([script, "restart"], {"PREPARE": "0", "RANK_LOCAL": "1"})
            self.assertEqual(code, 0, out)
            self.assertEqual(out.count(f"(PREPARE=1 {want} restart makes one)"), 3, out)
            self.assertIn(f"Restart  {want} restart", out)
            for h in ["head", "w1", "w2", "w3"]:   # already running everywhere: nothing to do
                self.containers(h, NAME)
            code, out = self.run_script([script], {"PREPARE": "0"})
            self.assertEqual(code, 0, out)
            self.assertIn(f"nothing to do. {want} restart restarts it", out)

    def test_smoke_test_needs_a_real_completion(self):
        ok_body = '{"choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"completion_tokens": 1}}'
        cases = {"HTTP 500 with a completion's body": (ok_body, "500"),
                 "HTTP 200 with an error object": ('{"error": {"message": "boom"}, "choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"completion_tokens": 1}}', "200"),
                 "HTTP 200 without reply text": ('{"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": {"completion_tokens": 0}}', "200"),
                 "HTTP 200 with an empty error object": ('{"error": {}, "choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"completion_tokens": 1}}', "200"),
                 "HTTP 200, not the assistant's": ('{"choices": [{"message": {"role": "user", "content": "OK"}}], "usage": {"completion_tokens": 1}}', "200"),
                 "HTTP 200, then curl fails (a timeout)": (ok_body, "200", "28")}
        for name, (body, status, *exit_) in cases.items():
            with self.subTest(name):
                (self.sim / "smoke").write_text(body)
                (self.sim / "smoke_status").write_text(status)
                if exit_:
                    (self.sim / "smoke_exit").write_text(exit_[0])
                else:
                    (self.sim / "smoke_exit").unlink(missing_ok=True)
                code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
                self.assertNotEqual(code, 0, out)
                self.assertIn("the smoke test request failed", out)
                self.assertNotIn("is now LIVE", out)
                for h in ["head", "w1", "w2", "w3"]:
                    self.containers(h)

    def test_smoke_test_error_null_passes_other_2xx_fails(self):
        ok = '{"error": null, "choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"completion_tokens": 1}}'
        (self.sim / "smoke").write_text(ok)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h)
        (self.sim / "smoke_status").write_text("201")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("the smoke test request failed", out)
        self.assertNotIn("is now LIVE", out)

    def test_smoke_test_passes_without_a_role_and_with_python_optimizing(self):
        # a reply whose message has no "role" is the assistant's; the checks are no asserts (PYTHONOPTIMIZE drops those)
        (self.sim / "smoke").write_text('{"choices": [{"message": {"content": "OK"}}], "usage": {"completion_tokens": 1}}')
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h)
        (self.sim / "smoke").write_text('{"choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"completion_tokens": 1}}')
        (self.sim / "smoke_status").write_text("500")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "PYTHONOPTIMIZE": "1"})
        self.assertNotEqual(code, 0, out)
        self.assertNotIn("is now LIVE", out)

    def test_restart_stops_when_a_rank_cannot_be_stopped(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        self.stuck("w2")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("could not stop rank 2", out)
        self.assertEqual(self.launched(), [], out)
        self.assertNotIn("is now LIVE", out)

    def test_restart_stops_when_a_worker_cannot_be_checked(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        self.ps_fails("w3")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertEqual(self.launched(), [], out)

    def test_template_that_cannot_be_hashed_starts_nothing(self):
        # no marker yet, or an empty one: a failed hash must not read as "Ready" and start the ranks
        (self.repo / "model" / "chat_template.jinja").unlink()
        state = self.home / ".local/state/dsv41-tensorfold"
        for marker in [None, ""]:
            with self.subTest(marker=marker):
                if marker is not None:
                    state.mkdir(parents=True, exist_ok=True)
                    (state / "prepared").write_text(marker)
                code, out = self.run_script(["start.sh"])
                self.assertNotEqual(code, 0, out)
                self.assertNotIn("Ready:", out)
                self.assertIn("chat_template", out)
                self.assertEqual(self.launched(), [], out)
                self.assertFalse(self.prepared(), out)

    def test_old_marker_runs_prepare_once(self):
        state = self.home / ".local/state/dsv41-tensorfold"
        state.mkdir(parents=True)
        (state / "prepared").write_text("model=old image=old kit=old\n")
        code, out = self.run_script(["start.sh"])
        self.assertEqual(code, 0, out)
        self.assertTrue(self.prepared(), out)
        self.assertIn("Not ready yet", out)


class MasterAddr(ScriptTest):
    """The rendezvous address config.sh derives: a Tailscale address only when the node has no other (issue #21)."""

    def derived(self, getent_addrs, hostname_i):
        bin_ = self.tmp / "bin"
        (bin_ / "getent").write_text("#!/bin/sh\n" + "".join(f"echo '{a} STREAM host'\n" for a in getent_addrs) + ("exit 0\n" if getent_addrs else "exit 2\n"))
        (bin_ / "getent").chmod(0o755)
        (bin_ / "hostname").write_text(f"#!/bin/sh\n[ \"$1\" = -I ] && echo '{hostname_i}' || echo spark1\n")
        (bin_ / "hostname").chmod(0o755)
        code, out = self.bash('echo "@$MASTER_ADDR"')
        self.assertEqual(code, 0, out)
        return out.split("@")[-1].strip()

    def test_the_hostname_address_wins_as_before(self):
        self.assertEqual(self.derived(["192.0.2.5"], "192.0.2.5 100.101.1.2"), "192.0.2.5")

    def test_a_tailscale_hostname_address_gives_way_to_the_lan(self):
        self.assertEqual(self.derived(["100.101.1.2"], "100.101.1.2 192.0.2.5"), "192.0.2.5")
        self.assertEqual(self.derived(["100.64.0.1", "198.51.100.7"], "100.64.0.1 198.51.100.7"), "198.51.100.7")

    def test_tailscale_first_in_hostname_i_without_getent(self):
        self.assertEqual(self.derived([], "100.127.255.1 fd7a::1 192.0.2.5"), "192.0.2.5")

    def test_an_address_just_outside_the_tailscale_block_is_kept(self):
        self.assertEqual(self.derived(["100.63.0.1"], "100.63.0.1 192.0.2.5"), "100.63.0.1")
        self.assertEqual(self.derived(["100.128.0.1"], "100.128.0.1 192.0.2.5"), "100.128.0.1")

    def test_only_tailscale_is_still_used(self):
        self.assertEqual(self.derived(["100.101.1.2"], "100.101.1.2"), "100.101.1.2")
        self.assertEqual(self.derived([], "100.101.1.2"), "100.101.1.2")

    def test_an_explicit_master_addr_wins(self):
        self.env["MASTER_ADDR"] = "100.101.1.2"
        self.assertEqual(self.derived(["192.0.2.5"], "192.0.2.5"), "100.101.1.2")


class LauncherHint(ScriptTest):
    """The start script a message names fits the fabric: ./start.sh stops on a ring, ./start-switchless.sh on a switch."""

    def test_launcher_fits_the_fabric(self):
        code, out = self.bash("launcher ring; launcher mesh")
        self.assertEqual(code, 0, out)
        self.assertEqual(out.split(), ["./start-switchless.sh", "./start.sh"])

    def test_prepare_names_the_launcher_for_the_fabric_it_found(self):
        source = (self.repo / "scripts" / "prepare.sh").read_text()
        done = [l for l in source.splitlines() if l.startswith('log "Done:')]
        self.assertEqual(len(done), 1, done)
        for fabric, want in (("ring", "./start-switchless.sh"), ("mesh", "./start.sh")):
            code, out = self.bash(f'log() {{ echo "$*"; }}\nFABRIC={fabric}\n{done[0]}')
            self.assertEqual(code, 0, out)
            self.assertIn(f"Start the server with {want} (port 8888).", out)


class RingGraphs(ScriptTest):
    BODY = """graph_xml() { echo "<graphs rank=\\"$1\\" cr=\\"$2\\" peer=\\"$3\\"/>"; }
FABRIC=ring
ring_graphs
for r in 0 1 2 3; do echo "G$r=${NODE_GRAPH[r]:-}"; done"""

    def graph_files(self):
        found = sorted(str(p.relative_to(self.tmp)) for p in self.tmp.rglob("nccl-graph-*"))
        return found

    def test_graph_files_written_on_every_rank(self):
        code, out = self.bash(self.BODY)
        self.assertEqual(code, 0, out)
        files = self.graph_files()
        self.assertEqual(len(files), 12, files)   # 3 a rank: the ring and the two links
        for r in range(4):
            self.assertRegex(out, rf"G{r}=\S*nccl-graph-rank{r}\b")

    def test_dry_run_writes_no_graph_files(self):
        code, out = self.bash(self.BODY, {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.graph_files(), [])
        ssh = (self.sim / "ssh.log").read_text() if (self.sim / "ssh.log").exists() else ""
        self.assertNotIn("cat >", ssh)
        self.assertNotIn("mkdir", ssh)
        self.assertFalse((self.home / ".local/state/dsv41-tensorfold").exists())
        # the printed docker commands still name each rank's graph files
        for r in range(4):
            self.assertRegex(out, rf"G{r}=\S*nccl-graph-rank{r}\b")
        self.assertIn("DRY_RUN", out)

    def test_dry_run_skips_a_worker_it_cannot_reach(self):
        body = "WORKER_DOWN[2]=1\n" + self.BODY
        code, out = self.bash(body, {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("G2=\n", out)
        self.assertEqual(self.graph_files(), [])


class PreparedState(ScriptTest):
    BODY = 'image_ident() { echo image-ident; }\nprepared_state'

    def state(self, extra=None):
        code, out = self.bash(self.BODY, extra)
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        return out.strip()

    def assertChanges(self, extra=None, edit=None):
        before = self.state()
        if edit:
            edit()
        after = self.state(extra)
        self.assertNotEqual(before, after, f"the marker did not change: {after}")

    def test_same_settings_same_marker(self):
        self.assertEqual(self.state(), self.state())

    def test_hf_cache_changes_marker(self):
        self.assertChanges({"HF_CACHE": str(self.tmp / "other-hf")})

    def test_nfs_path_changes_marker(self):
        self.assertChanges({"NFS_PATH": "/other"})

    def test_rank_local_changes_the_marker_only_when_on(self):
        self.assertEqual(self.state(), self.state({"RANK_LOCAL": "0"}))
        self.assertChanges({"RANK_LOCAL": "1"})
        self.assertNotEqual(self.state({"RANK_LOCAL": "1"}), self.state({"RANK_LOCAL": "1", "WORKER_RANK_DIR": "/other"}))

    def test_nfs_volume_changes_marker(self):
        self.assertChanges({"NFS_VOLUME": "other-volume"})

    def test_each_nfs_server_changes_marker(self):
        for var in ["NFS_SERVER", "NFS_SERVER2", "NFS_SERVER3"]:
            with self.subTest(var=var):
                self.assertChanges({var: "10.9.9.9"})

    def test_chat_template_changes_marker(self):
        tpl = self.repo / "model" / "chat_template.jinja"
        self.assertChanges(edit=lambda: tpl.write_text(tpl.read_text() + "\n{# changed #}\n"))

    def test_same_length_template_change_changes_marker(self):
        tpl = self.repo / "model" / "chat_template.jinja"
        def edit():
            t = tpl.read_text()
            i = next(k for k, ch in enumerate(t) if ch.isalpha())
            tpl.write_text(t[:i] + ("X" if t[i] != "X" else "Y") + t[i + 1:])
        self.assertChanges(edit=edit)

    def test_spaces_in_paths_count(self):
        a = self.state({"HF_CACHE": "/data/hf cache"})
        b = self.state({"HF_CACHE": "/data/hfcache"})
        self.assertNotEqual(a, b)

    def test_no_two_settings_give_the_same_marker(self):
        # a field's value holding the next field's text must not shift it into that field
        pairs = [
            ({"HF_CACHE": "/a nfs=vol:/b", "NFS_PATH": "/c", "NFS_VOLUME": "vol"},
             {"HF_CACHE": "/a", "NFS_PATH": "/b nfs=vol:/c", "NFS_VOLUME": "vol"}),
            ({"HF_CACHE": "/a nfs_path=/b", "NFS_PATH": "/c"}, {"HF_CACHE": "/a", "NFS_PATH": "/b nfs_path=/c"}),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(self.state(a), self.state(b))

    def test_template_that_cannot_be_hashed_fails(self):
        (self.repo / "model" / "chat_template.jinja").unlink()
        code, out = self.bash(self.BODY)
        self.assertNotEqual(code, 0, out)
        self.assertIn("chat_template", out)

    def test_marker_does_not_depend_on_the_locale(self):
        a = self.state({"HF_CACHE": "/data/\u00e5", "LC_ALL": "C"})
        b = self.state({"HF_CACHE": "/data/\u00e5", "LC_ALL": "en_US.UTF-8"})
        self.assertEqual(a, b)

    def test_writing_the_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        body = 'image_ident() { echo image-ident; }\nmkdir -p "$STATE_DIR"\nwrite_prepared_marker'
        code, out = self.bash(body)
        self.assertEqual(code, 0, out)
        self.assertEqual(marker.read_text().strip(), self.state())

    def test_failed_hash_keeps_the_previous_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        marker.parent.mkdir(parents=True)
        marker.write_text("model=valid-before\n")
        (self.repo / "model" / "chat_template.jinja").unlink()
        code, out = self.bash('image_ident() { echo image-ident; }\nwrite_prepared_marker')
        self.assertNotEqual(code, 0, out)
        self.assertEqual(marker.read_text(), "model=valid-before\n")
        self.assertFalse(marker.with_name("prepared.new").exists(), out)

    def test_failed_rename_keeps_the_previous_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        marker.parent.mkdir(parents=True)
        marker.write_text("model=valid-before\n")
        fake = self.tmp / "bin" / "mv"
        fake.write_text("#!/bin/sh\necho 'mv: cannot move' >&2\nexit 1\n")
        fake.chmod(0o755)
        code, out = self.bash('image_ident() { echo image-ident; }\nwrite_prepared_marker')
        self.assertNotEqual(code, 0, out)
        self.assertEqual(marker.read_text(), "model=valid-before\n")

    def test_existing_fields_still_change_marker(self):
        for var, value in [("ORDER", "0,1,2,3"), ("ABLIT", "1"), ("WORKER2", "w9")]:
            with self.subTest(var=var):
                self.assertChanges({var: value})


# a stand-in for the image's tf-dsv41-rank-cache: --index-only prints "[rank G]" (or fails for a rank in $FAIL_INDEX,
# with nothing on stdout, as the real one does on a tensor it cannot describe); otherwise it writes the weight file
# under the name the engine looks for, and refuses an OUT_DIR that already holds one of that rank (as the real one)
FAKE_RANK_CACHE = r"""#!BASH
g=$3; out=${2/#\/hf/$HF_CACHE}
if [[ " $* " == *" --index-only "* ]]; then
  [[ ",$FAIL_INDEX," != *",$g,"* ]] || { echo "layers.0.attn.wq_a.weight: FileNotFound" >&2; exit 1; }
  printf '[rank %s]' "$g"; exit 0
fi
echo "$g" >> "$SIM/built"
compgen -G "$out/rank${g}of4-*.bin" >/dev/null && { echo "$out already holds a weight file of rank $g" >&2; exit 1; }
printf 'weights %s' "$g" > "$out/rank${g}of4-$(printf '[rank %s]' "$g" | sha256sum | cut -c1-20).bin"
"""


class PrepareWorkerCopies(ScriptTest):
    """Step 8 of prepare.sh with RANK_LOCAL=1: the worker's folder holds only the file it reads, up to date."""

    def setUp(self):
        super().setUp()
        source = (self.repo / "scripts" / "prepare.sh").read_text()
        self.step = source[source.index("# ---------------------------------------------------------------- 8."):source.index("write_prepared_marker || die")]
        # docker run's copy, done by the shell: /hf is this Spark's HF_CACHE, /out the -v folder
        (self.tmp / "bin" / "docker").write_text("#!" + BASH + r"""
echo "$*" >> "$SIM/docker-args"
out=""; for a in "$@"; do [[ "$a" == *:/out ]] && out=${a%:/out}; done
script=${!#}; script=${script//\/hf/$HF_CACHE}; script=${script//\/out/$out}
exec """ + BASH + r""" -c "$script"
""")
        self.env["HF_CACHE"] = str(self.tmp / "hf")
        code, out = self.bash('echo "@$DATA_DIR|$WORKER_RANK_DIR"')
        self.assertEqual(code, 0, out)
        data_dir, rank_dir = out.split("@")[-1].strip().split("|")
        self.cache = Path(data_dir) / "rank-cache"
        self.cache.mkdir(parents=True)
        for n in range(4):
            (self.cache / f"rank{n}of4-new.bin").write_bytes(b"n" * (10 + n))
        self.home_dir = {h: Path(rank_dir.replace("$HOME", str(self.sim / "home" / h))) for h in ("w1", "w2", "w3")}

    def run_step(self):
        return self.bash("die() { echo \"DIED: $*\" >&2; exit 1; }\nnode_order() { echo 0,1,2,3; }\n" + self.step,
                         {"RANK_LOCAL": "1", "IMAGE": "runtime-only", "BASE_IMAGE": "base-missing"})

    def test_copies_on_the_runtime_image_and_keeps_the_time(self):
        code, out = self.run_step()
        self.assertEqual(code, 0, out)
        self.assertNotIn("base-missing", (self.sim / "docker-args").read_text())
        for i, h in enumerate(("w1", "w2", "w3"), 1):
            copy = self.home_dir[h] / f"rank{i}of4-new.bin"
            self.assertEqual(copy.read_bytes(), (self.cache / copy.name).read_bytes())
            self.assertEqual(int(copy.stat().st_mtime), int((self.cache / copy.name).stat().st_mtime))
        (self.sim / "docker-args").unlink()
        code, out = self.run_step()
        self.assertEqual(code, 0, out)
        self.assertFalse((self.sim / "docker-args").exists(), "an up-to-date copy was copied again")

    def test_an_old_weight_file_beside_the_new_one_is_removed(self):
        # two files of one node stop the rank at start (rank_cache.find: SeveralRankCaches)
        d = self.home_dir["w2"]
        d.mkdir(parents=True)
        (d / "rank2of4-old.bin").write_bytes(b"o" * 12)
        (d / "rank3of4-old.bin").write_bytes(b"o" * 13)
        (d / "rank2of4-old.bin.part").write_bytes(b"o")
        (d / "keep.txt").write_text("not a weight file")
        code, out = self.run_step()
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(f.name for f in d.iterdir()), ["keep.txt", "rank2of4-new.bin"])

    def test_a_copy_of_the_same_name_and_size_but_older_is_copied_again(self):
        d = self.home_dir["w1"]
        d.mkdir(parents=True)
        stale = d / "rank1of4-new.bin"
        stale.write_bytes(b"s" * 11)
        os.utime(stale, (1000, 1000))
        code, out = self.run_step()
        self.assertEqual(code, 0, out)
        self.assertEqual(stale.read_bytes(), b"n" * 11)


class PrepareWeightFiles(ScriptTest):
    """Step 5 of prepare.sh, the loop that keeps or rebuilds each node's weight file, run with in_hf as a local stand-in."""

    def setUp(self):
        super().setUp()
        tools = self.tmp / "tools"
        tools.mkdir()
        (tools / "tf-dsv41-rank-cache").write_text(FAKE_RANK_CACHE.replace("BASH", BASH))
        (tools / "tf-dsv41-rank-cache").chmod(0o755)
        if not shutil.which("sha256sum"):
            (tools / "sha256sum").write_text("#!/bin/sh\nexec shasum -a 256 \"$@\"\n")
            (tools / "sha256sum").chmod(0o755)
        self.env["TOOLS"] = str(tools)
        self.env["HF_CACHE"] = str(self.tmp / "hf")
        lines = (self.repo / "scripts" / "prepare.sh").read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("for g in 0 1 2 3; do")
                     and "--index-only" in "\n".join(lines[i:i + 4]))
        end = next(i for i in range(start, len(lines)) if lines[i] == "done")
        self.loop = "\n".join(lines[start:end + 1])

    def run_loop(self, fail_index=""):
        body = (
            'in_hf() { local ep; while [[ "$1" == --* ]]; do [[ "$1" != --entrypoint ]] || ep=$2; shift 2; done; shift\n'
            '  PATH="$TOOLS:$PATH" "$ep" "$@"; }\n'
            'IMAGE=image SNAP=/snap SNAP_C=/hf/snap\n'
            'mkdir -p "$DATA_DIR/rank-cache"\n' + self.loop)
        return self.bash(body, {"FAIL_INDEX": fail_index})

    def rank_files(self):
        d = next((self.tmp / "hf").glob("tensorfold-dsv41/*/rank-cache"), None)
        return sorted(f.name for f in d.glob("rank*of4-*.bin")) if d else []

    def built(self):
        f = self.sim / "built"
        return f.read_text().split() if f.exists() else []

    def test_builds_then_keeps_the_weight_files(self):
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.rank_files()), 4, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"])
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("ready"), 4, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"], "a ready weight file was rebuilt")

    def test_failed_index_keeps_that_ranks_weight_file(self):
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        before = self.rank_files()
        for rank in "0123":
            with self.subTest(rank=rank):
                code, out = self.run_loop(fail_index=rank)
                self.assertNotEqual(code, 0, out)
                self.assertIn(f"node {rank}'s weight-file index", out)
                self.assertEqual(self.rank_files(), before, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"], "a weight file was rebuilt after a failed index")


if __name__ == "__main__":
    unittest.main()
