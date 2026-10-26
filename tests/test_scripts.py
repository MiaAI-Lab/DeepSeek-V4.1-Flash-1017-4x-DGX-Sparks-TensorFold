"""Regression tests for stop.sh, the ring's NCCL graph files under DRY_RUN=1, and the prepared marker.

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
# fake ssh: ssh [-o opt]... host command...; a host listed in $SIM/unreachable cannot be reached
while [[ "$1" == -o ]]; do shift 2; done
host=$1; shift
echo "$host $*" >> "$SIM/ssh.log"
grep -qxF "$host" "$SIM/unreachable" 2>/dev/null && { echo "ssh: connect to host $host: No route to host" >&2; exit 255; }
mkdir -p "$SIM/home/$host"
SIM_HOST=$host HOME="$SIM/home/$host" exec BASH -c "$*"
"""

FAKE_DOCKER = r"""#!BASH
# fake docker: each Spark's containers are lines in $SIM/hosts/<host>/containers; a container named in
# $SIM/hosts/<host>/stuck cannot be stopped or removed
h=${SIM_HOST:-head}; d="$SIM/hosts/$h"; mkdir -p "$d"; touch "$d/containers"
echo "$h docker $*" >> "$SIM/docker.log"
has() { grep -qxF "$1" "$d/containers"; }
stuck() { grep -qxF "$1" "$d/stuck" 2>/dev/null; }
last=${!#}
case "$1" in
  ps) cat "$d/containers" ;;
  inspect) has "$last" || { echo "Error: No such object: $last" >&2; exit 1; }
           [[ "$*" != *State.Running* ]] || echo true ;;
  stop) stuck "$last" && { echo "Error response from daemon: cannot stop container: $last" >&2; exit 1; }; true ;;
  rm) stuck "$last" && { echo "Error response from daemon: cannot remove container: $last" >&2; exit 1; }
      grep -vxF "$last" "$d/containers" > "$d/c.new" || true; mv "$d/c.new" "$d/containers" ;;
  logs) echo "a log line" ;;
  image) exit 1 ;;
  *) exit 0 ;;
esac
"""

FAKE_CURL = """#!/bin/sh
echo '{"requests_running": 0}'
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

    def test_dry_run_stops_nothing(self):
        self.all_up()
        code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("nothing is stopped", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [NAME], h)


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

    def test_nfs_volume_changes_marker(self):
        self.assertChanges({"NFS_VOLUME": "other-volume"})

    def test_each_nfs_server_changes_marker(self):
        for var in ["NFS_SERVER", "NFS_SERVER2", "NFS_SERVER3"]:
            with self.subTest(var=var):
                self.assertChanges({var: "10.9.9.9"})

    def test_chat_template_changes_marker(self):
        tpl = self.repo / "model" / "chat_template.jinja"
        self.assertChanges(edit=lambda: tpl.write_text(tpl.read_text() + "\n{# changed #}\n"))

    def test_existing_fields_still_change_marker(self):
        for var, value in [("ORDER", "0,1,2,3"), ("ABLIT", "1"), ("WORKER2", "w9")]:
            with self.subTest(var=var):
                self.assertChanges({var: value})


if __name__ == "__main__":
    unittest.main()
