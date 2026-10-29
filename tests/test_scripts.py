"""Regression tests for stop.sh, ./start.sh restart, the ring's NCCL graph files under DRY_RUN=1, the prepared marker
and prepare.sh's weight-file loop.

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


class StartRestart(ScriptTest):
    """./start.sh with only what needs a Spark replaced: the links (detect_links), flock, ss, free, hostname, curl's
    answers (the server ready, the smoke test's reply) and scripts/prepare.sh (it records that it ran)."""
    FAKES = {
        "flock": "#!/bin/sh\nexit 0\n",
        "ss": "#!/bin/sh\nexit 0\n",
        "free": "#!/bin/sh\necho 'Mem: 128 1 1 0 0 120'\n",
        "hostname": "#!/bin/sh\necho 10.0.0.1\n",
        "curl": "#!/bin/sh\ncase \"$*\" in *completions*) echo '{\"choices\": [{\"message\": {\"content\": \"OK\"}}], \"usage\": {\"completion_tokens\": 1}}' ;; "
                "*) echo '{\"requests_running\": 0}' ;; esac\n",
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
