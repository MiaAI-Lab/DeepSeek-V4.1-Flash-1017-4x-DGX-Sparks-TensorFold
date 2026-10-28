"""Offline checks: launcher locks must not escape into SSH or log followers."""
from test_scripts import ScriptTest


class LauncherLocks(ScriptTest):
    def test_worker_closes_lock_fds_only_in_child(self):
        code, out = self.bash(r'''
ssh() {
  "$BASH" -c 'if { : >&8; } 2>/dev/null; then exit 8; fi
              if { : >&9; } 2>/dev/null; then exit 9; fi'
}
exec 8>"$SIM/start.lock"
exec 9>"$SIM/prepare.lock"
worker 1 true
: >&8
: >&9
''')
        self.assertEqual(code, 0, out)

    def test_log_pipeline_closes_start_lock_in_both_children(self):
        # Exercise the actual launch line, not a second copy of the fixed pipeline.
        pipeline = next(line for line in (self.repo / "start.sh").read_text().splitlines()
                        if line.startswith('docker logs -f "$CONTAINER_NAME"'))
        code, out = self.bash(r'''
probe='if { : >&8; } 2>/dev/null; then echo inherited; else echo closed; fi'
docker() { "$BASH" -c "$probe" >"$SIM/docker.fd"; }
sed() { "$BASH" -c "$probe" >"$SIM/sed.fd"; cat; }
exec 8>"$SIM/start.lock"
D= R=
''' + pipeline + '\nwait "$!"\n: >&8\n')
        self.assertEqual(code, 0, out)
        for name in ("docker", "sed"):
            self.assertEqual((self.sim / (name + ".fd")).read_text().strip(), "closed", name)
