"""Offline checks for the Engram AIO ring's io_uring backend (``0055``).

    python3 -m unittest discover -s tests -v

The engine ships as a patch series (the image build fetches the tree and apples ``zig/patches``), so these read the
patch. They pin the things that would silently rot: the default stays Linux AIO, the flag is one ``start.sh``
forwards, the ring falls back instead of failing, and the new module is standard-library only (it must build for
every target the server does, and it must not drag the CUDA modules into a host test).
"""
import os
import re
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PATCH = os.path.join(ROOT, "zig", "patches", "0055-engram-uring-reads.patch")


class EngramUring(unittest.TestCase):
    def setUp(self):
        with open(PATCH, encoding="utf-8") as f:
            self.patch = f.read()

    def test_unset_means_auto_and_named_means_exactly_that(self):
        # 1 = io_uring, 0 = Linux AIO, unset = probe the file system once and follow the answer
        self.assertIn('+        const v = std.c.getenv("TF_DS_ENGRAM_URING");', self.patch)
        self.assertIn('+        if (std.mem.eql(u8, setting, "0")) return initBackend(gpa, io, slots, .aio);', self.patch)
        self.assertIn("+        a.auto = true;", self.patch)

    def test_the_probe_asks_the_file_system_not_the_mount_name(self):
        # two reads whose completions are already there when io_submit returns mean it never queued (OpenZFS);
        # anything else (still in flight) means the AIO path is fine and cheaper
        self.assertIn("+    fn queuesAio(a: *Aio, fd: std.c.fd_t) bool {", self.patch)
        self.assertIn("+        if (ready >= 2) return false;", self.patch)
        self.assertIn("+        a.autoUpgrade(if (t.fd_wd >= 0) t.fd_wd else t.fd_w);", self.patch)

    def test_a_blocked_syscall_says_what_to_do(self):
        # the common way io_uring is "unavailable" is a container's seccomp profile faking ENOSYS for io_uring_setup;
        # the warning has to name that, or the operator is left with a depth-1 ring and no idea why
        self.assertIn("+            std.log.warn(\"engram: io_submit does not queue on this file system", self.patch)
        self.assertIn("--security-opt seccomp=unconfined", self.patch)

    def test_falls_back_instead_of_failing(self):
        # a kernel without io_uring (seccomp, or older than 5.1) must not fail a boot
        self.assertIn("error.UringUnavailable => std.log.warn(", self.patch)

    def test_flag_is_forwarded(self):
        with open(os.path.join(ROOT, "start.sh"), encoding="utf-8") as f:
            start = f.read()
        self.assertIn(r"grep -E '^(TENSORFOLD|TF_DS|TF_SSE)_[A-Z0-9_]+='", start)
        self.assertTrue("TF_DS_ENGRAM_URING".startswith("TF_DS_"))

    def test_new_module_is_stdlib_only(self):
        body = self.patch.split("+++ b/zig/src/core/uring_read.zig", 1)[1]
        body = body.split("diff --git", 1)[0]
        imports = re.findall(r'^\+const (\w+) = @import\("([^"]+)"\);', body, re.MULTILINE)
        for name, path in imports:
            with self.subTest(module=name):
                self.assertIn(path, ("std",), "uring_read.zig must import only std, or a host test drags CUDA in")

    def test_requests_are_forced_async(self):
        # without IOSQE_ASYNC io_uring tries the read inline first, and a file system that ignores IOCB_NOWAIT (the
        # same behaviour that makes its io_submit path synchronous) completes it there -- correct results, queue
        # depth 1, and no test would notice. this is the one flag that makes the backend do anything.
        self.assertIn("+            sqe.flags |= linux.IOSQE_ASYNC;", self.patch)

    def test_module_is_part_of_the_core_module(self):
        self.assertIn("+pub const uring_read = @import(\"uring_read.zig\");", self.patch)
        self.assertIn("+    _ = uring_read;", self.patch)

    def test_both_backends_are_tested(self):
        added = [l for l in self.patch.splitlines() if l.startswith("+")]
        added = "\n".join(added)
        # the new test drives the uring backend; the existing one is retargeted to aio explicitly, so neither
        # silently ends up exercising the other's path
        self.assertIn('test "io_uring rows equal the pool\'s', added)
        self.assertIn("Aio.initBackend(a, io, 32, .uring)", added)
        self.assertIn("Aio.initBackend(a, io, 16, .aio)", added)


if __name__ == "__main__":
    unittest.main()
