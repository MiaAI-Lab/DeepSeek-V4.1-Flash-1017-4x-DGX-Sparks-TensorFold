"""Offline checks for the request-body ceiling (``0054``): the flag's default is the old 32 MiB, it stays a var read
once at startup, and the name is one the launcher forwards. No GPU, weights, Docker or Sparks needed.

    python3 -m unittest discover -s tests -v

This repository ships the engine as a patch series (the image build fetches the tree and applies ``zig/patches``), so
the checks read the patch rather than a checkout.
"""
import os
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PATCH = os.path.join(ROOT, "zig", "patches", "0054-server-max-body.patch")


class BodyLimit(unittest.TestCase):
    def setUp(self):
        with open(PATCH, encoding="utf-8") as f:
            self.patch = f.read()

    def test_default_is_the_old_ceiling(self):
        # the removed line is the const; the added one is a var with the same 32 MiB, so a deployment that sets
        # nothing serves exactly as before
        self.assertIn("-pub const limit = 32 * 1024 * 1024;", self.patch)
        self.assertIn("+pub var limit: usize = 32 * 1024 * 1024;", self.patch)

    def test_ceiling_is_a_flag(self):
        self.assertIn('+    const v = std.c.getenv("TF_DS_MAX_BODY_MIB") orelse return;', self.patch)
        # whole MiB, and only a positive value replaces the default
        self.assertIn("+    if (mib > 0) limit = mib * 1024 * 1024;", self.patch)

    def test_read_before_serving(self):
        self.assertIn("+    http_body.initLimit();", self.patch)
        self.assertIn('+const http_body = @import("http_body.zig");', self.patch)

    def test_flag_namespace_is_forwarded(self):
        with open(os.path.join(ROOT, "start.sh"), encoding="utf-8") as f:
            start = f.read()
        # start.sh forwards every exported TENSORFOLD_/TF_DS_/TF_SSE_ variable to the ranks; a name outside that
        # namespace would be silently dropped
        self.assertIn(r"grep -E '^(TENSORFOLD|TF_DS|TF_SSE)_[A-Z0-9_]+='", start)
        self.assertTrue("TF_DS_MAX_BODY_MIB".startswith("TF_DS_"))

    def test_applies_last(self):
        names = sorted(n for n in os.listdir(os.path.dirname(PATCH)) if n.endswith(".patch"))
        self.assertEqual(names[-1], os.path.basename(PATCH), "the image build applies the series in glob order")


if __name__ == "__main__":
    unittest.main()
