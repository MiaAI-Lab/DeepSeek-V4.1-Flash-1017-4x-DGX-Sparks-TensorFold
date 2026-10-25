## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this recipe, a good description usually covers:
      * which setting in scripts/config.sh, script behavior or patch changes
      * whether the change affects measured numbers (prefill/decode throughput, TTFT,
        concurrency, the KV pool's memory cost) and in which direction
      * for a patch: why the output stays byte-identical

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this recipe that usually means:

      * `bash -n start.sh start-switchless.sh stop.sh scripts/*.sh` (syntax check)
      * `python3 -m py_compile tools/*.py`
      * `shellcheck start.sh start-switchless.sh stop.sh scripts/*.sh` if available
      * `scripts/prepare.sh` (a patch change rebuilds the image; every patch must apply
        with `git apply` against BertholomusAI's TensorFold fork checkout, `zig/Dockerfile`)
      * an actual launch, plus
        `docker logs dsv41-1017-tf 2>&1 | grep -E "CUDA sm_|serving"`
      * `tools/validate.py` and, for anything touching long-context chunking or sharding,
        `tools/needle_bench.py` against the running server
      * if behavior changed, the measured numbers with the new settings, stating
        which configuration they came from (see README "Performance")

    If you changed a patch, confirm replies are byte-identical to before: the same
    prompt with a fixed seed and temperature 0, or comparing drafted against undrafted
    (`DRAFTS=0`).

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (script syntax check, `scripts/prepare.sh`, a launch, or a re-measurement).
- [ ] A patch change keeps every reply byte-identical, and I said how I checked it.
- [ ] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [ ] If my change affects memory, I checked it against a live `free -h` / the engine's own startup line on all four Sparks, not just one.
- [ ] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
