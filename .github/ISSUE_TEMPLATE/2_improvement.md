---
name: Improvement
about: Something in this recipe already works, but could be faster, safer, clearer or better defaulted.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Thank you for using this recipe!

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you want to propose an improvement to something that already exists, fill
     out the template below. For something that does not exist at all, please use
     the Feature request template instead.
-->

## What could be better

<!--
     Describe the current behavior and why it is not ideal. Examples that fit this
     repo:
       * a default in scripts/config.sh that is wrong for most people
       * a memory setting (POOL, PARALLEL, CONTEXT) that is too conservative or does not fit
       * a patch in zig/patches that no longer applies to a newer TF_REF
       * startup time (the four ~80 GB weight files build once; after that it's load time)
       * a check in tools/validate.py or tools/needle_bench.py that measures the wrong thing
-->

## Proposed change

<!--
     What should it do instead? If this touches TensorFold's engine, it belongs in a
     patch under zig/patches/ (a unified diff against BertholomusAI's TensorFold fork
     checkout, applied with `git apply` by zig/Dockerfile), and it must keep replies
     byte-identical.
-->

## Measured impact

<!--
     This recipe lives or dies on measured numbers, so please include them where you
     can. The README records results in this shape:

       * prefill tok/s and TTFT at 4k-256k tokens, and decode tok/s at 1, 2, 4 and 8
         concurrent requests, measured with tools/validate.py and the server's own
         `usage.prompt_tokens`/timing
       * a tools/needle_bench.py sweep (length x depth) PASS/FAIL grid, for anything
         touching long-context chunking or the four-node 2D split

     Before/after pairs are much more useful than a single number, and please say
     which configuration each number came from (POOL, PARALLEL, CONTEXT, which
     TF_DS_* flags) -- they move a lot between settings, and switch vs. switchless
     fabrics are not directly comparable.

     If you are comparing two configurations, please use the same sample count for
     both, and alternate restarts: boot-to-boot noise on a Spark is a few percent.
-->

## Risk

<!--
     Does this change memory use or output? A setting that raises memory can make the
     server refuse to start, or starve another node on a shared box (see the KV-pool
     sizing note near the top of the README); a patch that changes output is not
     acceptable. Say so here if it does.
-->
