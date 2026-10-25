---
name: Bug report
about: The recipe is not behaving as expected, is producing an error, or the numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this recipe!

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's "Performance" and
     "Configuration" sections list the settings that affect behavior and the defaults
     shipped in scripts/config.sh. -->

- Hardware: <!-- 4x DGX Spark (GB10, 128 GB unified memory each); switch or switchless ring? -->
- Memory available before launch: <!-- `free -h` on every Spark; other GPU workloads running on any of them? -->
- Image: <!-- `docker images | grep tensorfold-dsv41`, or `docker inspect -f '{{index .Config.Labels "tf.ref"}}' dsv41-1017-tf` -->
- Fabric: <!-- `./start.sh` (switch) or `./start-switchless.sh` (ring) -->
- `start.sh` invocation: <!-- e.g. `./start-switchless.sh`, or `POOL=4194304 ./start.sh --rebuild` -->
- Changed settings: <!-- CONTEXT, POOL, PARALLEL, DRAFTS, DERISKED, ABLIT, ENGRAM_PACKED, RDMA, any TF_DS_* flag -->
- Startup lines: <!-- `docker logs dsv41-1017-tf 2>&1 | grep -E "CUDA sm_|serving"` (rank 0); each worker's own `docker logs` for the others -->

---

## Steps to Reproduce

<!-- Please include full steps so that we can reproduce the problem. -->

1. Run `./start.sh` or `./start-switchless.sh` <!-- describe any overrides and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate the bug -->
3. ... <!-- for example "a tool call with an array parameter returns a string" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context here: a minimal request that reproduces bad output, JSON
responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.

      NOTE: the model thinks before it answers by default, in "reasoning_content" rather
      than "content". Budget enough max_tokens (1,000+ for anything non-trivial) or send
      "chat_template_kwargs": {"enable_thinking": false}: with a small budget and thinking
      on, the reply is often still in its reasoning and "content" comes back empty on a
      perfectly healthy server. That is not a bug.

      To tell a patch bug from model behavior, compare a drafted reply ("draft": true,
      the default) against an undrafted one (temperature 0, a fixed seed, DRAFTS=0 on a
      separate restart): they must be identical.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 1000,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the backticks, and mention whether it came
      from `start.sh`/`start-switchless.sh`, `scripts/prepare.sh`, `docker logs dsv41-1017-tf`
      (or a worker's own), or a client.

      Common culprits worth checking before filing:
        * "only N GiB memory available here/on <worker>" -> another GPU workload is using
          memory on that Spark, or POOL x the engine's per-position cost does not fit
          (README "ABLIT (opt-in)" and the KV-pool [!TIP] near the top cover the sizing math).
        * `start.sh` refuses port 8888 -> something else listens there; set PORT.
        * "the fabric this script serves" / a link mismatch -> you ran start.sh on a
          switchless ring, or start-switchless.sh on a switch; use the other script.
        * `scripts/prepare.sh` fails applying a patch -> zig/patches is pinned to TF_REF
          in scripts/config.sh; a patch only applies cleanly against that exact commit.
        * First start takes a few minutes -> the four ~80 GB weight files are built from
          the checkpoint once (tf-dsv41-rank-cache) and cached; that is expected, not a hang.
-->

```

```

</details>
