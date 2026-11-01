# Changelog

- **2026-11-01** -- `0047`-`0050`: the head node's host memory over days, and two small engine fixes from the same
  tree. Everything here was measured on a four-GB10 switchless ring running this recipe in production; none of it
  changes a default, and none of it is about throughput (a build with and without these four measured the same
  prefill and decode within noise — that is the honest result).
  * `0047` -- `prepare()` no longer renders the whole history a second time just to compute `history_len`, and
    `str + str` in the template evaluator resizes the arena's last block in place instead of `std.mem.concat`.
    The two together are what turned a 500k-token agent session into a 3.4-6.6 GB anonymous block on the head:
    the second render produced the prompt text twice into a per-request arena, and each `{% set out = out + … %}`
    reallocated and copied the whole prefix. Long sessions now peak at 231/191/142 MB instead of GBs.
  * `0048` -- the keep-alive connection arena is capped at 1 MiB (`retain_with_limit` instead of
    `retain_capacity`), render scratch allocates from the template's own arena, and the tokenizer's
    encode/decode scratch runs on a 4 KiB stack. Without these, a connection kept the arena of the largest
    request it had ever seen, and every render/encode left its scratch in it.
  * `0049` -- host row tables hold 8192 rows (inert at the default `chunk_rows = 2048`), and two opt-in decode
    switches: `TF_DS_SOLO_OVERHEAD` (fixed head-window overhead instead of a live estimate; **measured -9.3 %
    single-stream decode when set with `DRAFT_MS=25`, so it defaults off) and `TF_DS_NO_COPIES` (suffix-lookup
    copy proposals, off = matches the Python reference).
  * `0050` -- a refused Engram row id no longer leaks the AIO batch's buffers.

  Measured on the ring, 12M-position pool + 6 GiB snapshot store + `--learn`, four nodes: `device_memory`
  100,679,832,360 B with a peak of 100,686,517,032 B (**6.5 MB of growth since start**), head `AnonPages`
  flat at 5.0-5.2 GB over hours, `MemAvailable` 12.0-12.4 GB, and the largest anonymous block 231 MB.
  Before these patches the same stack reached 22.7-23.8 GB anon and died in the kernel's global OOM killer.
  Apply order: they expect the whole upstream series (0001-0036 + 0039-0046 + 0037) already applied; `0050`
  is independent, `0049`'s `native.zig` hunk needs the `errdefer st.lanes.deinit()` context `0034` adds.

- **2026-11-01** -- README notes from three issues, none of them changing a default: `TF_DS_RDMA_CPUS` and the GB10's
  core layout ([making](https://github.com/making), [#40](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/40));
  what `__vm_enough_memory` in `dmesg` means ([#36](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/36));
  a worker's NFS path across the ring ([n1ur0](https://github.com/n1ur0), [#21](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/21)).
- **2026-11-01** -- `MASTER_ADDR` is derived from a Tailscale address (`100.64.0.0/10`) only when the Spark has no other
  ([n1ur0](https://github.com/n1ur0), [#21](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/21)).
  `worker_nfs` runs on the runtime image too, the last utility left on `BASE_IMAGE` after Tyler Folkman's change below.
- **2026-10-31** -- `0037`, `NVFP4=1`: [NVIDIA's NVFP4 release](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)
  of the model. Routed experts on GB10's block-scaled FP4 MMA (`mma_fp4`, Ash Hart's in TensorFold) with each expert's own
  4-bit activations (`tf_dsn_nv_quant`, TensorFold's format), MXFP4 Engram rows decoded on the host, the DSpark stages and
  the shared expert unchanged; weight files and GPU memory the size of DeepSeek's. `check_nvfp4.py` on real tensors:
  weights equal the source's MXFP4 exactly, quantizer bit-exact, decode and prefill kernels within 4e-10 of fp64, row
  bits independent of the launch. Against the FP8 image then served: prefill +4.3 to +6.2 %, decode +1.1 % at c1, +0.3 % at
  c2, +1.7 % at c4, +3.9 % at c8, +6.4 % at c16 ([README](README.md#nvfp4-opt-in)); `tools/validate.py` all passed, 612 replies
  equal at every concurrency.
- **2026-10-29** -- (Tyler Folkman, tfolkman) The launcher's lock descriptors stay out of persistent children: a surviving
  ssh or `docker logs -f` could hold `start.lock` or `prepare.lock` after its launcher exited and block the next start
  (`tests/test_launcher_locks.py`). On a switch, NCCL's bootstrap uses a RoCE subnet every rank shares, chosen the same way
  on all of them, instead of each node's default route; rings and triangles keep the default route, and `SOCKET_IFNAME`
  still overrides it (`tests/test_mesh_bootstrap.py`; not run on this ring). The preparation utilities run on the
  runtime image, so a worker holding it never pulls the build base (`tests/test_prepare_image.py`).
- **2026-10-29** -- (SvangenStudios) `scripts/prepare.sh` stops when it cannot compute a node's weight-file index
  instead of deleting that node's weight file. The index was hashed inside the container's own `bash -c`, which the
  script's `pipefail` does not reach, so a failed `tf-dsv41-rank-cache --index-only` (a missing or unreadable shard)
  hashed empty output, matched no weight file, and the loop deleted the node's good one (~80 GB) before a rebuild that
  failed the same way. Regression tests in `tests/test_scripts.py` (`PrepareWeightFiles`).
- **2026-10-29** -- (Christopher A. Williamson) [`docs/engram-zfs-dataset.md`](docs/engram-zfs-dataset.md): what an Engram host's ZFS dataset has
to be set to, measured on one GB10 of this cluster -- `recordsize=4K`, `encryption=off`, `compression=off`,
`primarycache=metadata` -- with the row counts behind each one. A pool's defaults (`recordsize=128K`, `zstd`) made every
row read pull a whole compressed record: about 500 rows a second, and over 500 seconds to prefill a 16k-token prompt.
The recipe itself is unchanged.

- **2026-10-28** -- (shalk3) Each worker's weight file from its own disk, opt-in with `RANK_LOCAL=1`: `scripts/prepare.sh`
  copies the ~77-81 GB file a worker reads there, and `start.sh` mounts the copy when its name, size and modification
  time match spark1's, else that worker reads over NFS as before, with a warning. On a four-GB10 ring whose spark3
  mounts over a 2.5 GbE LAN (the others over CX7), the engine's load fell from 315 s to 96 s and a start from
  mains-off to serving from 417 s to 197 s; the replies are unchanged (the same file, read from another disk). The workers need the space:
  one short of it keeps NFS. `tests/test_scripts.py` covers a matching copy, a partial one, none, and `RANK_LOCAL=0`.
- **2026-10-28** -- (SvangenStudios) `stop.sh` reads the native server's `/health` schema: `live.connections`, the
  requests running and waiting. It used to read `requests_running`, which this server does not publish, so it never
  warned before cutting requests off. It takes the count only from an HTTP 200 with no `error` (an explicit `null` is
  none); a `/health` it cannot read now warns that the open requests are unknown instead of reading as none.
  `start.sh`'s smoke test passes only on HTTP 200 with the assistant's reply and no `error` (again, `null` is none); an
  HTTP 500 or an error reply shaped like a completion used to end in "is now LIVE". Regression tests in
  `tests/test_scripts.py`, `tests/test_start_smoke.py` and `tests/test_stop_health.py` (the real curl against a loopback
  HTTP server).
- **2026-10-29** -- (SvangenStudios) `start.sh` puts an IPv6 address in brackets in the URLs it builds: the one
  its readiness and smoke checks call, and the API address it prints when the server is live. `HOST=::1` used to give
  `http://::1:8888`, which curl cannot call. The engine's own `--host` stays the raw address, and `0.0.0.0` and `::`
  still map to `127.0.0.1` for the checks. Regression tests in `tests/test_start_urls.py`.
- **2026-10-29** -- `0039` (SvangenStudios): when a request's reply grammar could not be built at admission (xgrammar
  compile error or out of memory), `LaneHost.setup` dropped and freed the job but returned `drop`'s `true`, so both
  callers went on with the freed job; it now returns `false`, and a grammar built before a failed `Stream.init` is
  freed. CPU fault-injection and lifecycle tests: 24 of 24 with the patch, 12 fault paths failing without it; live on
  this cluster with `0040`-`0043` proposed alongside: `tools/validate.py` 8 of 8 equal, structured and tool-call
  replies unchanged. The failure path itself was not caused on the GPU.
- **2026-10-29** -- `0040` (SvangenStudios): once a decode or drafter graph executable enters its cache, the cache
  keeps ownership even when its first launch fails (the local `errdefer exec.deinit()` destroyed it and left it in
  the map: a later hit launched a destroyed handle and teardown destroyed it twice). Capture agreement, cache
  insertion and launch order are unchanged; no retry is added. CPU tests: 20 of 20 with the patch, 6 failing without
  it; real-GPU cache probes on four GB10 and an A/B/A run through the full TP4 model (52 requests per phase,
  identical token IDs). Recovery from a lost CUDA context is not claimed.
- **2026-10-29** -- `0041` (SvangenStudios): the engine's server tests (`zig test` on `zig/src/server/root.zig`)
  compile again: two opt-in tests in `tool_grammar.zig` still called `HfText.load` with the removed template argument
  and a removed `encodePlain` (also noted in #20). Test code only; no production code or tokenizer API changes. On
  CPU: compile error before, 70 passed and 2 skipped (the checkpoint tests) after.
- **2026-10-29** -- `0042` (SvangenStudios): a JSON answer (`response_format` `json_object`/`json_schema`,
  `guided_json`, the matching `structured_outputs` form) stays JSON when the request also offers `tools`: it is no
  longer parsed as tool calls, hidden or Harmony-post-processed; the tools stay in the prompt. On this cluster 8 of 25
  structured and tool-call checks failed before (the JSON came back as `tool_calls`, whole replies with
  `content: null`) and 25 of 25 pass with it; ordinary tool calls, `required`/named and the 400 for schema +
  `required` are unchanged. CPU: 132 of 132 JSON answers preserved over 288 handler observations.
- **2026-10-29** -- `0043` (SvangenStudios): a reply made only of DSML closing tags (the form reported in #14), with
  tools offered, gets `tensorfold.tool_parse_warning: "orphan_dsml_closers"` (whole replies and the final SSE event)
  and a log line with the request id. Diagnostic only: content, token IDs, usage and finish reason are unchanged, no
  call is invented and no retry started. It does not fix #14. CPU: 42 exact warning cases and ten negatives; live:
  no warning on 25 ordinary, JSON and tool-call replies.
- **2026-10-28** -- `0046` (shalk3), the weight loader's read-ahead. A rank's weight file holds its tensors in the loader's
  order and the loader read them one at a time, so every read waited on the disk: ~1 GB/s from an NVMe that reads
  9 GB/s in parallel. Eight threads now `pread` the file in 32 MiB chunks up to 4 GiB past the loader's furthest
  read and drop the bytes, so the loader's own reads come from the page cache; `TF_DS_LOAD_PREFETCH=0` turns it off.
  Measured with `tf-dsv41-load` on one worker (rank 1, 81.2 GB, 1,578 tensors, the file evicted from the page cache
  before each run): 83.4 s before, 27.2 / 27.3 / 27.4 s with it; the whole file pre-read into the page cache first
  took 48.0 s (the cache and the landing weights compete for memory, hence the bounded lead). Every weight tensor's
  sha256 is equal to the old loader's. On the ring: greedy replies byte-identical (3 prompts x 300 tokens) and the
  same decode and prefill speed. Over NFS the network, not the reads, is the limit: it pays off with the weight file
  on the worker's own disk.
- **2026-10-28** -- `0036`, structured output and required tool calls
  ([#6](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/6)): upstream TensorFold's
  structured output (BobClawblaw's lane-core grammar and libtfgrammar, Ash Hart's required tool calls and grammar-aware
  drafting) and BertholomusAI's DeepSeek-V4.1 hooks and DSML tool-call grammar (bertholomus), ported onto this pin; the
  image builds libtfgrammar from xgrammar v0.2.8. `response_format` (`json_schema`, `json_object`), the `guided_*`
  fields and `tool_choice` "required" or a named function used to return HTTP 400; now every token is drawn among the
  tokens the grammar allows. Tested on this cluster: 15 of 15 checks (a JSON schema greedy, sampled and with thinking,
  `json_object`, `guided_regex`, `guided_choice`, a streamed schema, `tool_choice` required with and without thinking, a
  named function, `auto` as before, a malformed regex refused with xgrammar's reason, a plain reply beside a structured
  one equal to its solo run); eight requests at once, half structured, all valid; `tools/validate.py` passed with its 8
  greedy replies equal to the saved ones; decode speed unchanged (61 tok/s on one stream). One structured stream alone
  runs about as fast as the same reply unconstrained (25 cities as a JSON schema: 108 tok/s greedy and 121 sampled,
  against 104 and 109).
- **2026-10-28** -- `model/chat_template.jinja` writes each piece out as it is made instead of appending to one string
  with `{% set ns.out = ns.out + ... %}` (hashd1ve's fix from the two-Spark recipe, its issue #1 and `fd2cdb2`, carried
  over byte for byte by manuel1618 in #18, for #17). The engine's Jinja copies the whole string on every `+` and keeps
  each copy until the render ends, and every request renders twice, so rank 0's host memory grew with messages x prompt
  length: in the engine's own template module one render of a 1,500-turn conversation (1.6 MB of prompt) peaked at 24.8
  GB, now 0.4 GB. Prompts are unchanged: 14,688 conversations render byte-identically before and after, in the engine's
  template module and in jinja2. On this cluster: `/tokenize` of a 500-turn agent conversation (108k tokens) gave the
  same ids with both templates while rank 0's memory rose 1,967 MiB before and 110 MiB after (200 turns: 349 and 36
  MiB); `tools/validate.py` passed with its 8 greedy replies equal to the saved ones, and 5,000 random conversations
  rendered identically in jinja2.
- **2026-10-28** -- `0035` (Mia's AI Lab), the ranks' startup rendezvous
  ([#3](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/3)). Rank 0 used to open
  its ports for workers 1, 2 and 3 one after another, each only once the worker before it had connected, and only
  after its own weights had loaded; a worker retries its connect for 300 s after loading its own weights. So a worker
  that loaded more than 300 s before the ones ahead of it exited with `error: ConnectFailed` (in #3: rank 3 loaded in
  113 s, ranks 1 and 2 in ~843 s over NFS). Rank 0 now opens all three ports when it starts, before its weights load,
  and a worker that connects early waits in its port's backlog. `TF_DS_CONNECT_S` sets a worker's retry window (300 s
  by default; it now covers only rank 0's own start), and a worker that gives up logs the address and port it tried.
  Diagnosed by shanedertrain in #3, reported by tlacerdap. Tested on this cluster with rank 0 and rank 3 started
  together and ranks 1 and 2 nine minutes later: before `0035` rank 3 exited with `ConnectFailed` 427 s in; with it,
  rank 3's connection waited on rank 0's port and the server answered 696 s in.
- **2026-10-28** -- seven pull requests merged (#2, #4, #5, #8, #10, #13, #15), tested together on this cluster. Three
  of them had named their patch `0027`: #10's keeps the number, #8's is `0033` here and #2's `0034`. Against the image
  before them (`0001`-`0025`), on four GB10s with this cluster's switches: `tools/validate.py` passed and its 8 greedy
  replies were equal; a 131k-token prompt read cold in 34.7 s as before; a follow-up resumed from the previous turn's
  state gave the same tokens as the same prompt read cold (8 of 8); a 35-image prompt resent read in 1.0 s instead of
  20.1 s, its tokens equal; a 12-screenshot history resumed on every turn, and a changed or swapped picture resumed none
  of what followed it; a 9-token request arriving 1 s into a cold 50,939-token prompt had its first token in 0.3 s; idle
  CPU fell from ~310 % to 0.15-0.35 % a rank.
- **2026-10-28** -- `0032` (Mia's AI Lab): `0031`'s bounded admission only with `TF_DS_PREFILL_YIELD=1` (on, this
  cluster). Without the switch a prompt is read whole, as before `0031`, and its Engram read-ahead keeps running
  between its chunks.
- **2026-10-28** -- `0030`, `0031` (Mauvis Ledford): `0030` skips vision work covered by a restored image-keyed prefix;
  `0031` admits and advances requests between unchanged prompt chunks. Partially filled streams keep separate progress
  and image rows on the head, and all TP ranks select the same slot when resuming. No arithmetic or checkpoint
  settings change.
- **2026-10-28** -- `0028`, `0029` (upstream TensorFold, backported by Mauvis Ledford): native SSE keepalives, every 15
  seconds during silent streaming generations (Chat/Responses use comments; Messages uses ping events;
  `TF_SSE_KEEPALIVE_S=0` disables them), and the golden test server's stdout drained so a full log pipe cannot stall the
  regression suite. Authorship and source commits are recorded in `CREDITS.md`.
- **2026-10-27** -- `0033` (manuel1618, its PR's `0027`): `/metrics` publishes `prompt_tokens_cached_total`, the prompt
  tokens of finished requests restored from kept prompt states, and repeats the split under vLLM's
  `prefix_cache_hits_total` / `prefix_cache_queries_total`. Before, `/metrics` had only `prompt_tokens_total`, which
  counts cached and computed prompt tokens alike, so a dashboard could not tell them apart (a 93 %-cached agent workload
  read as all computed). The per-request `usage.prompt_tokens_details.cached_tokens` is unchanged. Backport of
  BertholomusAI's `ce0d9053` to this recipe's engine pin, plus the one line that turns the series on for the
  DeepSeek-V4.1 engine.
- **2026-10-27** -- `0027` (Thomas Kempka), the image-keyed prompt cache: a kept prompt state is keyed by the request's
  images as well as its tokens, so a chat whose history holds a picture (a screenshot in a tool result, say) resumes
  its prefill instead of re-running it from 0 on every turn. Measured on 4x GB10 TP4 (16 streams, 2.8k tok/s prefill):
  one 150k-token prompt replayed in 0.64 s (82x) without an image and 53.6 s (1.0x) with one before this patch; both
  replay now. Text-only requests are unchanged (the key is 0 for every prefix, so the tokens alone decide), a state
  kept for a picture is never learned to disk, and it is not resumed by a prompt whose prefix holds no image.
- **2026-10-26** -- `0034` (uchkw, its PR's `0027`): park the three RDMA proxy threads when no streams are live.
  Synchronize CUDA on every rank, agree READY/COMMIT, then block in futex waits; wake before compute and before
  stop/join. Idle followers block on the control link instead of spinning. A four-GB10 ring measured about 300% CPU per
  node before and 0.21-0.32% after; three idle/wake chat cycles passed. Numerical kernels and weights are unchanged.
  All ranks must run the updated image (new idle control frame). Ported onto `0031` (its frame value is 11).
- **2026-10-26** -- (SvangenStudios) `stop.sh` exits 1 when a rank may still run (a rank it could not stop, or a Spark
  whose containers it could not list); `DRY_RUN=1` writes no NCCL graph files; the prepared marker also covers
  `HF_CACHE`, the NFS settings and the chat template, and a state that cannot be computed never reads as ready. Offline
  regression tests in `tests/test_scripts.py` (standard library only, no Spark).
- **2026-10-26** -- (SvangenStudios) `tools/validate.py`, `tools/needle_bench.py`: a check passes only on a real
  completion. An HTTP error, an `error` object, a reply that is not the assistant's, a malformed tool call or a short or
  empty `--compare` reference now fails instead of passing; bad arguments stop with a message. Offline regression tests
  in `tests/test_validation_tools.py` (standard library only, no GPU).
- **2026-10-26** -- `0026` (knapcio): the RDMA rings' queue pairs take the port's active MTU (`ibv_query_port`,
  capped at 4096) as their path MTU instead of a fixed 4096. On a switched fabric whose Ethernet MTU is 1500 the RoCE
  active MTU is 1024, and with the fixed 4096 every request failed (`RDMA=1`); with the port's MTU it serves. On a
  fabric whose active MTU is 4096 the value is the same as before.
- **2026-10-25** -- `ABLIT` (new, off by default): grafts drowzeys's Keys abliteration overlay (`attn.wo_b`,
  layers 10-35) onto the checkpoint before the rank caches are baked, applied via a new checkpoint view rather
  than the vendor's own index-remap script (our engine never reads `model.safetensors.index.json`). Live-validated:
  tensor-level proof the grafted bytes match the sidecar, `validate.py` all passed, needle test passed at 65k/262k.
- **2026-10-25** -- `tools/needle_bench.py` (new): a needle-in-a-haystack retrieval sweep, 4k to ~997k tokens x
  0/25/50/75/100% depth. 45 of 45 cells passed.
- **2026-10-25** -- WP6(b), `TF_DS_IDX_SHARD` (`0025`): the indexer scores and picks only its own owned quarter of a
  chunk's rows, instead of every node redundantly scoring all of them. On by default (`scripts/local.sh`).
- **2026-10-25** -- resumable prefill, expert early exit, draft-row costs (`0022`-`0024`, opt-in): `TF_DS_PREFILL_YIELD`
  and `TF_DS_EXPERT_EXIT` on (this cluster); `TF_DS_DRAFT_ROW_MS` measured mixed, kept off.
- **2026-10-25** -- the decode plan's Gate 0 and draft economics (`0017`-`0021`, `ROUND_MS`, `DRAFT_MS`): 64-row
  drafting slabs fix 13+ concurrent drafting streams failing outright; CUDA graph capture agreed across every rank;
  PDL controls (off); `ROUND_MS`/`DRAFT_MS` set from measurement; `TF_DS_SAMPLE_CAND` opt-in.
- **2026-10-24** -- WP5, `TF_DS_OWNER_GATHER` (`0016`): the quarter-HC gather moves only each rank's owned quarter
  over the column link, instead of the full chunk. On by default.
- **2026-10-24** -- WP4, `TF_DS_QUARTER_HC` (`0015`): each node computes the HC-mix step for only its own quarter
  of a chunk, then all-gathers the result. On by default.
- **2026-10-24** -- WP6(a), `TF_DS_IDX_RB` (`0014`): the indexer's prompt-chunk scoring kernel at the kit's own RB=2
  capture instead of RB=1. Off by default (opt-in).
- **2026-10-24** -- WP2, projection-tile tuning (`0013`, Hector's measured tuning): a better tile config for the
  K-split dense FP8 GEMMs. On by default.
- **2026-10-24** -- WP0 4.3b (`0012`): an offline replay tool proves the HC-mix kernels are row-independent, the
  entry gate WP4 needed.
- **2026-10-24** -- pair-sum hardening (`0009`-`0011`): `TF_DS_PAIR_SUM`'s startup agreement hardened, Engram
  worker-wait timing, a multi-layer tensor dump.
- **2026-10-23** -- guarded live-trial tooling for the prefill work (removed later, once the work was done).
- **2026-10-23** -- pair-sum gathers, opt-in (`0008`, Hector): sums each column's two TP2 partials over the cable
  between the partners and swaps with the other pair, instead of a four-quarter all-gather.
- **2026-10-18** -- reasoning effort `max` by default; aligned indexer row blocks (`0006`, `0007`).
- **2026-10-18** -- a resumed chunk's positions copied to a 16-byte boundary (`0005`).
- **2026-10-18** -- a general Triton variant when no exact one was captured (`0004`).
- Engram's packed rows, opt-in (`0003`): off by default, not yet measured.
- **2026-10-18** -- the four speed agents' work merged: native FP8/MXFP4 kernels, the ring's node order and links,
  video input, tool-call arguments as objects.
- **v0.1.0 (unreleased)** -- first version: DeepSeek-V4.1-Flash-1017 on four DGX Sparks, TP4, on BertholomusAI's
  DeepSeek-V4.1-Flash family of TensorFold's native engine.
